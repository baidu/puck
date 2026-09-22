// Copyright (c) 2026 Baidu, Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>

#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <gflags/gflags.h>

#include "interface/index.h"
#include "pyapi_wrapper/py_api_wrapper.h"
#include "quantizer/FLAT.h"
#include "quantizer/LVQ.h"
#include "quantizer/factory.h"
#include "tinker/tinker_index.h"

namespace puck {
namespace {

constexpr size_t TEST_DIM = 128;
constexpr size_t LVQ_TRAIN_VECTOR_COUNT = 256;
constexpr size_t DATASET_BASE_COUNT = 10000;
constexpr size_t RECALL_QUERY_COUNT = 10;
constexpr uint32_t RECALL_TOPK = 100;
constexpr size_t FILTER_REGION_COUNT = 8;
constexpr uint32_t FILTER_TOPK = 20;
constexpr int64_t TEST_MEMORY_CAPACITY = 4 * 1024 * 1024;
constexpr int64_t DATASET_MEMORY_CAPACITY = 400LL * 1024 * 1024;

std::filesystem::path dataset_file(const std::string& name) {
    const std::filesystem::path path = std::filesystem::path(PUCK_TEST_DATA_DIR) / name;
    if (!std::filesystem::is_regular_file(path)) {
        throw std::runtime_error("SIFTsmall data file does not exist: " + path.string());
    }
    return path;
}

std::vector<std::vector<float>> read_fvecs(const std::filesystem::path& path,
                                           size_t count) {
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("failed to open fvecs file: " + path.string());
    }
    std::vector<std::vector<float>> vectors;
    vectors.reserve(count);
    for (size_t row = 0; row < count; ++row) {
        int32_t dim = 0;
        input.read(reinterpret_cast<char*>(&dim), sizeof(dim));
        if (!input || dim != static_cast<int32_t>(TEST_DIM)) {
            throw std::runtime_error("invalid fvecs dimension or truncated header: " + path.string());
        }
        std::vector<float> vector(TEST_DIM);
        input.read(reinterpret_cast<char*>(vector.data()),
                   static_cast<std::streamsize>(vector.size() * sizeof(float)));
        if (!input) {
            throw std::runtime_error("truncated fvecs vector: " + path.string());
        }
        vectors.push_back(std::move(vector));
    }
    return vectors;
}

std::vector<std::vector<uint32_t>> read_ivecs(const std::filesystem::path& path,
                                              size_t count) {
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("failed to open ivecs file: " + path.string());
    }
    std::vector<std::vector<uint32_t>> vectors;
    vectors.reserve(count);
    for (size_t row = 0; row < count; ++row) {
        int32_t dim = 0;
        input.read(reinterpret_cast<char*>(&dim), sizeof(dim));
        if (!input || dim <= 0 || dim > 10000) {
            throw std::runtime_error("invalid ivecs dimension or truncated header: " + path.string());
        }
        const std::streampos data_position = input.tellg();
        input.seekg(0, std::ios::end);
        const std::streampos end_position = input.tellg();
        input.seekg(data_position);
        if (data_position < 0 || end_position < data_position
                || static_cast<uint64_t>(end_position - data_position)
                        < static_cast<uint64_t>(dim) * sizeof(uint32_t)) {
            throw std::runtime_error("invalid ivecs length: " + path.string());
        }
        std::vector<uint32_t> vector(static_cast<size_t>(dim));
        input.read(reinterpret_cast<char*>(vector.data()),
                   static_cast<std::streamsize>(vector.size() * sizeof(uint32_t)));
        if (!input) {
            throw std::runtime_error("truncated ivecs vector: " + path.string());
        }
        vectors.push_back(std::move(vector));
    }
    return vectors;
}

void write_fvecs(const std::filesystem::path& path,
                 const std::vector<std::vector<float>>& vectors) {
    std::ofstream output(path, std::ios::binary);
    if (!output.is_open()) {
        throw std::runtime_error("failed to create fvecs file: " + path.string());
    }
    const int32_t dim = static_cast<int32_t>(TEST_DIM);
    for (const std::vector<float>& vector : vectors) {
        if (vector.size() != TEST_DIM) {
            throw std::runtime_error("unexpected vector dimension while writing fvecs");
        }
        output.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
        output.write(reinterpret_cast<const char*>(vector.data()),
                     static_cast<std::streamsize>(vector.size() * sizeof(float)));
    }
    if (!output.good()) {
        throw std::runtime_error("failed to write fvecs file: " + path.string());
    }
}

class FlagGuard {
public:
    explicit FlagGuard(std::vector<std::string> names) {
        for (const std::string& name : names) {
            std::string value;
            if (google::GetCommandLineOption(name.c_str(), &value)) {
                _values.emplace_back(name, value);
            }
        }
    }

    ~FlagGuard() {
        for (const auto& value : _values) {
            google::SetCommandLineOption(value.first.c_str(), value.second.c_str());
        }
    }

    FlagGuard(const FlagGuard&) = delete;
    FlagGuard& operator=(const FlagGuard&) = delete;

private:
    std::vector<std::pair<std::string, std::string>> _values;
};

class TemporaryDirectory {
public:
    explicit TemporaryDirectory(const std::string& test_name)
        : _path(std::filesystem::temp_directory_path() /
                ("puck_tinker_" + test_name)) {
        std::error_code error;
        std::filesystem::remove_all(_path, error);
        error.clear();
        if (!std::filesystem::create_directories(_path, error) || error) {
            throw std::runtime_error("failed to create temporary directory: " + _path.string());
        }
    }

    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(_path, error);
    }

    const std::filesystem::path& path() const {
        return _path;
    }

    TemporaryDirectory(const TemporaryDirectory&) = delete;
    TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;

private:
    std::filesystem::path _path;
};

std::vector<std::string> tinker_flag_names() {
    return {"index_path", "index_file_name", "feature_file_name",
            "brief_file_name", "feature_dim", "whether_norm", "threads_count",
            "tinker_file_prefix", "tinker_idmapper_file_name",
            "tinker_quantizer_file_name", "tinker_quantizer_type",
            "tinker_neighborhood_count", "tinker_ef_construction",
            "tinker_memory_capacity", "tinker_brief", "tinker_cartesian"};
}

void set_flag(const char* name, const std::string& value) {
    ASSERT_FALSE(google::SetCommandLineOption(name, value.c_str()).empty());
}

void configure_tinker(const std::filesystem::path& path,
                      bool brief, bool cartesian) {
    set_flag("index_path", path.string());
    set_flag("index_file_name", "index.dat");
    set_flag("feature_file_name", "features.fvecs");
    set_flag("brief_file_name", "labels.txt");
    set_flag("feature_dim", std::to_string(TEST_DIM));
    set_flag("whether_norm", "false");
    set_flag("threads_count", "2");
    set_flag("tinker_file_prefix", "tinker");
    set_flag("tinker_idmapper_file_name", "idmapper.dat");
    set_flag("tinker_quantizer_file_name", "quantizer.dat");
    set_flag("tinker_quantizer_type", "FLAT");
    set_flag("tinker_neighborhood_count", "4");
    set_flag("tinker_ef_construction", "16");
    set_flag("tinker_memory_capacity", std::to_string(TEST_MEMORY_CAPACITY));
    set_flag("tinker_brief", brief ? "true" : "false");
    set_flag("tinker_cartesian", cartesian ? "true" : "false");
}

InsertRequest make_insert_request(const std::vector<float>& feature,
                                  size_t local_id,
                                  const std::vector<std::string>* labels) {
    InsertRequest request;
    request.feature = feature.data();
    request.dim = static_cast<uint32_t>(feature.size());
    request.local_id = local_id;
    request.labels = labels;
    return request;
}

Request make_search_request(const std::vector<float>& feature, uint32_t topk,
                            const std::vector<std::string>* labels) {
    Request request;
    request.feature = feature.data();
    request.topk = topk;
    request.ef_search = 32;
    request.labels = labels;
    return request;
}

void expect_ids(const std::vector<uint32_t>& actual, uint32_t result_num,
                std::vector<uint32_t> expected) {
    std::vector<uint32_t> result(actual.begin(), actual.begin() + result_num);
    std::sort(result.begin(), result.end());
    std::sort(expected.begin(), expected.end());
    EXPECT_EQ(result, expected);
}

void run_lvq_round_trip(const std::string& type,
                        const std::filesystem::path& directory,
                        const std::vector<float>& source) {
    std::unique_ptr<Quantizer> quantizer;
    ASSERT_EQ(make_quantizer_dynamic(type, quantizer, TEST_DIM, false), 0);
    ASSERT_NE(quantizer, nullptr);
    ASSERT_EQ(quantizer->train(), 0);

    const std::filesystem::path quantizer_path = directory / (type + ".dat");
    ASSERT_EQ(quantizer->save(quantizer_path.string()), 0);
    std::unique_ptr<Quantizer> loaded;
    ASSERT_EQ(load_quantizer_dynamic(quantizer_path.string(), loaded, false), 0);
    ASSERT_EQ(loaded->dim(), TEST_DIM);

    std::vector<char> encoded(loaded->data_length());
    std::vector<float> decoded(TEST_DIM);
    QuantizerParams params;
    params.code_level = loaded->support_rerank() ? 2 : 0;
    ASSERT_EQ(loaded->encode(encoded.data(), source.data(), params), 0);
    ASSERT_EQ(loaded->decode(decoded.data(), encoded.data(), params), 0);
    double squared_error = 0.0;
    for (size_t index = 0; index < TEST_DIM; ++index) {
        ASSERT_TRUE(std::isfinite(decoded[index]));
        const double difference = static_cast<double>(source[index])
                - static_cast<double>(decoded[index]);
        squared_error += difference * difference;
    }
    EXPECT_LT(squared_error / static_cast<double>(TEST_DIM), 100.0);
}

uint64_t stable_hash(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

size_t region_for_node(size_t node_id) {
    constexpr size_t total_weight = 36;
    const size_t target = stable_hash(node_id) % total_weight;
    size_t cumulative = 0;
    for (size_t rank = 1; rank <= FILTER_REGION_COUNT; ++rank) {
        cumulative += FILTER_REGION_COUNT - rank + 1;
        if (target < cumulative) {
            return rank - 1;
        }
    }
    return FILTER_REGION_COUNT - 1;
}

size_t type_for_node(size_t node_id) {
    constexpr uint64_t type_hash_salt = 0x8f3f73b5cf1c9adeULL;
    return stable_hash(node_id ^ type_hash_salt) % 2;
}

std::vector<std::string> labels_for_region(size_t region) {
    return {"region=" + std::to_string(region), "type=ad"};
}

std::vector<std::string> non_cartesian_labels_for_node(size_t node_id) {
    return {"region=" + std::to_string(region_for_node(node_id)),
            "type=" + std::to_string(type_for_node(node_id))};
}

float squared_l2(const float* lhs, const float* rhs, size_t dim) {
    float distance = 0.0F;
    for (size_t i = 0; i < dim; ++i) {
        const float difference = lhs[i] - rhs[i];
        distance += difference * difference;
    }
    return distance;
}

void write_label_file(const std::filesystem::path& path, bool cartesian) {
    std::ofstream output(path);
    if (!output.is_open()) {
        throw std::runtime_error("failed to create label file: " + path.string());
    }
    for (size_t node_id = 0; node_id < DATASET_BASE_COUNT; ++node_id) {
        const std::vector<std::string> labels = cartesian
                ? labels_for_region(region_for_node(node_id))
                : non_cartesian_labels_for_node(node_id);
        output << labels[0] << '\t' << labels[1] << '\n';
    }
    if (!output.good()) {
        throw std::runtime_error("failed to write label file: " + path.string());
    }
}

void build_brief_index(const std::filesystem::path& directory, bool cartesian) {
    configure_tinker(directory, true, cartesian);
    set_flag("threads_count", "4");
    set_flag("tinker_quantizer_type", "LVQ4x8");
    set_flag("tinker_neighborhood_count", "16");
    set_flag("tinker_ef_construction", "100");
    set_flag("tinker_memory_capacity", std::to_string(DATASET_MEMORY_CAPACITY));
    std::error_code copy_error;
    const bool copied = std::filesystem::copy_file(
            dataset_file("siftsmall_base.fvecs"),
            directory / "features.fvecs",
            std::filesystem::copy_options::none, copy_error);
    ASSERT_TRUE(copied) << copy_error.message();
    write_label_file(directory / "labels.txt", cartesian);
    TinkerIndex builder;
    ASSERT_EQ(builder.train(), 0);
    ASSERT_EQ(builder.build(), 0);
}

double compute_ordinary_recall(TinkerIndex& index, size_t query_count,
                               uint32_t topk) {
    const std::vector<std::vector<float>> queries = read_fvecs(
            dataset_file("siftsmall_query.fvecs"), query_count);
    const std::vector<std::vector<uint32_t>> groundtruth = read_ivecs(
            dataset_file("siftsmall_groundtruth.ivecs"), query_count);
    size_t matches = 0;
    for (size_t query_id = 0; query_id < query_count; ++query_id) {
        std::vector<float> distances(topk);
        std::vector<uint32_t> local_ids(topk);
        Response response;
        response.distance = distances.data();
        response.local_idx = local_ids.data();
        Request request = make_search_request(queries[query_id], topk, nullptr);
        request.ef_search = 1000;
        if (index.search(&request, &response) != 0
                || response.result_num != topk) {
            throw std::runtime_error("ordinary search failed");
        }
        const std::unordered_set<uint32_t> expected(
                groundtruth[query_id].begin(), groundtruth[query_id].end());
        for (uint32_t result_id : local_ids) {
            if (expected.count(result_id) != 0) {
                ++matches;
            }
        }
    }
    return static_cast<double>(matches)
            / static_cast<double>(query_count * topk);
}

double compute_cartesian_filtered_recall(TinkerIndex& index, size_t query_count) {
    const std::vector<std::vector<float>> queries = read_fvecs(
            dataset_file("siftsmall_query.fvecs"), query_count);
    const std::vector<std::vector<float>> db = read_fvecs(
            dataset_file("siftsmall_base.fvecs"), DATASET_BASE_COUNT);
    double total_recall = 0.0;
    for (size_t query_id = 0; query_id < query_count; ++query_id) {
        const size_t region = region_for_node(stable_hash(query_id));
        const std::vector<std::string> labels = labels_for_region(region);
        std::vector<std::pair<float, uint32_t>> exact;
        exact.reserve(DATASET_BASE_COUNT);
        for (size_t node_id = 0; node_id < DATASET_BASE_COUNT; ++node_id) {
            if (region_for_node(node_id) == region) {
                exact.emplace_back(squared_l2(queries[query_id].data(),
                                              db[node_id].data(), TEST_DIM),
                                   node_id);
            }
        }
        std::partial_sort(exact.begin(), exact.begin() + FILTER_TOPK,
                          exact.end());
        std::unordered_set<uint32_t> groundtruth;
        for (uint32_t i = 0; i < FILTER_TOPK; ++i) {
            groundtruth.insert(exact[i].second);
        }

        std::vector<float> distances(FILTER_TOPK);
        std::vector<uint32_t> local_ids(FILTER_TOPK);
        Response response;
        response.distance = distances.data();
        response.local_idx = local_ids.data();
        Request request = make_search_request(queries[query_id], FILTER_TOPK,
                                              &labels);
        request.ef_search = 1000;
        if (index.search(&request, &response) != 0
                || response.result_num != FILTER_TOPK) {
            throw std::runtime_error("cartesian filtered search failed");
        }
        for (uint32_t i = 0; i < response.result_num; ++i) {
            if (region_for_node(local_ids[i]) != region) {
                throw std::runtime_error(
                        "cartesian filtered search returned mismatched label");
            }
            if (groundtruth.count(local_ids[i]) != 0) {
                total_recall += 1.0 / (FILTER_TOPK * query_count);
            }
        }
    }
    return total_recall;
}

void compute_non_cartesian_filtered_recall(TinkerIndex& index,
                                           size_t query_count,
                                           double* partial_recall,
                                           double* full_recall) {
    const std::vector<std::vector<float>> queries = read_fvecs(
            dataset_file("siftsmall_query.fvecs"), query_count);
    const std::vector<std::vector<float>> db = read_fvecs(
            dataset_file("siftsmall_base.fvecs"), DATASET_BASE_COUNT);
    double partial = 0.0;
    double full = 0.0;
    for (size_t query_id = 0; query_id < query_count; ++query_id) {
        const size_t label_seed = stable_hash(query_id);
        const size_t region = region_for_node(label_seed);
        const size_t type = type_for_node(label_seed);
        for (size_t match_mode = 0; match_mode < 2; ++match_mode) {
            std::vector<std::string> labels{
                "region=" + std::to_string(region)};
            if (match_mode != 0) {
                labels.push_back("type=" + std::to_string(type));
            }

            std::vector<std::pair<float, uint32_t>> exact;
            exact.reserve(DATASET_BASE_COUNT);
            for (size_t node_id = 0; node_id < DATASET_BASE_COUNT; ++node_id) {
                if (region_for_node(node_id) != region
                        || (match_mode != 0 && type_for_node(node_id) != type)) {
                    continue;
                }
                exact.emplace_back(squared_l2(queries[query_id].data(),
                                              db[node_id].data(), TEST_DIM),
                                   node_id);
            }
            if (exact.size() < FILTER_TOPK) {
                throw std::runtime_error(
                        "not enough non-cartesian groundtruth");
            }
            std::partial_sort(exact.begin(), exact.begin() + FILTER_TOPK,
                              exact.end());
            std::unordered_set<uint32_t> groundtruth;
            for (uint32_t i = 0; i < FILTER_TOPK; ++i) {
                groundtruth.insert(exact[i].second);
            }

            std::vector<float> distances(FILTER_TOPK);
            std::vector<uint32_t> local_ids(FILTER_TOPK);
            Response response;
            response.distance = distances.data();
            response.local_idx = local_ids.data();
            Request request = make_search_request(queries[query_id], FILTER_TOPK,
                                                  &labels);
            request.ef_search = 1000;
            if (index.search(&request, &response) != 0
                    || response.result_num != FILTER_TOPK) {
                throw std::runtime_error(
                        "non-cartesian filtered search failed");
            }
            for (uint32_t i = 0; i < response.result_num; ++i) {
                const uint32_t node_id = local_ids[i];
                if (region_for_node(node_id) != region
                        || (match_mode != 0 && type_for_node(node_id) != type)) {
                    throw std::runtime_error(
                            "non-cartesian AND filter mismatch");
                }
                const double hit = groundtruth.count(node_id) != 0
                        ? 1.0 / (FILTER_TOPK * query_count) : 0.0;
                if (match_mode == 0) {
                    partial += hit;
                } else {
                    full += hit;
                }
            }
        }
    }
    *partial_recall = partial;
    *full_recall = full;
}

TEST(TinkerQuantizerTest, FlatEncodeDecodeAndPersistence) {
    TemporaryDirectory directory("flat_quantizer");
    const std::vector<float> source = read_fvecs(
            dataset_file("siftsmall_base.fvecs"), 1).front();
    QuantizerFLAT quantizer(TEST_DIM);
    std::vector<char> encoded(quantizer.data_length());
    std::vector<float> decoded(TEST_DIM);
    QuantizerParams params;
    ASSERT_EQ(quantizer.encode(encoded.data(), source.data(), params), 0);
    ASSERT_EQ(quantizer.decode(decoded.data(), encoded.data(), params), 0);
    EXPECT_EQ(decoded, source);

    const std::filesystem::path quantizer_path = directory.path() / "flat.dat";
    ASSERT_EQ(quantizer.save(quantizer_path.string()), 0);
    QuantizerFLAT loaded;
    ASSERT_EQ(loaded.load(quantizer_path.string()), 0);
    ASSERT_EQ(loaded.dim(), TEST_DIM);
    std::fill(decoded.begin(), decoded.end(), 0.0F);
    ASSERT_EQ(loaded.decode(decoded.data(), encoded.data(), params), 0);
    EXPECT_EQ(decoded, source);
}

TEST(TinkerQuantizerTest, LvqFactoryTrainingAndPersistence) {
    FlagGuard flags(tinker_flag_names());
    TemporaryDirectory directory("lvq_quantizer");
    configure_tinker(directory.path(), false, true);
    const std::vector<std::vector<float>> training_vectors = read_fvecs(
            dataset_file("siftsmall_learn.fvecs"), LVQ_TRAIN_VECTOR_COUNT);
    write_fvecs(directory.path() / "features.fvecs", training_vectors);

    run_lvq_round_trip("LVQ8x0", directory.path(), training_vectors.front());
    run_lvq_round_trip("LVQ4x8", directory.path(), training_vectors.front());
    std::unique_ptr<Quantizer> invalid;
    EXPECT_NE(make_quantizer_dynamic("LVQ3x7", invalid, TEST_DIM, false), 0);
    EXPECT_EQ(invalid, nullptr);
}

TEST(TinkerIndexTest, CreateInsertSearchAndReset) {
    FlagGuard flags(tinker_flag_names());
    TemporaryDirectory directory("lifecycle");
    configure_tinker(directory.path(), false, true);
    std::unique_ptr<Index> index = std::make_unique<TinkerIndex>();
    TinkerIndex* tinker = static_cast<TinkerIndex*>(index.get());
    ASSERT_EQ(tinker->create_empty_index(), 0);

    const std::vector<std::vector<float>> features = read_fvecs(
            dataset_file("siftsmall_base.fvecs"), 2);
    const std::vector<float>& first = features[0];
    const std::vector<float>& second = features[1];
    InsertRequest first_insert = make_insert_request(first, 101, nullptr);
    InsertRequest second_insert = make_insert_request(second, 202, nullptr);
    ASSERT_EQ(index->insert(&first_insert), 0);
    ASSERT_EQ(index->insert(&second_insert), 0);
    ASSERT_EQ(tinker->size(), 2U);

    std::vector<float> distances(2);
    std::vector<uint32_t> local_ids(2);
    Response response;
    response.distance = distances.data();
    response.local_idx = local_ids.data();
    Request request = make_search_request(second, 2, nullptr);
    ASSERT_EQ(index->search(&request, &response), 0);
    ASSERT_EQ(response.result_num, 2U);
    EXPECT_EQ(response.local_idx[0], 202U);

    EXPECT_EQ(tinker->reset(), 0);
    EXPECT_EQ(tinker->size(), 0U);
}

TEST(TinkerIndexTest, OptionalEfSearchDefaultsToTopkAndPromotesPositiveWidths) {
    FlagGuard flags(tinker_flag_names());
    TemporaryDirectory directory("optional_ef_search");
    const std::vector<std::vector<float>> features = read_fvecs(
            dataset_file("siftsmall_base.fvecs"), 4);
    const std::vector<std::string> labels = {"shared"};
    const std::vector<std::string> unknown_labels = {"unknown"};
    std::vector<float> distances(features.size());
    std::vector<uint32_t> local_ids(features.size());
    Response response;
    response.distance = distances.data();
    response.local_idx = local_ids.data();
    for (bool brief : {false, true}) {
        for (bool cartesian : {false, true}) {
            configure_tinker(directory.path(), brief, cartesian);
            TinkerIndex index;
            ASSERT_EQ(index.create_empty_index(), 0);
            Request request;
            request.feature = features[0].data();
            request.topk = static_cast<uint32_t>(features.size());
            request.labels = brief ? &labels : nullptr;
            // Default width is valid even before any vectors have been inserted.
            ASSERT_EQ(index.search(&request, &response), 0);
            EXPECT_EQ(response.result_num, 0U);
            for (size_t id = 0; id < features.size(); ++id) {
                InsertRequest insert = make_insert_request(
                        features[id], id, brief ? &labels : nullptr);
                ASSERT_EQ(index.insert(&insert), 0);
            }
            ASSERT_EQ(index.search(&request, &response), 0);
            ASSERT_EQ(response.result_num, features.size());
            expect_ids(local_ids, response.result_num, {0, 1, 2, 3});
            const std::vector<uint32_t> default_ids = local_ids;
            const std::vector<float> default_distances = distances;
            const std::vector<int64_t> invalid_widths = {
                    -1, -(int64_t{1} << 32), std::numeric_limits<int64_t>::min()};
            for (int64_t width : invalid_widths) {
                request.ef_search = width;
                EXPECT_NE(index.search(&request, &response), 0);
                if (brief) {
                    request.labels = &unknown_labels;
                    EXPECT_NE(index.search(&request, &response), 0);
                    request.labels = &labels;
                }
            }
            for (int64_t width : {int64_t{0}, int64_t{1},
                                 static_cast<int64_t>(request.topk),
                                 static_cast<int64_t>(request.topk) + 1}) {
                request.ef_search = width;
                ASSERT_EQ(index.search(&request, &response), 0);
                ASSERT_EQ(response.result_num, features.size());
                EXPECT_EQ(local_ids, default_ids);
                EXPECT_EQ(distances, default_distances);
                if (brief) {
                    request.labels = &unknown_labels;
                    ASSERT_EQ(index.search(&request, &response), 0);
                    EXPECT_EQ(response.result_num, 0U);
                    request.labels = &labels;
                }
            }
        }
    }
}

TEST(TinkerIndexTest, NativeWrapperValidatesEfSearchAfterLoad) {
    FlagGuard flags(tinker_flag_names());
    TemporaryDirectory directory("native_ef_search");
    configure_tinker(directory.path(), false, true);
    TinkerIndex index;
    ASSERT_EQ(index.create_empty_index(), 0);
    const std::vector<std::vector<float>> features = read_fvecs(
            dataset_file("siftsmall_base.fvecs"), 4);
    for (size_t id = 0; id < features.size(); ++id) {
        InsertRequest insert = make_insert_request(features[id], id, nullptr);
        ASSERT_EQ(index.insert(&insert), 0);
    }
    ASSERT_EQ(index.save_quantizer(), 0);
    ASSERT_EQ(index.save_index(), 0);
    // load() chooses the on-disk engine, not necessarily the constructor engine.
    py_puck_api::PySearcher searcher(static_cast<int>(IndexType::PUCK));
    ASSERT_EQ(searcher.load(), 0);
    const uint32_t topk = static_cast<uint32_t>(features.size());
    std::vector<float> distances(topk);
    std::vector<uint32_t> local_ids(topk);
    for (int64_t width : {int64_t{-1}, -(int64_t{1} << 32),
                         std::numeric_limits<int64_t>::min()}) {
        EXPECT_NE(searcher.search(1, features[0].data(), topk, width,
                                 distances.data(), local_ids.data()), 0);
    }
    for (int64_t width : {int64_t{0}, int64_t{1},
                         static_cast<int64_t>(topk), static_cast<int64_t>(topk) + 1}) {
        ASSERT_EQ(searcher.search(1, features[0].data(), topk, width,
                                 distances.data(), local_ids.data()), 0);
        expect_ids(local_ids, topk, {0, 1, 2, 3});
    }
}

TEST(TinkerIndexTest, NonCartesianFilteringPersistsAndUsesAndSemantics) {
    FlagGuard flags(tinker_flag_names());
    TemporaryDirectory directory("non_cartesian");
    configure_tinker(directory.path(), true, false);
    const std::vector<std::vector<std::string>> labels = {
            {"A", "B"}, {"A", "C"}, {"B", "C"}, {"A", "B", "C"}};
    const std::vector<std::vector<float>> features = read_fvecs(
            dataset_file("siftsmall_base.fvecs"), labels.size());
    TinkerIndex index;
    ASSERT_EQ(index.create_empty_index(), 0);
    for (size_t id = 0; id < labels.size(); ++id) {
        InsertRequest insert = make_insert_request(features[id], id, &labels[id]);
        ASSERT_EQ(index.insert(&insert), 0);
    }

    std::vector<float> distances(labels.size());
    std::vector<uint32_t> local_ids(labels.size());
    Response response;
    response.distance = distances.data();
    response.local_idx = local_ids.data();
    const std::vector<std::string> ab_labels = {"A", "B"};
    Request request = make_search_request(features[0], labels.size(), &ab_labels);
    ASSERT_EQ(index.search(&request, &response), 0);
    ASSERT_EQ(response.result_num, 2U);
    expect_ids(local_ids, response.result_num, {0, 3});

    const std::vector<std::string> a_labels = {"A"};
    request = make_search_request(features[0], labels.size(), &a_labels);
    ASSERT_EQ(index.search(&request, &response), 0);
    ASSERT_EQ(response.result_num, 3U);
    expect_ids(local_ids, response.result_num, {0, 1, 3});

    ASSERT_EQ(index.save_quantizer(), 0);
    ASSERT_EQ(index.save_index(), 0);
    TinkerIndex loaded;
    ASSERT_EQ(loaded.init(), 0);
    request = make_search_request(features[0], labels.size(), &ab_labels);
    ASSERT_EQ(loaded.search(&request, &response), 0);
    ASSERT_EQ(response.result_num, 2U);
    expect_ids(local_ids, response.result_num, {0, 3});
}

TEST(TinkerIndexTest, CartesianLabelsAndUnifiedApiValidation) {
    FlagGuard flags(tinker_flag_names());
    TemporaryDirectory directory("cartesian_validation");
    configure_tinker(directory.path(), true, true);
    TinkerIndex index;
    ASSERT_EQ(index.create_empty_index(), 0);

    const std::vector<std::string> cn_ad = {"region=cn", "type=ad"};
    const std::vector<std::string> ad_cn = {"type=ad", "region=cn"};
    const std::vector<std::string> us_ad = {"region=us", "type=ad"};
    const std::vector<std::vector<float>> features = read_fvecs(
            dataset_file("siftsmall_base.fvecs"), 2);
    const std::vector<float>& cn_feature = features[0];
    const std::vector<float>& us_feature = features[1];
    InsertRequest cn_insert = make_insert_request(cn_feature, 7, &cn_ad);
    InsertRequest us_insert = make_insert_request(us_feature, 9, &us_ad);
    ASSERT_EQ(index.insert(&cn_insert), 0);
    ASSERT_EQ(index.insert(&us_insert), 0);

    std::vector<float> distances(2);
    std::vector<uint32_t> local_ids(2);
    Response response;
    response.distance = distances.data();
    response.local_idx = local_ids.data();
    Request request = make_search_request(cn_feature, 2, &ad_cn);
    ASSERT_EQ(index.search(&request, &response), 0);
    ASSERT_EQ(response.result_num, 1U);
    EXPECT_EQ(response.local_idx[0], 7U);

    const std::vector<std::string> unknown = {"region=unknown", "type=ad"};
    request = make_search_request(cn_feature, 2, &unknown);
    ASSERT_EQ(index.search(&request, &response), 0);
    EXPECT_EQ(response.result_num, 0U);

    EXPECT_NE(index.insert(nullptr), 0);
    EXPECT_NE(index.search(nullptr, &response), 0);
    request = make_search_request(cn_feature, 2, &cn_ad);
    EXPECT_NE(index.search(&request, nullptr), 0);
    InsertRequest wrong_dim = make_insert_request(cn_feature, 11, &cn_ad);
    wrong_dim.dim = static_cast<uint32_t>(TEST_DIM - 1);
    EXPECT_NE(index.insert(&wrong_dim), 0);
    InsertRequest null_feature = make_insert_request(cn_feature, 11, &cn_ad);
    null_feature.feature = nullptr;
    EXPECT_NE(index.insert(&null_feature), 0);
    std::vector<std::string> empty_labels;
    InsertRequest empty_insert = make_insert_request(cn_feature, 11, &empty_labels);
    EXPECT_NE(index.insert(&empty_insert), 0);
    request = make_search_request(cn_feature, 2, nullptr);
    EXPECT_EQ(index.search(&request, &response), 0);
    request = make_search_request(cn_feature, 2, &empty_labels);
    EXPECT_EQ(index.search(&request, &response), 0);
    request = make_search_request(cn_feature, 2, &cn_ad);
    request.feature = nullptr;
    EXPECT_NE(index.search(&request, &response), 0);
}

TEST(TinkerDatasetTest, TrainBuildInitSearchRecall) {
    FlagGuard flags(tinker_flag_names());
    TemporaryDirectory directory("dataset_recall");
    configure_tinker(directory.path(), false, true);
    set_flag("threads_count", "4");
    set_flag("tinker_quantizer_type", "LVQ4x8");
    set_flag("tinker_neighborhood_count", "16");
    set_flag("tinker_ef_construction", "100");
    set_flag("tinker_memory_capacity", std::to_string(DATASET_MEMORY_CAPACITY));
    std::error_code copy_error;
    const bool copied = std::filesystem::copy_file(
            dataset_file("siftsmall_base.fvecs"),
            directory.path() / "features.fvecs",
            std::filesystem::copy_options::none, copy_error);
    ASSERT_TRUE(copied) << copy_error.message();

    TinkerIndex builder;
    ASSERT_EQ(builder.train(), 0);
    ASSERT_EQ(builder.build(), 0);

    TinkerIndex index;
    ASSERT_EQ(index.init(), 0);
    ASSERT_EQ(index.size(), DATASET_BASE_COUNT);
    const std::vector<std::vector<float>> queries = read_fvecs(
            dataset_file("siftsmall_query.fvecs"), RECALL_QUERY_COUNT);
    const std::vector<std::vector<uint32_t>> groundtruth = read_ivecs(
            dataset_file("siftsmall_groundtruth.ivecs"), RECALL_QUERY_COUNT);
    size_t matches = 0;
    for (size_t query_id = 0; query_id < RECALL_QUERY_COUNT; ++query_id) {
        std::vector<float> distances(RECALL_TOPK);
        std::vector<uint32_t> local_ids(RECALL_TOPK);
        Response response;
        response.distance = distances.data();
        response.local_idx = local_ids.data();
        Request request = make_search_request(queries[query_id], RECALL_TOPK, nullptr);
        request.ef_search = 1000;
        ASSERT_EQ(index.search(&request, &response), 0);
        ASSERT_EQ(response.result_num, RECALL_TOPK);
        const std::unordered_set<uint32_t> expected(
                groundtruth[query_id].begin(), groundtruth[query_id].end());
        for (uint32_t result_id : local_ids) {
            EXPECT_LT(result_id, DATASET_BASE_COUNT);
            if (expected.count(result_id) != 0) {
                ++matches;
            }
        }
    }
    const double recall = static_cast<double>(matches)
            / static_cast<double>(RECALL_QUERY_COUNT * RECALL_TOPK);
    LOG(INFO) << "ordinary recall: " << recall;
    EXPECT_GT(recall, 0.9);
}

TEST(TinkerDatasetTest, BriefRecallRegression) {
    FlagGuard flags(tinker_flag_names());
    TemporaryDirectory directory("brief_recall");
    build_brief_index(directory.path(), /*cartesian=*/true);

    TinkerIndex index;
    ASSERT_EQ(index.init(), 0);
    const double ordinary_recall =
            compute_ordinary_recall(index, RECALL_QUERY_COUNT, RECALL_TOPK);
    LOG(INFO) << "brief ordinary recall: " << ordinary_recall;
    EXPECT_GT(ordinary_recall, 0.65);
    const double filtered_recall =
            compute_cartesian_filtered_recall(index, RECALL_QUERY_COUNT);
    LOG(INFO) << "brief filtered recall: " << filtered_recall;
    EXPECT_GT(filtered_recall, 0.75);
}

TEST(TinkerDatasetTest, NonCartesianBriefRecall) {
    FlagGuard flags(tinker_flag_names());
    TemporaryDirectory directory("non_cartesian_recall");
    build_brief_index(directory.path(), /*cartesian=*/false);

    TinkerIndex index;
    ASSERT_EQ(index.init(), 0);
    const double ordinary_recall =
            compute_ordinary_recall(index, RECALL_QUERY_COUNT, RECALL_TOPK);
    LOG(INFO) << "non-cartesian ordinary recall: " << ordinary_recall;
    EXPECT_GT(ordinary_recall, 0.9);
    double partial_recall = 0.0;
    double full_recall = 0.0;
    compute_non_cartesian_filtered_recall(index, RECALL_QUERY_COUNT,
                                          &partial_recall, &full_recall);
    LOG(INFO) << "non-cartesian partial-label recall: " << partial_recall;
    LOG(INFO) << "non-cartesian full-label recall: " << full_recall;
    EXPECT_GT(partial_recall, 0.75);
    EXPECT_GT(full_recall, 0.75);
}

TEST(TinkerDatasetTest, FilterMetadataLoadFailure) {
    FlagGuard flags(tinker_flag_names());
    TemporaryDirectory directory("filter_metadata_failure");
    build_brief_index(directory.path(), /*cartesian=*/true);

    const std::filesystem::path metadata = directory.path() / "tinker.filter_meta";
    const std::filesystem::path backup = metadata.string() + ".test_backup";
    std::error_code error;

    std::filesystem::rename(metadata, backup, error);
    ASSERT_FALSE(error) << error.message();
    TinkerIndex missing_index;
    EXPECT_NE(missing_index.init(), 0);
    std::filesystem::rename(backup, metadata, error);
    ASSERT_FALSE(error) << error.message();

    std::filesystem::copy_file(
            metadata, backup, std::filesystem::copy_options::overwrite_existing,
            error);
    ASSERT_FALSE(error) << error.message();
    {
        std::ofstream corrupt(metadata, std::ios::binary | std::ios::trunc);
        const uint8_t truncated = 0;
        corrupt.write(reinterpret_cast<const char*>(&truncated),
                      sizeof(truncated));
    }
    TinkerIndex corrupted_index;
    EXPECT_NE(corrupted_index.init(), 0);
    std::filesystem::remove(metadata, error);
    ASSERT_FALSE(error) << error.message();
    std::filesystem::rename(backup, metadata, error);
    ASSERT_FALSE(error) << error.message();
}

}  // namespace
}  // namespace puck
