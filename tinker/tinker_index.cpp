//Copyright (c) 2024 Baidu, Inc.  All Rights Reserved.
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.
/**
 * @file tinker_index.cpp
 * @author hananoyuuki@qq.com
 * @date 2024/9/23 20:05
 * @brief tinker_index.cpp version: 2.0 after refactoring
 *
 **/
#include <queue>
#include <memory>
#include <omp.h>
#include <future>
#include <queue>
#include <unordered_set>
#include <filesystem>
#include <algorithm>
#include <cmath>
#include <cstring>
#include "gflags/puck_gflags.h"
#include "tinker/internal/utils.h"
#include "common/normalization.h"
#include "tinker/tinker_index.h"
#include "quantizer/factory.h"

namespace puck {

TinkerIndex::TinkerIndex() {
    _index_type = puck::IndexType::TINKER;  // hard code
    _dim = puck::FLAGS_feature_dim;
    _thread_num = puck::FLAGS_threads_count;
    _ram_capacity = puck::FLAGS_tinker_memory_capacity;
    _M = puck::FLAGS_tinker_neighborhood_count;
    _ef_construction = puck::FLAGS_tinker_ef_construction;

    // 裁边松弛系数暂时写死为hnsw的1.0
    // 后续优化->可配置，作为index.dat描述的参数之一
    // 因为目前并没有研究说明其对索引质量产生影响的根本原因
    _alpha = 1.0; // hnsw rng param
    _brief = FLAGS_tinker_brief ? 1 : 0;
    _cartesian = FLAGS_tinker_cartesian ? 1 : 0;
    _whether_norm = puck::FLAGS_whether_norm;
}

bool TinkerIndex::is_brief_enabled() const {
    return _brief != 0;
}

bool TinkerIndex::is_cartesian_enabled() const {
    return _cartesian != 0;
}

std::mutex& TinkerIndex::get_filter_label_mutex(LabelId label_id) {
    const unsigned int offset_filter_label_mutex =
        label_id & FILTER_LABEL_HASH_LOWBIT;
    return _filter_label_mutex[offset_filter_label_mutex];
}

std::string TinkerIndex::canonicalize_labels(
        const std::vector<std::string>& labels) const {
    std::vector<std::string> sorted_labels = labels;
    // 先去重
    std::sort(sorted_labels.begin(), sorted_labels.end());
    sorted_labels.erase(std::unique(sorted_labels.begin(), sorted_labels.end()),
                        sorted_labels.end());

    std::string canonical;
    // 格式为 label_size:label|label_size:label|label_size:label|
    for (const std::string& label : sorted_labels) {
        canonical.append(std::to_string(label.size()));
        canonical.push_back(':');
        canonical.append(label);
        canonical.push_back('|');
    }
    return canonical;
}

int TinkerIndex::get_or_create_label_id(
    const std::vector<std::string>& labels,
    std::vector<LabelId>* label_ids) {
    if (label_ids == nullptr || labels.empty()) {
        return -1;
    }
    std::vector<std::string> normalized = labels;
    std::sort(normalized.begin(), normalized.end());
    normalized.erase(std::unique(normalized.begin(), normalized.end()),
                     normalized.end());
    if (_cartesian != 0) {
        // 笛卡尔索引的多标签会被视为一个bind标签
        normalized = {canonicalize_labels(normalized)};
    }

    std::unique_lock<std::shared_mutex> lock(_label_dictionary_mutex);
    label_ids->clear();
    for (const std::string& label : normalized) {
        auto iter = _label_to_id.find(label);
        if (iter == _label_to_id.end()) {
            const LabelId new_id = static_cast<LabelId>(_id_to_label.size());
            _label_to_id.emplace(label, new_id);
            _id_to_label.push_back(label);
            label_ids->push_back(new_id);
        } else {
            label_ids->push_back(iter->second);
        }
    }
    std::sort(label_ids->begin(), label_ids->end());
    return 0;
}

int TinkerIndex::get_label_ids(
    const std::vector<std::string>& labels,
    std::vector<LabelId>* label_ids) const {
    if (label_ids == nullptr || labels.empty()) {
        return -1;
    }
    std::vector<std::string> normalized = labels;
    std::sort(normalized.begin(), normalized.end());
    normalized.erase(std::unique(normalized.begin(), normalized.end()),
                     normalized.end());
    if (_cartesian != 0) {
        normalized = {canonicalize_labels(normalized)};
    }
    label_ids->clear();
    std::shared_lock<std::shared_mutex> lock(_label_dictionary_mutex);
    for (const std::string& label : normalized) {
        auto iter = _label_to_id.find(label);
        if (iter == _label_to_id.end()) {
            label_ids->clear();
            return -1;
        }
        label_ids->push_back(iter->second);
    }
    std::sort(label_ids->begin(), label_ids->end());
    return 0;
}

bool TinkerIndex::node_matches_any_label(
    unsigned int node_id,
    const std::vector<LabelId>& label_ids) const {
    if (node_id >= _elements_capacity || _node_label_id_vectors == nullptr) {
        return false;
    }
    const std::shared_ptr<const std::vector<LabelId>> node_labels =
        std::atomic_load_explicit(&_node_label_id_vectors[node_id],
                                  std::memory_order_acquire);
    if (node_labels == nullptr) {
        return false;
    }
    return std::any_of(label_ids.begin(), label_ids.end(),
                       [&node_labels](LabelId label_id) {
                           return std::find(node_labels->begin(),
                                            node_labels->end(),
                                            label_id) != node_labels->end();
                       });
}

bool TinkerIndex::node_matches_all_labels(
    unsigned int node_id,
    const std::vector<LabelId>& label_ids) const {
    if (node_id >= _elements_capacity || _node_label_id_vectors == nullptr) {
        return false;
    }
    const std::shared_ptr<const std::vector<LabelId>> node_labels =
        std::atomic_load_explicit(&_node_label_id_vectors[node_id],
                                  std::memory_order_acquire);
    if (node_labels == nullptr) {
        return false;
    }
    return std::all_of(label_ids.begin(), label_ids.end(),
                       [&node_labels](LabelId label_id) {
                           return std::find(node_labels->begin(),
                                            node_labels->end(),
                                            label_id) != node_labels->end();
                       });
}

std::vector<unsigned int> TinkerIndex::get_filter_entrypoints(
    const std::vector<LabelId>& label_ids) const {
    std::vector<unsigned int> result;
    std::shared_lock<std::shared_mutex> lock(_filter_entrypoint_mutex);
    for (LabelId label_id : label_ids) {
        auto iter = _filter_entrypoints.find(label_id);
        if (iter != _filter_entrypoints.end()) {
            result.insert(result.end(), iter->second.begin(), iter->second.end());
        }
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

int TinkerIndex::update_filter_entrypoints(
    const std::vector<LabelId>& label_ids,
    unsigned int node_id) {
    std::unique_lock<std::shared_mutex> lock(_filter_entrypoint_mutex);
    for (LabelId label_id : label_ids) {
        std::vector<unsigned int>& nodes = _label_to_nodes[label_id];
        nodes.push_back(node_id);
        std::vector<unsigned int>& entrypoints = _filter_entrypoints[label_id];
        const size_t desired = std::max<size_t>(1, static_cast<size_t>(
            std::ceil(std::log2(static_cast<double>(nodes.size())))));
        if (entrypoints.size() < desired) {
            entrypoints.push_back(node_id);
        }
    }
    return 0;
}

int TinkerIndex::check_index_type() {
    if (_index_type != IndexType::TINKER) {
        LOG(ERROR) << "index_type is not TINKER";
        return -1;
    }
    return 0;
}

int TinkerIndex::init() {
    LOG(INFO) << "init TinkerIndex ...";

    if(read_model_file() != 0){
        LOG(ERROR) << "read model file failed";
        return -1;
    }

    if(read_index() != 0) {
        LOG(ERROR) << "read_index has error.";
        return -1;
    }

    if (_brief != 0 && load_filter_metadata() != 0) {
        LOG(ERROR) << "load filter metadata failed for brief index";
        return -1;
    }

    return 0;
}

int TinkerIndex::read_model_file() {
    // 1. 读取index.dat索引描述文件
    std::string index_file_path = puck::FLAGS_index_path + "/" + puck::FLAGS_index_file_name;
    std::ifstream model_file(index_file_path, std::ios::binary);
    if (!model_file.is_open()) {
        LOG(ERROR) << "open model file error";
        return -1;
    }
    // 1.* 检查index.dat文件实际大小 - 防止注入攻击
    model_file.seekg(0, std::ios::end);
    ssize_t real_file_size = model_file.tellg();
    model_file.seekg(0, std::ios::beg);

    ssize_t file_size = 0;
    model_file.read(reinterpret_cast<char*>(&file_size), sizeof(ssize_t));

    if(file_size != real_file_size){
        LOG(ERROR) << "index.dat file size error";
        return -1;
    }

    // 2. 反序列化相关参数
    model_file.read(reinterpret_cast<char*>(&_index_type), sizeof(_index_type));
    if(check_index_type() != 0) {
        LOG(ERROR) << "check_index_type has error.";
        return -1;
    }
    model_file.read(reinterpret_cast<char*>(&_dim), sizeof(_dim));
    model_file.read(reinterpret_cast<char*>(&_M), sizeof(_M));
    model_file.read(reinterpret_cast<char*>(&_ef_construction), sizeof(_ef_construction));
    // model_file.read(reinterpret_cast<char*>(&_thread_num), sizeof(_thread_num));
    model_file.read(reinterpret_cast<char*>(&_enterpoint_id), sizeof(_enterpoint_id));
    unsigned int stored_temp = 0;
    model_file.read(reinterpret_cast<char*>(&stored_temp), sizeof(stored_temp));
    _total_elements_stored.store(stored_temp, std::memory_order_release);

    _brief = 0;
    _cartesian = 1;
    uint8_t brief_value = 0;
    uint8_t cartesian_value = 1;
    uint8_t whether_norm_value = _whether_norm ? 1 : 0;
    const std::streampos legacy_end = model_file.tellg();
    model_file.read(reinterpret_cast<char*>(&brief_value), sizeof(brief_value));
    if (model_file.gcount() == sizeof(brief_value)) {
        if (brief_value > 1) {
            LOG(ERROR) << "invalid brief value: " << brief_value;
            return -1;
        }
        _brief = brief_value;
        const std::streampos cartesian_pos = model_file.tellg();
        model_file.read(reinterpret_cast<char*>(&cartesian_value),
                        sizeof(cartesian_value));
        if (model_file.gcount() == sizeof(cartesian_value)) {
            if (cartesian_value > 1) {
                LOG(ERROR) << "invalid cartesian value: "
                           << cartesian_value;
                return -1;
            }
            _cartesian = cartesian_value;
            const std::streampos whether_norm_pos = model_file.tellg();
            model_file.read(reinterpret_cast<char*>(&whether_norm_value),
                            sizeof(whether_norm_value));
            if (model_file.gcount() == sizeof(whether_norm_value)) {
                if (whether_norm_value > 1) {
                    LOG(ERROR) << "invalid whether_norm value: "
                               << static_cast<int>(whether_norm_value);
                    return -1;
                }
                const bool persisted_whether_norm = whether_norm_value != 0;
                if (persisted_whether_norm != puck::FLAGS_whether_norm) {
                    LOG(WARNING) << "persisted whether_norm="
                                 << persisted_whether_norm
                                 << " overrides runtime flag="
                                 << puck::FLAGS_whether_norm;
                }
                _whether_norm = persisted_whether_norm;
            } else if (model_file.eof()) {
                model_file.clear();
                model_file.seekg(whether_norm_pos);
            } else {
                LOG(ERROR) << "read whether_norm field failed";
                return -1;
            }
        } else if (model_file.eof()) {
            model_file.clear();
            model_file.seekg(cartesian_pos);
        } else {
            LOG(ERROR) << "read cartesian field failed";
            return -1;
        }
    } else if (model_file.eof()) {
        model_file.clear();
        model_file.seekg(legacy_end);
    } else {
        LOG(ERROR) << "read brief field failed";
        return -1;
    }

    if(!model_file.good() || model_file.tellg() != file_size) {
        LOG(ERROR) << "read model file error";
        return -1;
    }
    model_file.close();
    LOG(INFO) << "read model file succ.";
    return 0;
}

int TinkerIndex::save_model_file() {
    std::string index_file_path = puck::FLAGS_index_path + "/" + puck::FLAGS_index_file_name;
    std::ofstream model_file(index_file_path, std::ios::binary);
    if (!model_file.is_open()) {
        LOG(ERROR) << "open model file error";
        return -1;
    }
    // 1. 序列化相关参数
    unsigned int stored_temp = _total_elements_stored.load(std::memory_order_acquire);
    ssize_t conf_size = sizeof(ssize_t);
    conf_size += sizeof(_index_type)+
            sizeof(_dim)+
            sizeof(_M)+
            sizeof(_ef_construction)+
            // sizeof(_thread_num)+
            sizeof(_enterpoint_id)+
            sizeof(stored_temp)+
            sizeof(_brief)+
            sizeof(_cartesian)+
            sizeof(uint8_t);
    model_file.write(reinterpret_cast<char*>(&conf_size), sizeof(ssize_t));
    model_file.write(reinterpret_cast<char*>(&_index_type), sizeof(_index_type));
    model_file.write(reinterpret_cast<char*>(&_dim), sizeof(_dim));
    model_file.write(reinterpret_cast<char*>(&_M), sizeof(_M));
    model_file.write(reinterpret_cast<char*>(&_ef_construction), sizeof(_ef_construction));
    // model_file.write(reinterpret_cast<char*>(&_thread_num), sizeof(_thread_num));
    model_file.write(reinterpret_cast<char*>(&_enterpoint_id), sizeof(_enterpoint_id));
    model_file.write(reinterpret_cast<char*>(&stored_temp), sizeof(stored_temp));
    uint8_t brief_value = _brief;
    model_file.write(reinterpret_cast<char*>(&brief_value), sizeof(brief_value));
    uint8_t cartesian_value = _cartesian;
    model_file.write(reinterpret_cast<char*>(&cartesian_value),
                     sizeof(cartesian_value));
    uint8_t whether_norm_value = _whether_norm ? 1 : 0;
    model_file.write(reinterpret_cast<char*>(&whether_norm_value),
                     sizeof(whether_norm_value));
    model_file.flush();
    if (!model_file) {
        LOG(ERROR) << "write model file failed";
        model_file.close();
        return -1;
    }
    model_file.close();
    if (model_file.fail()) {
        LOG(ERROR) << "close model file failed";
        return -1;
    }
    LOG(INFO) << "save model file succ.";
    return 0;
}

int TinkerIndex::check_feature_data() {
    std::string feature_file_path =
        puck::FLAGS_index_path + "/" + puck::FLAGS_feature_file_name;
    std::ifstream feature_data_stream(feature_file_path, std::ios::binary | std::ios::in);
    LOG(INFO) << "opened feature file: " << feature_file_path << "\n";
    if (!feature_data_stream.is_open()) {
        LOG(ERROR) << "open feature file error";
        return -1;
    }

    feature_data_stream.seekg(0, std::ios::end);
    const std::streamoff stream_size = feature_data_stream.tellg();
    if (stream_size < 0) {
        LOG(ERROR) << "get feature file size failed";
        return -1;
    }
    const size_t feature_data_stream_size = static_cast<size_t>(stream_size);
    if(feature_data_stream_size == 0){
        LOG(INFO) << "empty feature file, will init a empty index.";
        return 0;
    }

    int dim_read = 0;
    feature_data_stream.seekg(0, std::ios::beg);
    feature_data_stream.read(reinterpret_cast<char*>(&dim_read), sizeof(int));
    if (!feature_data_stream || dim_read != puck::FLAGS_feature_dim) {
        LOG(ERROR) << "feature dim not match, file dim: " << dim_read 
                   << ", conf dim: " << puck::FLAGS_feature_dim;
        return -1;
    }

    // 计算feature条数
    size_t ntotal = feature_data_stream_size / (sizeof(int) + dim_read * sizeof(float));
    if (ntotal * (sizeof(int) + static_cast<size_t>(dim_read) * sizeof(float)) !=
        feature_data_stream_size) {
        LOG(ERROR) << "feature file has a truncated record";
        return -1;
    }
    if (ntotal > static_cast<size_t>(std::numeric_limits<int>::max())) {
        LOG(ERROR) << "feature count exceeds supported range: " << ntotal;
        return -1;
    }

    // 分块校验每条记录的维度，仅确认文件格式合法
    feature_data_stream.seekg(0, std::ios::beg);
    size_t cur = 0;
    size_t block_num = 1024 * 1024;
    std::unique_ptr<char[]> buffer(new char[block_num * (sizeof(int) + dim_read * sizeof(float))]);
    while(cur < ntotal) {
        size_t res = ntotal - cur;
        size_t true_read_num = res > block_num ? block_num : res;
        feature_data_stream.read(buffer.get(), true_read_num * (sizeof(int) + dim_read * sizeof(float)));
        if(!feature_data_stream.good()) {
            LOG(ERROR) << "read feature data error";
            return -1;
        }
        for(size_t i = 0; i < true_read_num; ++i) {
            const char* cur_feature_p =
                    buffer.get() + i * (sizeof(int) + dim_read * sizeof(float));
            int cur_dim = 0;
            std::memcpy(&cur_dim, cur_feature_p, sizeof(cur_dim));
            if(cur_dim != dim_read){
                LOG(ERROR) << "feature dim not match";
                return -1;
            }
        }
        cur += true_read_num;
    }

    return int(ntotal);
}

int TinkerIndex::train(){
    int fea_num = check_feature_data();
    if(fea_num < 0) {
        LOG(ERROR) << "check_feature_data has error.";
        return -1;
    }
    else if(fea_num == 0 && FLAGS_tinker_quantizer_type != "FLAT"){
        LOG(ERROR) << "quantizer " << FLAGS_tinker_quantizer_type 
                    << " not support init an empty index";
        return -1;
    }

    if (make_quantizer_dynamic(FLAGS_tinker_quantizer_type, _quantizer,
                               _dim, _whether_norm) != 0 ||
        _quantizer->train() != 0) {
        LOG(ERROR) << "train quantizer failed";
        return -1;
    }

    if (save_quantizer() != 0) {
        return -1;
    }

    return 0;
}

int TinkerIndex::adapt_data_width(){
    if(!_quantizer){
        LOG(ERROR) << "cannot calculate data width: no quantizer specified";
        return -1;
    }
    // linknum + linklist
    _linklist_length = (_M + 1) * sizeof(uint32_t);
    _fea_length = _quantizer->data_length();
    _data_length = _linklist_length + _fea_length;
    _elements_capacity = _ram_capacity / _data_length;
    return 0;
}

int TinkerIndex::build() {
    const int checked_feature_count = check_feature_data();
    if (checked_feature_count < 0) {
        LOG(ERROR) << "check_feature_data has error before build";
        return -1;
    }
    // build阶段采用FLAT量化器，保持精确距离比较
    if (make_quantizer_dynamic("FLAT", _quantizer, _dim, _whether_norm) != 0) {
        return -1;
    }
    _quant_sdc_distance_func = get_quant_sdc_distance_func(_quantizer);
    _quant_adc_distance_func = get_quant_adc_distance_func(_quantizer);

    // 计算FLAT量化器意义下的数据宽度和pad
    if(adapt_data_width() != 0){
        LOG(ERROR) << "adapt data width error";
        return -1;
    }

    // 先释放旧内存，避免新旧共存导致内存峰值
    _fea_ptr.reset();
    _linklist_ptr.reset();
    _memory_to_local_idx.clear();
    _memory_to_local_idx.shrink_to_fit();

    // 分配内存
    _fea_ptr.reset(new char[_fea_length * _elements_capacity]());
    _linklist_ptr.reset(new char[_linklist_length * _elements_capacity]());
    _visited_pool.init(_elements_capacity);
    _node_mutex.reset(new std::shared_mutex[NODE_MUTEX_COUNT]);
    _filter_label_mutex.reset(new std::mutex[FILTER_LABEL_MUTEX_COUNT]);
    _node_label_ids.reset(new std::atomic<LabelId>[_elements_capacity]);
    _node_label_id_vectors.reset(
        new std::shared_ptr<const std::vector<LabelId>>[
            _elements_capacity]);
    for (unsigned int i = 0; i < _elements_capacity; ++i) {
        _node_label_ids[i].store(INVALID_LABEL_ID, std::memory_order_relaxed);
    }
    _memory_to_local_idx.resize(_elements_capacity, -1);

    std::vector<std::vector<std::string>> build_labels;
    if (_brief != 0) {
        const std::string label_file_path = puck::FLAGS_index_path + "/" +
                                            puck::FLAGS_brief_file_name;
        std::ifstream label_stream(label_file_path);
        if (!label_stream.is_open()) {
            LOG(ERROR) << "open brief label file failed: " << label_file_path;
            return -1;
        }
        std::string line;
        while (std::getline(label_stream, line)) {
            std::vector<std::string> labels = utils::ssplit(line, '\t');
            labels.erase(std::remove_if(labels.begin(), labels.end(),
                [](const std::string& label) { return label.empty(); }), labels.end());
            if (labels.empty()) {
                LOG(ERROR) << "brief label line is empty";
                return -1;
            }
            build_labels.push_back(std::move(labels));
        }
    }

    // 流插入构建的方式构建索引
    _total_elements_stored.store(0, std::memory_order_release);
    _total_elements_inserted.store(0, std::memory_order_release);
    std::ifstream feature_data_stream(
        puck::FLAGS_index_path + "/" + puck::FLAGS_feature_file_name,
        std::ios::binary | std::ios::in);
    if (!feature_data_stream.is_open()) {
        LOG(ERROR) << "open feature file error";
        return -1;
    }

    feature_data_stream.seekg(0, std::ios::end);
    size_t feature_data_stream_size = feature_data_stream.tellg();

    if(feature_data_stream_size == 0){
        save_index(puck::FLAGS_index_path + "/" + FLAGS_tinker_quantizer_file_name);
        return 0;
    }

    // 1. 获取feature file中数据条数
    int feature_data_file_dim = 0;
    feature_data_stream.seekg(0, std::ios::beg);
    feature_data_stream.read((char*)&feature_data_file_dim, sizeof(int));
    if(feature_data_file_dim != _dim){
        LOG(ERROR) << "feature data dim not match, "
                    << "feature_data_file_dim: " << feature_data_file_dim 
                    << ", _dim: " << _dim;
        return -1;
    }
    
    size_t total_fea_num = feature_data_stream_size / (sizeof(int) + sizeof(float) * _dim);
    if (total_fea_num != static_cast<size_t>(checked_feature_count)) {
        LOG(ERROR) << "feature count changed after validation: "
                   << total_fea_num << "/" << checked_feature_count;
        return -1;
    }
    if (_brief != 0 && build_labels.size() != total_fea_num) {
        LOG(ERROR) << "brief label count does not match feature count: "
                   << build_labels.size() << "/" << total_fea_num;
        return -1;
    }
    feature_data_stream.seekg(0, std::ios::beg);
    size_t cur_fea_num = 0;
    std::atomic<bool> build_failed{false};
    if(total_fea_num == 0) {
        LOG(ERROR) << "inner error: unexpect total_fea_num";
        return -1;
    }
    else{
        LOG(INFO) << "total elements to build: " << total_fea_num;
    }

    // 2. 读取部分fea data
    auto read_fea_part=
            [&total_fea_num, &cur_fea_num, &feature_data_stream, dim = _dim]() 
            -> std::pair<size_t, std::unique_ptr<float[]>>{
                constexpr size_t nread_max = 1 << 12;
                size_t nread = 
                    total_fea_num - cur_fea_num > nread_max ? 
                    nread_max : total_fea_num - cur_fea_num;
                if(nread == 0){
                    return {0, nullptr};
                }
                std::pair<size_t, std::unique_ptr<float[]>> ret = {nread, nullptr};
                ret.second.reset(new float[nread * dim]);
                for(size_t i = 0; i < nread; ++i) {
                    int record_dim = 0;
                    feature_data_stream.read(reinterpret_cast<char*>(&record_dim), sizeof(record_dim));
                    if (!feature_data_stream || record_dim != dim) {
                        LOG(ERROR) << "invalid feature dimension while building, record="
                                   << cur_fea_num + i << ", dim=" << record_dim;
                        return {0, nullptr};
                    }
                    feature_data_stream.read(reinterpret_cast<char*>(ret.second.get()) + i * sizeof(float) * dim, 
                                             sizeof(float) * dim);  // feature_data: dim * float32
                    if (!feature_data_stream) {
                        LOG(ERROR) << "read feature data failed while building, record="
                                   << cur_fea_num + i;
                        return {0, nullptr};
                    }
                }
                // LOG(INFO) << "read feature data: [" << cur_fea_num << ", " << cur_fea_num + nread << ")";       
                return ret;
            };

    // 3. 每次读取一部分数据插入索引，直到全部处理完毕
    /// 记录插入的节点数量，用于推算当前节点是数据集的第几个节点
    std::future<std::pair<size_t, std::unique_ptr<float[]>>> read_fea_part_task = 
                std::async(std::launch::async, read_fea_part);
    // std::unique_ptr<char[]> fea_data_ptr;
    // size_t last_cur_fea_num = 0, last_nread = 0;
    do {
        auto task_result = read_fea_part_task.get();
        if (task_result.first == 0 && cur_fea_num < total_fea_num) {
            LOG(ERROR) << "feature file read stopped before expected count";
            return -1;
        }
        size_t last_cur_fea_num = cur_fea_num;
        cur_fea_num += task_result.first;
        read_fea_part_task = std::async(std::launch::async, read_fea_part);
        if(task_result.first > 0){
            // 如果初始图为空，先插一个0节点
            ssize_t pre_inserted_nodeid = -1;
            if(_total_elements_stored.load(std::memory_order_acquire) == 0){
                int insert_result = 0;
                if (_brief == 0) {
                    insert_result = insert(task_result.second.get(),
                                           last_cur_fea_num);
                } else {
                    InsertRequest request;
                    request.feature = task_result.second.get();
                    request.local_id = last_cur_fea_num;
                    request.labels = &build_labels[last_cur_fea_num];
                    insert_result = insert_filtered(&request);
                }
                if (insert_result != 0) {
                    LOG(ERROR) << "insert first build node failed";
                    return -1;
                }
                pre_inserted_nodeid = last_cur_fea_num;
            }
            // assert(_total_elements_stored > 0);
            // 注意这里omp不保证: 插入顺序 == 数据集顺序。所以额外传入一个数据集顺序的id
            #pragma omp parallel for schedule(dynamic)
            for(size_t i = 0; i < task_result.first; ++i) {
                // 跳过可预知的已被插入的0号节点
                if(pre_inserted_nodeid >= 0 && last_cur_fea_num + i == size_t(pre_inserted_nodeid)) continue;
                int insert_result = 0;
                if (_brief == 0) {
                    insert_result = insert(task_result.second.get() + i * _dim,
                                           last_cur_fea_num + i);
                } else {
                    InsertRequest request;
                    request.feature = task_result.second.get() + i * _dim;
                    request.local_id = last_cur_fea_num + i;
                    request.labels = &build_labels[last_cur_fea_num + i];
                    insert_result = insert_filtered(&request);
                }
                if (insert_result != 0) {
                    build_failed.store(true, std::memory_order_release);
                }
            }
            if (build_failed.load(std::memory_order_acquire)) {
                LOG(ERROR) << "insert build node failed";
                return -1;
            }
            LOG(INFO) << "inserted feature data: [" 
                      << last_cur_fea_num << ", " 
                      << last_cur_fea_num + task_result.first << ")";
        }
    }while(cur_fea_num < total_fea_num);
    // 尾判
    if(_total_elements_inserted.load(std::memory_order_acquire) != 
            _total_elements_stored.load(std::memory_order_acquire)){
        LOG(ERROR) << "inner error in feature insert: didn't insert all fea data";
        return -1;
    }
    LOG(INFO) << "build index done: total " 
              << _total_elements_stored.load(std::memory_order_acquire) << " elements";

    // 转存为指定量化格式
    if (save_index(puck::FLAGS_index_path + "/" +
                   FLAGS_tinker_quantizer_file_name) != 0) {
        LOG(ERROR) << "save built index failed";
        return -1;
    }
    return 0;
}

int TinkerIndex::insert(const float* fea, const size_t local_id){
    std::vector<float> preprocessed;
    if (preprocess_vector(fea, _dim, _whether_norm, &preprocessed) != 0) {
        LOG(ERROR) << "insert vector preprocessing failed, local_id=" << local_id;
        return -1;
    }
    return insert_preprocessed(preprocessed.data(), local_id);
}

int TinkerIndex::insert(const InsertRequest* request) {
    if (request == nullptr || request->feature == nullptr ||
        request->dim != static_cast<uint32_t>(_dim)) {
        LOG(ERROR) << "invalid insert request";
        return -1;
    }
    if (request->labels != nullptr && !request->labels->empty()) {
        return insert_filtered(request);
    }
    return insert(request->feature, request->local_id);
}

int TinkerIndex::insert_preprocessed(const float* fea, const size_t local_id){
    if (_brief != 0) {
        LOG(ERROR) << "brief index requires labels for insert";
        return -1;
    }
    std::lock_guard<std::mutex> insert_lock(_insert_mutex);
    unsigned int cur_global_id = 0;
    // 判断节点容量
    unsigned int expected = _total_elements_inserted.load(std::memory_order_acquire);
    unsigned int desired = 0;
    do {
        if(expected + 1 > _elements_capacity) {
            LOG(ERROR) << "tinker index no enough capacity: " << expected << "/" << _elements_capacity;
            return -1;
        }
        desired = expected + 1;
    } while(!_total_elements_inserted.compare_exchange_weak(
        expected, desired, 
        std::memory_order_release, 
        std::memory_order_acquire));
    // 新id在尾部插入
    cur_global_id = expected; 
    // 更新id映射
    if(_memory_to_local_idx[cur_global_id] != -1){
        LOG(ERROR) << "inner error in insert: id conflict";
        _total_elements_inserted.fetch_sub(1, std::memory_order_release);
        return -1;
    }
    _memory_to_local_idx[cur_global_id] = local_id;

    // 对齐并量化编码fea
    char *fea_ptr = get_node_fea(cur_global_id);
    unsigned int *link_ptr = reinterpret_cast<unsigned int*>(get_node_linklist(cur_global_id));

    QuantizerParams common_params{};
    if (_quantizer->encode(fea_ptr, fea, common_params) != 0) {
        LOG(ERROR) << "encode insert vector failed, local_id=" << local_id;
        _memory_to_local_idx[cur_global_id] = -1;
        _total_elements_inserted.fetch_sub(1, std::memory_order_release);
        return -1;
    }
    if(cur_global_id != 0) {
        std::priority_queue<std::pair<float, unsigned int>> result_heap;
        if (knn_search(fea, _ef_construction, result_heap) != 0) {
            LOG(ERROR) << "search neighbors for inserted vector failed, local_id="
                       << local_id;
            _memory_to_local_idx[cur_global_id] = -1;
            _total_elements_inserted.fetch_sub(1, std::memory_order_release);
            return -1;
        }
        if (purge_link(cur_global_id, _M, result_heap) != 0) {
            _memory_to_local_idx[cur_global_id] = -1;
            _total_elements_inserted.fetch_sub(1, std::memory_order_release);
            return -1;
        }
        link_ptr[0] = 0;
        while(!result_heap.empty()){
            std::pair<float, unsigned int> tp = result_heap.top();
            result_heap.pop();
            if (add_neighbor(cur_global_id, tp.second) != 0 ||
                add_neighbor(tp.second, cur_global_id) != 0) {
                LOG(ERROR) << "add neighbor for inserted vector failed";
                return -1;
            }
        }
    }
    _total_elements_stored.fetch_add(1, std::memory_order_release);
    return 0;
}

int TinkerIndex::insert_filtered(const InsertRequest* request) {
    if (request == nullptr) {
        LOG(ERROR) << "filtered insert request is null";
        return -1;
    }
    std::vector<float> preprocessed;
    if (preprocess_vector(request->feature, _dim, _whether_norm,
                          &preprocessed) != 0) {
        LOG(ERROR) << "filtered insert vector preprocessing failed, local_id="
                   << request->local_id;
        return -1;
    }
    InsertRequest preprocessed_request;
    preprocessed_request.feature = preprocessed.data();
    preprocessed_request.dim = request->dim;
    preprocessed_request.local_id = request->local_id;
    preprocessed_request.labels = request->labels;
    return insert_filtered_preprocessed(&preprocessed_request);
}

int TinkerIndex::insert_filtered_preprocessed(
        const InsertRequest* request) {
    if (_brief == 0 || request == nullptr || request->feature == nullptr ||
        request->labels == nullptr || request->labels->empty()) {
        LOG(ERROR) << "invalid filtered insert request";
        return -1;
    }

    std::vector<LabelId> label_ids{};
    if (get_or_create_label_id(*request->labels, &label_ids) != 0 ||
        label_ids.empty()) {
        LOG(ERROR) << "create label ids failed";
        return -1;
    }
    std::lock_guard<std::mutex> insert_lock(_insert_mutex);

    unsigned int expected = _total_elements_inserted.load(std::memory_order_acquire);
    unsigned int desired = 0;
    do {
        if (expected + 1 > _elements_capacity) {
            LOG(ERROR) << "brief index no enough capacity: "
                       << expected << "/" << _elements_capacity;
            return -1;
        }
        desired = expected + 1;
    } while (!_total_elements_inserted.compare_exchange_weak(
        expected, desired, std::memory_order_release, std::memory_order_acquire));

    const unsigned int node_id = expected;
    _memory_to_local_idx[node_id] = static_cast<int>(request->local_id);
    QuantizerParams encode_params{};
    if (_quantizer->encode(get_node_fea(node_id), request->feature,
                           encode_params) != 0) {
        LOG(ERROR) << "encode filtered insert vector failed, local_id="
                   << request->local_id;
        _memory_to_local_idx[node_id] = -1;
        _total_elements_inserted.fetch_sub(1, std::memory_order_release);
        return -1;
    }
    _node_label_ids[node_id].store(label_ids.front(),
                                    std::memory_order_release);
    std::atomic_store_explicit(
        &_node_label_id_vectors[node_id],
        std::make_shared<const std::vector<LabelId>>(label_ids),
        std::memory_order_release);
    std::unique_lock<std::mutex> label_lock(
        get_filter_label_mutex(label_ids.front()));

    unsigned int* link_ptr = reinterpret_cast<unsigned int*>(
        get_node_linklist(node_id));
    link_ptr[0] = 0;
    if (node_id != 0) {
        std::priority_queue<std::pair<float, unsigned int>> candidates;
        if (filtered_knn_search(request->feature, label_ids,
                                _ef_construction, false, candidates) != 0) {
            _node_label_ids[node_id].store(INVALID_LABEL_ID, std::memory_order_release);
            std::atomic_store_explicit(&_node_label_id_vectors[node_id],
                                       std::shared_ptr<const std::vector<LabelId>>(),
                                       std::memory_order_release);
            _memory_to_local_idx[node_id] = -1;
            _total_elements_inserted.fetch_sub(1, std::memory_order_release);
            return -1;
        }
        if (filtered_purge_link(node_id, _M, candidates) != 0) {
            _node_label_ids[node_id].store(INVALID_LABEL_ID, std::memory_order_release);
            std::atomic_store_explicit(&_node_label_id_vectors[node_id],
                                       std::shared_ptr<const std::vector<LabelId>>(),
                                       std::memory_order_release);
            _memory_to_local_idx[node_id] = -1;
            _total_elements_inserted.fetch_sub(1, std::memory_order_release);
            return -1;
        }
        while (!candidates.empty()) {
            const auto candidate = candidates.top();
            candidates.pop();
            if (add_neighbor(node_id, candidate.second) != 0 ||
                add_neighbor(candidate.second, node_id) != 0) {
                LOG(ERROR) << "add neighbor for filtered insert failed";
                return -1;
            }
        }
    }

    if (update_filter_entrypoints(label_ids, node_id) != 0) {
        LOG(ERROR) << "update filter entrypoints failed";
        return -1;
    }
    _total_elements_stored.fetch_add(1, std::memory_order_release);
    return 0;
}

int TinkerIndex::add_neighbor(const unsigned int from, const unsigned int to){
    // 跳过自环（应该不会出现这种情况）
    if(from == to){
        return 0;
    }
    if (_brief != 0 && _cartesian == 0) {
        const std::shared_ptr<const std::vector<LabelId>> to_labels =
            std::atomic_load_explicit(&_node_label_id_vectors[to],
                                      std::memory_order_acquire);
        const std::shared_ptr<const std::vector<LabelId>> from_labels =
            std::atomic_load_explicit(&_node_label_id_vectors[from],
                                      std::memory_order_acquire);
        if (to_labels == nullptr || from_labels == nullptr ||
            !node_matches_any_label(from, *to_labels) ||
            !node_matches_any_label(to, *from_labels)) {
            return 0;
        }
    } else if (_brief != 0 &&
               _node_label_ids[from].load(std::memory_order_acquire) !=
               _node_label_ids[to].load(std::memory_order_acquire)) {
        return 0;
    }
    std::shared_mutex &_from_mutex = get_node_mutex(from);

    // from节点的读锁
    std::shared_lock<std::shared_mutex> _from_readguard(_from_mutex);
    // 检查如果to已经在from的邻居中则直接返回
    unsigned int *link_ptr = reinterpret_cast<unsigned int*>(get_node_linklist(from));
    unsigned int linksize = link_ptr[0];
    for(unsigned int i = 1; i <= linksize; ++i){
        if(link_ptr[i] == to){
            return 0;
        }
    }
    _from_readguard.unlock();
    // from节点的写锁
    std::unique_lock<std::shared_mutex> _from_writeguard(_from_mutex);
    std::vector<float> buffer(_dim * 2);
    QuantizerParams common_params{.decoded_buffer = reinterpret_cast<char*>(buffer.data())};
    // 再读一遍防止TOCTOU问题
    linksize = link_ptr[0];
    if(linksize + 1 <= _M){
        link_ptr[++linksize] = to;
        link_ptr[0] = linksize;
    }
    else{
        // 触发裁边
        std::priority_queue<std::pair<float, unsigned int>> candidate_neighbors;
        float dist = _quant_sdc_distance_func(get_node_fea(from), get_node_fea(to), common_params);
        candidate_neighbors.emplace(dist, to);
        for(unsigned int i = 1; i <= linksize; ++i){
            dist = _quant_sdc_distance_func(get_node_fea(from), get_node_fea(link_ptr[i]), common_params);
            candidate_neighbors.emplace(dist, link_ptr[i]);
        }
        if (_brief != 0) {
            if (filtered_purge_link(from, _M, candidate_neighbors) != 0) {
                return -1;
            }
        } else {
            if (purge_link(from, _M, candidate_neighbors) != 0) {
                return -1;
            }
        }
        link_ptr[0] = candidate_neighbors.size();
        linksize = 0;
        while(!candidate_neighbors.empty()){
            link_ptr[link_ptr[0] - linksize] = candidate_neighbors.top().second;
            linksize++;
            candidate_neighbors.pop();
        }
    }
    return 0;
}

/*
int TinkerIndex::reorder_index(){
    LOG(INFO) << "starting reorder index...";
    std::chrono::high_resolution_clock::time_point st = std::chrono::high_resolution_clock::now();
    VisitedBitmap visited;
    (void) visited.get(_total_elements_stored - 1);
    std::unique_ptr<unsigned int[]> new_node_order(new unsigned int[_total_elements_stored]);
    std::unique_ptr<unsigned int[]> new_node_order_inv(new unsigned int[_total_elements_stored]);
    std::vector<unsigned int> _q;
    // 1. ann pack, 找到seed节点距离空间最近邻，计算新节点序
    // QuantizerParams rerank_params{};
    // size_t new_node_order_offset = 0;
    // for(unsigned int i = 0; i < _total_elements_stored; i++){
    //     if(!visited.get(i)){
    //         _q.push_back(i);
    //     }
    //     while(!_q.empty()){
    //         std::vector<unsigned int> batch;
    //         for(unsigned int id : _q){
    //             if(!visited.visit(id)){
    //                 new_node_order[new_node_order_offset++] = id;
    //                 batch.push_back(id);
    //                 if(new_node_order_offset % 1000000 == 0){
    //                     LOG(INFO) << "reorder index " << new_node_order_offset << " / " << _total_elements_stored;
    //                 }
    //             }
    //         }
    //         _q.clear();
    //         #pragma omp parallel for schedule(dynamic)
    //         for(size_t j = 0; j < batch.size(); ++j){
    //             std::priority_queue<std::pair<float, unsigned int>> result_heap;
    //             std::vector<float> feature_buffer(_dim);
    //             _quantizer->decode(feature_buffer.data(), get_node_fea(batch[j]), rerank_params);
    //             knn_search(feature_buffer.data(), {batch[j]}, _num_fea_in_page * 20, result_heap);

    //             std::vector<unsigned int> nearest_ids;
    //             while(!result_heap.empty()){
    //                 if(nearest_ids.size() < _num_fea_in_page && !visited.get(result_heap.top().second)){
    //                     nearest_ids.push_back(result_heap.top().second);
    //                 }
    //                 result_heap.pop();
    //             }
    //             std::reverse(nearest_ids.begin(), nearest_ids.end());

    //             std::unique_lock<std::mutex> _global_lockguard(_global_mutex);
    //             for(unsigned int id : nearest_ids) _q.push_back(id);
    //         }
    //     }
    // }
    size_t new_node_order_offset = 0;
    for(unsigned int i = 0; i < _total_elements_stored; ++i){
        if(visited.visit(i)){
            continue;
        }
        unsigned int counter = 0;
        counter++;
        new_node_order[new_node_order_offset++] = i;
        unsigned int *link_ptr = reinterpret_cast<unsigned int*>(get_node_linklist(i));
        unsigned int linksize = link_ptr[0];
        for(unsigned int j = 1; j <= linksize && counter < _num_fea_in_page; ++j){
            if(!visited.visit(link_ptr[j])){
                new_node_order[new_node_order_offset++] = link_ptr[j];
                counter++;
            }
        }
    }
    #pragma omp parallel for schedule(dynamic)
    for(unsigned int i = 0; i < _total_elements_stored; ++i){
        new_node_order_inv[new_node_order[i]] = i;
    }
    new_node_order.reset();
    // 2. node suffle
    // i号节点 <- new_node_order[i]号节点
    // new_node_order_inv[i]号的节点 <- i号节点
    // 3.1 更新linklist
    #pragma omp parallel for schedule(dynamic)
    for(unsigned int i = 0; i < _total_elements_stored; ++i){
        unsigned int *link_ptr = reinterpret_cast<unsigned int*>(get_node_linklist(i));
        unsigned int linksize = link_ptr[0];
        for(unsigned int j = 1; j <= linksize; ++j){
            link_ptr[j] = new_node_order_inv[link_ptr[j]];
        }
    }
    // 3.2 环形原地置换
    visited.reset();
    unsigned int _temp_id = 0;
    std::unique_ptr<char[]> _temp_buffer(new char[_data_length]);
    for(unsigned int i = 0; i < _total_elements_stored; ++i){
        unsigned int src = i, dst = new_node_order_inv[i];
        if(visited.get(dst)) continue;
        _temp_id = _memory_to_local_idx[src];
        memcpy(_temp_buffer.get(), get_node_fea(src), _data_length);
        while(!visited.get(dst) && src != dst){
            visited.visit(dst);
            std::swap(_temp_id, _memory_to_local_idx[dst]);
            std::swap_ranges(get_node_fea(dst), get_node_fea(dst) + _data_length, _temp_buffer.get());
            src = dst, dst = new_node_order_inv[src];
        }
    }
    visited.clear();
    _temp_buffer.reset();

    std::chrono::high_resolution_clock::time_point et = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::duration duration = et - st;
    LOG(INFO) << "reorder index finished in " 
              << std::chrono::duration_cast<std::chrono::seconds>(duration).count()
              << " s";
    return 0;
}
*/

int TinkerIndex::purge_link(const unsigned int id, size_t NN, 
        std::priority_queue<std::pair<float, unsigned int>> &candidate_neighbors){

    if(candidate_neighbors.size() < NN){
        return 0;
    }
    // 从近到远遍历candidates
    std::priority_queue<std::pair<float, unsigned int>> temp_heap;

    while(!candidate_neighbors.empty()){
        auto cand = candidate_neighbors.top();
        candidate_neighbors.pop();
        temp_heap.emplace(-cand.first, cand.second);
    }

    std::vector<std::pair<float, unsigned int>> result_vector;
    std::vector<float> buffer(_dim * 2);
    QuantizerParams common_params{.decoded_buffer = reinterpret_cast<char*>(buffer.data())};

    while(!temp_heap.empty()){
        if(result_vector.size() >= NN){
            break;
        }

        std::pair<float, unsigned int> cand = temp_heap.top();
        temp_heap.pop();

        bool good = true;
        for(std::pair<float, unsigned int> &nei : result_vector){
            float dist = _quant_sdc_distance_func(get_node_fea(cand.second), 
                get_node_fea(nei.second), common_params);

            if(dist * _alpha < -cand.first){
                good = false;
                break;
            }
        }
        if(good){
            // move ok
            result_vector.emplace_back(cand);
        }
    }

    // 注意result vector里面的元素距离是负数
    for(std::pair<float, unsigned int> nei : result_vector){
        candidate_neighbors.emplace(-nei.first, nei.second);
    }
    return 0;
}

int TinkerIndex::search_filtered(const Request* request, Response* response) {
    if (_brief == 0 || request == nullptr || response == nullptr ||
        request->feature == nullptr || request->labels == nullptr ||
        request->labels->empty() || request->ef_search < 0) {
        LOG(ERROR) << "invalid filtered search request";
        return -1;
    }

    const size_t search_width = resolve_search_width(request);
    std::vector<float> preprocessed;
    if (preprocess_vector(request->feature, _dim, _whether_norm,
                          &preprocessed) != 0) {
        LOG(ERROR) << "filtered search vector preprocessing failed";
        return -1;
    }

    std::vector<LabelId> query_labels;
    // 倒排表没这个label直接空结果return
    if (get_label_ids(*request->labels, &query_labels) != 0 ||
        query_labels.empty()) {
        response->result_num = 0;
        return 0;
    }

    std::priority_queue<std::pair<float, unsigned int>> result_heap;
    std::vector<unsigned int> entrypoints =
        get_filter_entrypoints(query_labels);
    // 起始点没这个label直接空结果return
    if (entrypoints.empty()) {
        response->result_num = 0;
        return 0;
    }
    if (filtered_knn_search(preprocessed.data(), query_labels,
                            search_width, true,
                            result_heap) != 0) {
        return -1;
    }

    response->result_num = std::min(request->topk,
                                    static_cast<uint32_t>(result_heap.size()));
    while (result_heap.size() > response->result_num) {
        result_heap.pop();
    }
    for (int i = static_cast<int>(response->result_num) - 1; i >= 0; --i) {
        response->distance[i] = result_heap.top().first;
        response->local_idx[i] = _memory_to_local_idx[result_heap.top().second];
        result_heap.pop();
    }
    return 0;
}

int TinkerIndex::search(const Request* request, Response* response){
    if(request == nullptr || response == nullptr){
        LOG(ERROR) << "request or response is null";
        return -1;
    }
    if(request->feature == nullptr){
        LOG(ERROR) << "feature is null";
        return -1;
    }
    if (request->ef_search < 0) {
        LOG(ERROR) << "Tinker ef_search must be non-negative";
        return -1;
    }
    if (request->labels != nullptr && !request->labels->empty()) {
        return search_filtered(request, response);
    }
    std::vector<float> preprocessed;
    if (preprocess_vector(request->feature, _dim, _whether_norm,
                          &preprocessed) != 0) {
        LOG(ERROR) << "search vector preprocessing failed";
        return -1;
    }
    if(_total_elements_stored.load() == 0){
        response->result_num = 0;
        return 0;
    }
    std::priority_queue<std::pair<float, unsigned int>> result_heap;
    const size_t search_width = resolve_search_width(request);
    if (knn_search(preprocessed.data(), search_width, result_heap) != 0) {
        return -1;
    }
    response->result_num = std::min(request->topk, uint32_t(result_heap.size()));
    while(result_heap.size() > response->result_num){
        result_heap.pop();
    }
    for(int i = response->result_num - 1; i >= 0; --i){
        response->distance[i] = result_heap.top().first;
        response->local_idx[i] = _memory_to_local_idx[result_heap.top().second];
        result_heap.pop();
    }

    return 0;
}

size_t TinkerIndex::resolve_search_width(const Request* request) const {
    return std::max(static_cast<size_t>(request->topk),
                    static_cast<size_t>(request->ef_search));
}

int TinkerIndex::knn_search(const float* query, size_t topk,
        std::priority_queue<std::pair<float, unsigned int>>& result_heap){
    std::vector<unsigned int> enterpoints;
    if (_brief == 0) {
        enterpoints.push_back(0);
    } else {
        std::shared_lock<std::shared_mutex> lock(_filter_entrypoint_mutex);
        for (const auto& item : _filter_entrypoints) {
            enterpoints.insert(enterpoints.end(), item.second.begin(),
                               item.second.end());
        }
    }
    return knn_search(query, enterpoints, topk, result_heap);
}

int TinkerIndex::knn_search(const float* query, const std::vector<unsigned int> enterpoint, size_t topk, 
        std::priority_queue<std::pair<float, unsigned int>>& result_heap){
            
    // 清空result_heap
    while(!result_heap.empty()){
        result_heap.pop();
    }
    auto _vl = _visited_pool.get_free_visitedlist();
    _vl->reset();
    std::priority_queue<std::pair<float, unsigned int>> candidate_heap;
    std::vector<float> buffer(_dim);
    QuantizerParams common_params{.decoded_buffer = reinterpret_cast<char*>(buffer.data()), .code_level = 0};
    QuantizerParams rerank_params{.decoded_buffer = reinterpret_cast<char*>(buffer.data()), .code_level = 2};

    utils::prefetch_blk(reinterpret_cast<const char*>(query), sizeof(float) * _dim);
    for(const unsigned int& ep : enterpoint){
        if(!_vl->visit(ep)){
            float dist = _quant_adc_distance_func(get_node_fea(ep), query, common_params);
            candidate_heap.emplace(-dist, ep);

            if(_quantizer->support_rerank()){
                dist = _quant_adc_distance_func(get_node_fea(ep), query, rerank_params);
            }
            result_heap.emplace(dist, ep);
        }
    }
    // std::chrono::high_resolution_clock::time_point st = std::chrono::high_resolution_clock::now();
    while(!candidate_heap.empty()){
        std::pair<float, unsigned int> cur_closest_node = candidate_heap.top();
        if(result_heap.size() >= topk && -cur_closest_node.first > result_heap.top().first){
            break;
        }
        candidate_heap.pop();
        unsigned int *cur_linklist = reinterpret_cast<unsigned int*>(get_node_linklist(cur_closest_node.second));
        std::shared_mutex& cur_mutex = get_node_mutex(cur_closest_node.second);

        // 遍历邻居更新candidate_heap
        /// 防止访问正在被修改linklist的节点
        std::shared_lock<std::shared_mutex> node_access_lockguard(cur_mutex);
        unsigned int linklist_size = cur_linklist[0];
        if (__builtin_expect(linklist_size > _M, 0)) {
            LOG(ERROR) << "invalid linklist size, node_id="
                       << cur_closest_node.second
                       << ", linklist_size=" << linklist_size
                       << ", max_size=" << _M;
            _visited_pool.release_visitedlist(_vl);
            return -1;
        }

        if(linklist_size == 0) continue;
        utils::prefetch_blk(get_node_fea(cur_linklist[1]), _fea_length);
        utils::prefetch_blk(get_node_linklist(cur_linklist[1]), _linklist_length);
        for(unsigned int i = 1; i <= linklist_size; ++i){
            // 乱序预取
            if(i + 1 <= linklist_size){
                utils::prefetch_blk(get_node_fea(cur_linklist[i + 1]), _fea_length);
                utils::prefetch_blk(get_node_linklist(cur_linklist[i + 1]), _linklist_length);
            }
            unsigned int neighbor_id = cur_linklist[i];
            if(!_vl->visit(neighbor_id)){
                const char *neighbor_fea = get_node_fea(neighbor_id);
                float dist = _quant_adc_distance_func(neighbor_fea, query, common_params);
                // distance_computations++;
                if(result_heap.size() < topk || dist < result_heap.top().first){
                    candidate_heap.emplace(-dist, neighbor_id);
                    if(_quantizer->support_rerank()){
                        dist = _quant_adc_distance_func(neighbor_fea, query, rerank_params);
                    }
                    result_heap.emplace(dist, neighbor_id);
                    if(result_heap.size() > topk){
                        result_heap.pop();
                    }
                }
            }
        }
    }
    // std::chrono::high_resolution_clock::time_point ed = std::chrono::high_resolution_clock::now();
    // timer += ed - st;
    _visited_pool.release_visitedlist(_vl);
    // LOG(INFO) << "distance_computations: " << distance_computations;
    return 0;
}

int TinkerIndex::filtered_knn_search(
        const float* query,
        const std::vector<LabelId>& query_labels,
        size_t topk,
        bool require_all_labels,
        std::priority_queue<std::pair<float, unsigned int>>& result_heap) {
    while (!result_heap.empty()) {
        result_heap.pop();
    }
    const std::vector<unsigned int> entrypoints =
        get_filter_entrypoints(query_labels);
    if (entrypoints.empty()) {
        return 0;
    }

    auto visited = _visited_pool.get_free_visitedlist();
    visited->reset();
    std::priority_queue<std::pair<float, unsigned int>> candidate_heap;
    std::vector<float> buffer(_dim);
    QuantizerParams common_params{
        .decoded_buffer = reinterpret_cast<char*>(buffer.data()), .code_level = 0};
    QuantizerParams rerank_params{
        .decoded_buffer = reinterpret_cast<char*>(buffer.data()), .code_level = 2};

    for (const unsigned int entrypoint : entrypoints) {
        if (entrypoint >= _elements_capacity || visited->visit(entrypoint) ||
            !node_matches_any_label(entrypoint, query_labels)) {
            continue;
        }
        float distance = _quant_adc_distance_func(
            get_node_fea(entrypoint), query, common_params);
        candidate_heap.emplace(-distance, entrypoint);
        if (_quantizer->support_rerank()) {
            distance = _quant_adc_distance_func(
                get_node_fea(entrypoint), query, rerank_params);
        }
        if (!require_all_labels ||
            node_matches_all_labels(entrypoint, query_labels)) {
            result_heap.emplace(distance, entrypoint);
            if (result_heap.size() > topk) {
                result_heap.pop();
            }
        }
    }

    while (!candidate_heap.empty()) {
        const auto current = candidate_heap.top();
        if (result_heap.size() >= topk &&
            -current.first > result_heap.top().first) {
            break;
        }
        candidate_heap.pop();
        const unsigned int* linklist =
            reinterpret_cast<const unsigned int*>(
                get_node_linklist(current.second));
        std::shared_lock<std::shared_mutex> lock(get_node_mutex(current.second));
        const unsigned int linklist_size = linklist[0];
        if (__builtin_expect(linklist_size > _M, 0)) {
            LOG(ERROR) << "invalid filtered linklist size, node_id="
                       << current.second
                       << ", linklist_size=" << linklist_size
                       << ", max_size=" << _M;
            _visited_pool.release_visitedlist(visited);
            return -1;
        }
        for (unsigned int i = 1; i <= linklist_size; ++i) {
            const unsigned int neighbor_id = linklist[i];
            if (visited->visit(neighbor_id) ||
                !node_matches_any_label(neighbor_id, query_labels)) {
                continue;
            }
            const char* neighbor_fea = get_node_fea(neighbor_id);
            float distance = _quant_adc_distance_func(
                neighbor_fea, query, common_params);
            candidate_heap.emplace(-distance, neighbor_id);
            if ((!require_all_labels ||
                 node_matches_all_labels(neighbor_id, query_labels)) &&
                (result_heap.size() < topk ||
                 distance < result_heap.top().first)) {
                if (_quantizer->support_rerank()) {
                    distance = _quant_adc_distance_func(
                        neighbor_fea, query, rerank_params);
                }
                result_heap.emplace(distance, neighbor_id);
                if (result_heap.size() > topk) {
                    result_heap.pop();
                }
            }
        }
    }
    _visited_pool.release_visitedlist(visited);
    return 0;
}

int TinkerIndex::filtered_purge_link(
        const unsigned int id,
        size_t max_degree,
        std::priority_queue<std::pair<float, unsigned int>>& candidate_neighbors) {
    if (_cartesian != 0 || candidate_neighbors.size() < max_degree) {
        return purge_link(id, max_degree, candidate_neighbors);
    }

    const std::shared_ptr<const std::vector<LabelId>> source_labels =
        std::atomic_load_explicit(&_node_label_id_vectors[id],
                                  std::memory_order_acquire);
    if (source_labels == nullptr) {
        return -1;
    }

    std::priority_queue<std::pair<float, unsigned int>> nearest_first;
    while (!candidate_neighbors.empty()) {
        const auto candidate = candidate_neighbors.top();
        candidate_neighbors.pop();
        nearest_first.emplace(-candidate.first, candidate.second);
    }

    std::vector<std::pair<float, unsigned int>> selected;
    std::vector<float> buffer(_dim * 2);
    QuantizerParams params{
        .decoded_buffer = reinterpret_cast<char*>(buffer.data())};
    while (!nearest_first.empty() && selected.size() < max_degree) {
        const auto candidate = nearest_first.top();
        nearest_first.pop();
        const std::shared_ptr<const std::vector<LabelId>> candidate_labels =
            std::atomic_load_explicit(
                &_node_label_id_vectors[candidate.second],
                std::memory_order_acquire);
        if (candidate_labels == nullptr) {
            continue;
        }

        std::vector<LabelId> common_labels;
        // 求source_labels和candidate_labels的公共标签
        std::set_intersection(
            source_labels->begin(), source_labels->end(),
            candidate_labels->begin(), candidate_labels->end(),
            std::back_inserter(common_labels));
        bool keep = true;
        for (const auto& neighbor : selected) {
            // 候选邻居和当前超球心没有公共标签，直接跳过
            if (!node_matches_all_labels(neighbor.second, common_labels)) {
                continue;
            }
            const float distance = _quant_sdc_distance_func(
                get_node_fea(candidate.second),
                get_node_fea(neighbor.second),
                params);
            // 三角裁边
            if (distance * _alpha < -candidate.first) {
                keep = false;
                break;
            }
        }
        if (keep) {
            selected.push_back(candidate);
        }
    }

    for (const auto& neighbor : selected) {
        candidate_neighbors.emplace(-neighbor.first, neighbor.second);
    }
    return 0;
}

inline char* TinkerIndex::get_node_fea(const unsigned int id) {
    if (__builtin_expect(
            id >= _elements_capacity ||
            _fea_ptr == nullptr ||
            _fea_length == 0,
            0)) {
        if (id >= _elements_capacity) {
            throw std::runtime_error(
                "TinkerIndex::get_node_fea out of range, id: "
                + std::to_string(id)
                + ", capacity: " + std::to_string(_elements_capacity));
        }
        throw std::runtime_error(
            "TinkerIndex::get_node_fea storage is not initialized");
    }
    const size_t offset_fea = _fea_length * id;
    return _fea_ptr.get() + offset_fea;
}

inline char* TinkerIndex::get_node_linklist(const unsigned int id) {
    if (__builtin_expect(
            id >= _elements_capacity ||
            _linklist_ptr == nullptr ||
            _linklist_length == 0,
            0)) {
        if (id >= _elements_capacity) {
            throw std::runtime_error(
                "TinkerIndex::get_node_linklist out of range, id: "
                + std::to_string(id)
                + ", capacity: " + std::to_string(_elements_capacity));
        }
        throw std::runtime_error(
            "TinkerIndex::get_node_linklist storage is not initialized");
    }
    const size_t offset_linklist = _linklist_length * id;
    return _linklist_ptr.get() + offset_linklist;
}

inline std::shared_mutex& TinkerIndex::get_node_mutex(const unsigned int id){
    unsigned int offset_node_mutex = id & NODE_HASH_LOWBIT;
    return _node_mutex[offset_node_mutex];
}

int TinkerIndex::read_index() {
    // 0. 读取量化器索引
    if (load_quantizer() != 0) {
        return -1;
    }
    _quant_sdc_distance_func = get_quant_sdc_distance_func(_quantizer);
    _quant_adc_distance_func = get_quant_adc_distance_func(_quantizer);

    if(adapt_data_width()){
        LOG(ERROR) << "data width not match";
        return -1;
    }
    LOG(INFO) << "get tinker_index meta_info: "
              << "_total_elements_stored: " << _total_elements_stored
              << ", _elements_capacity: " << _elements_capacity;

    if(_total_elements_stored > _elements_capacity){
        LOG(ERROR) << "lack of ram capacity, will use " 
                   << _data_length * _total_elements_stored
                   << ", but "
                   << FLAGS_tinker_memory_capacity
                   << " (bytes) left.";
        return -1;
    }

    _visited_pool.init(_elements_capacity);
    _node_mutex.reset(new std::shared_mutex[NODE_MUTEX_COUNT]);
    _filter_label_mutex.reset(new std::mutex[FILTER_LABEL_MUTEX_COUNT]);
    if (_brief != 0) {
        _node_label_ids.reset(new std::atomic<LabelId>[_elements_capacity]);
        _node_label_id_vectors.reset(
            new std::shared_ptr<const std::vector<LabelId>>[
                _elements_capacity]);
        for (unsigned int i = 0; i < _elements_capacity; ++i) {
            _node_label_ids[i].store(INVALID_LABEL_ID,
                                     std::memory_order_relaxed);
        }
    }

    std::string linklist_file_name = puck::FLAGS_index_path + "/" + FLAGS_tinker_file_prefix + ".relat";
    std::string feature_code_file_name = puck::FLAGS_index_path + "/" + FLAGS_tinker_file_prefix + ".feat";
    std::ifstream linklist_file(linklist_file_name, std::ios::binary);
    if(!linklist_file.is_open()){
        LOG(ERROR) << "open linklist file failed";
        return -1;
    }
    std::ifstream feature_code_file(feature_code_file_name, std::ios::binary);
    if(!feature_code_file.is_open()){
        LOG(ERROR) << "open feature file failed";
        return -1;
    }

    // 1. 检查graph文件大小
    // 1.1 tinker特征编码文件
    LOG(INFO) << "checking tinker feature file ...";
    size_t feature_code_file_size = 0;
    feature_code_file.seekg(0, std::ios::end);
    feature_code_file_size = feature_code_file.tellg();
    feature_code_file.seekg(0, std::ios::beg);

    size_t expect_feature_code_file_size = 
        _fea_length * _total_elements_stored.load(std::memory_order_acquire);
    if(feature_code_file_size != expect_feature_code_file_size){
        LOG(ERROR) << "feature code file size not match";
        return -1;
    }
    LOG(INFO) << "tinker feature file check succ, size: " << feature_code_file_size << " bytes";

    // 1.2 tinker邻居文件
    LOG(INFO) << "checking tinker linklist file ...";
    size_t linklist_file_size = 0;
    linklist_file.seekg(0, std::ios::end);
    linklist_file_size = linklist_file.tellg();
    linklist_file.seekg(0, std::ios::beg);

    size_t expect_linklist_file_size = _linklist_length * _total_elements_stored.load(std::memory_order_acquire);
    if(linklist_file_size != expect_linklist_file_size){
        LOG(ERROR) << "linklist file size not match";
        return -1;
    }
    LOG(INFO) << "tinker linklist file check succ, size: " << linklist_file_size << " bytes";

    // 2. 读取tinker graph索引
    // 2.1 读取特征编码文件
    LOG(INFO) << "reading tinker feature file ...";
    _fea_ptr.reset();
    _fea_ptr.reset(new char[_fea_length * _elements_capacity]());
    feature_code_file.read(_fea_ptr.get(), expect_feature_code_file_size);
    feature_code_file.close();
    LOG(INFO) << "read tinker feature succ.";

    // 2.2 读取邻居文件
    LOG(INFO) << "reading tinker linklist file ...";
    _linklist_ptr.reset();
    _linklist_ptr.reset(new char[_linklist_length * _elements_capacity]());
    linklist_file.read(_linklist_ptr.get(), expect_linklist_file_size);
    linklist_file.close();
    LOG(INFO) << "read tinker linklist succ.";

    // 3. 读取id mapper文件
    LOG(INFO) << "reading tinker id mapper file ...";
    std::string id_mapper_file_path = puck::FLAGS_index_path + "/" + FLAGS_tinker_idmapper_file_name;
    std::ifstream id_mapper_file(id_mapper_file_path, std::ios::binary);
    if(!id_mapper_file.is_open()){
        LOG(ERROR) << "open id mapper file failed";
        return -1;
    }
    _memory_to_local_idx.clear();
    _memory_to_local_idx.shrink_to_fit();
    _memory_to_local_idx.resize(_elements_capacity, -1);

    id_mapper_file.read(reinterpret_cast<char*>(_memory_to_local_idx.data()), 
        sizeof(int) * _total_elements_stored.load(std::memory_order_acquire));

    if(!id_mapper_file.good()){
        LOG(ERROR) << "read id mapper file failed";
        return -1;
    }
    id_mapper_file.close();
    LOG(INFO) << "read tinker id mapper succ.";

    LOG(INFO) << "read tinker index succ. size: "
              << _total_elements_stored.load(std::memory_order_acquire)
              << " remain capacity: "
              << _elements_capacity;
    
    _total_elements_inserted.store(
        _total_elements_stored.load(std::memory_order_acquire), 
        std::memory_order_release
    );
    return 0;
}

int TinkerIndex::save_filter_metadata() const {
    if (_brief == 0) {
        return 0;
    }
    const std::string path = puck::FLAGS_index_path + "/" +
                             FLAGS_tinker_file_prefix + ".filter_meta";
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        LOG(ERROR) << "open filter metadata file failed";
        return -1;
    }

    const uint32_t label_count = static_cast<uint32_t>(_id_to_label.size());
    const uint32_t node_count = _total_elements_stored.load(
        std::memory_order_acquire);
    file.write(reinterpret_cast<const char*>(&label_count), sizeof(label_count));
    file.write(reinterpret_cast<const char*>(&node_count), sizeof(node_count));
    for (const std::string& label : _id_to_label) {
        const uint32_t length = static_cast<uint32_t>(label.size());
        file.write(reinterpret_cast<const char*>(&length), sizeof(length));
        file.write(label.data(), length);
    }
    for (uint32_t node_id = 0; node_id < node_count; ++node_id) {
        const std::shared_ptr<const std::vector<LabelId>> label_ids =
            std::atomic_load_explicit(&_node_label_id_vectors[node_id],
                                      std::memory_order_acquire);
        if (label_ids == nullptr || label_ids->empty()) {
            LOG(ERROR) << "missing node labels";
            return -1;
        }
        if (_cartesian != 0) {
            const LabelId label_id = label_ids->front();
            file.write(reinterpret_cast<const char*>(&label_id),
                       sizeof(label_id));
        } else {
            const uint32_t count = static_cast<uint32_t>(label_ids->size());
            file.write(reinterpret_cast<const char*>(&count), sizeof(count));
            file.write(reinterpret_cast<const char*>(label_ids->data()),
                       sizeof(LabelId) * count);
        }
    }
    const uint32_t entrypoint_count = static_cast<uint32_t>(
        _filter_entrypoints.size());
    file.write(reinterpret_cast<const char*>(&entrypoint_count),
               sizeof(entrypoint_count));
    for (const auto& item : _filter_entrypoints) {
        const LabelId label_id = item.first;
        const uint32_t count = static_cast<uint32_t>(item.second.size());
        file.write(reinterpret_cast<const char*>(&label_id), sizeof(label_id));
        file.write(reinterpret_cast<const char*>(&count), sizeof(count));
        file.write(reinterpret_cast<const char*>(item.second.data()),
                   sizeof(unsigned int) * count);
    }
    if (!file.good()) {
        LOG(ERROR) << "write filter metadata failed";
        return -1;
    }
    return 0;
}

int TinkerIndex::load_filter_metadata() {
    const std::string path = puck::FLAGS_index_path + "/" +
                             FLAGS_tinker_file_prefix + ".filter_meta";
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        LOG(ERROR) << "open filter metadata file failed";
        return -1;
    }
    uint32_t label_count = 0;
    uint32_t node_count = 0;
    file.read(reinterpret_cast<char*>(&label_count), sizeof(label_count));
    file.read(reinterpret_cast<char*>(&node_count), sizeof(node_count));
    if (!file.good() || node_count != _total_elements_stored.load(
            std::memory_order_acquire)) {
        LOG(ERROR) << "invalid filter metadata header";
        return -1;
    }
    std::unordered_map<std::string, LabelId> label_to_id;
    std::vector<std::string> id_to_label;
    std::unordered_map<LabelId, std::vector<unsigned int>> label_to_nodes;
    std::unordered_map<LabelId, std::vector<unsigned int>> filter_entrypoints;
    for (uint32_t i = 0; i < label_count; ++i) {
        uint32_t length = 0;
        file.read(reinterpret_cast<char*>(&length), sizeof(length));
        std::string label(length, '\0');
        file.read(label.data(), length);
        if (!file.good()) {
            return -1;
        }
        label_to_id.emplace(label, i);
        id_to_label.push_back(label);
    }
    for (uint32_t node_id = 0; node_id < node_count; ++node_id) {
        std::vector<LabelId> node_labels;
        if (_cartesian != 0) {
            LabelId label_id = 0;
            file.read(reinterpret_cast<char*>(&label_id), sizeof(label_id));
            if (!file.good() || label_id >= label_count) {
                LOG(ERROR) << "invalid node label id";
                return -1;
            }
            node_labels.push_back(label_id);
        } else {
            uint32_t count = 0;
            file.read(reinterpret_cast<char*>(&count), sizeof(count));
            if (!file.good() || count == 0) {
                LOG(ERROR) << "invalid node label count";
                return -1;
            }
            node_labels.resize(count);
            file.read(reinterpret_cast<char*>(node_labels.data()),
                      sizeof(LabelId) * count);
            if (!file.good()) {
                return -1;
            }
            for (LabelId label_id : node_labels) {
                if (label_id >= label_count) {
                    LOG(ERROR) << "invalid node label id";
                    return -1;
                }
            }
        }
        _node_label_ids[node_id].store(node_labels.front(),
                                       std::memory_order_release);
        std::atomic_store_explicit(
            &_node_label_id_vectors[node_id],
            std::make_shared<const std::vector<LabelId>>(node_labels),
            std::memory_order_release);
        for (LabelId label_id : node_labels) {
            label_to_nodes[label_id].push_back(node_id);
        }
    }
    uint32_t entrypoint_count = 0;
    file.read(reinterpret_cast<char*>(&entrypoint_count), sizeof(entrypoint_count));
    for (uint32_t i = 0; i < entrypoint_count; ++i) {
        LabelId label_id = 0;
        uint32_t count = 0;
        file.read(reinterpret_cast<char*>(&label_id), sizeof(label_id));
        file.read(reinterpret_cast<char*>(&count), sizeof(count));
        std::vector<unsigned int>& entrypoints = filter_entrypoints[label_id];
        entrypoints.resize(count);
        file.read(reinterpret_cast<char*>(entrypoints.data()),
                  sizeof(unsigned int) * count);
        for (const unsigned int entrypoint : entrypoints) {
            if (label_id >= label_count || entrypoint >= node_count ||
                !node_matches_any_label(entrypoint, {label_id})) {
                LOG(ERROR) << "invalid filter entrypoint";
                return -1;
            }
        }
    }
    if (!file.good()) {
        LOG(ERROR) << "read filter metadata failed";
        return -1;
    }
    _label_to_id = std::move(label_to_id);
    _id_to_label = std::move(id_to_label);
    _label_to_nodes = std::move(label_to_nodes);
    _filter_entrypoints = std::move(filter_entrypoints);
    return 0;
}

int TinkerIndex::save_index() {
    // reorder_index();
    if(save_model_file() != 0){
        LOG(ERROR) << "save model file failed";
        return -1;
    }

    std::string linklist_file_name = puck::FLAGS_index_path + "/" + FLAGS_tinker_file_prefix + ".relat";
    std::string feature_code_file_name = puck::FLAGS_index_path + "/" + FLAGS_tinker_file_prefix + ".feat";
    std::ofstream linklist_file(linklist_file_name, std::ios::binary);
    if(!linklist_file.is_open()){
        LOG(ERROR) << "open linklist file failed";
        return -1;
    }
    std::ofstream feature_code_file(feature_code_file_name, std::ios::binary);
    if(!feature_code_file.is_open()){
        LOG(ERROR) << "open feature file failed";
        return -1;
    }
    feature_code_file.write(
        _fea_ptr.get(), 
        _fea_length * _total_elements_stored.load(std::memory_order_acquire)
    );
    if(!feature_code_file.good()){
        LOG(ERROR) << "write feature file failed";
        return -1;
    }
    feature_code_file.close();
    LOG(INFO) << "save tinker feature succ.";

    linklist_file.write(_linklist_ptr.get(), 
        _linklist_length * _total_elements_stored.load(std::memory_order_acquire)
    );
    if(!linklist_file.good()){
        LOG(ERROR) << "write linklist file failed";
        return -1;
    }
    linklist_file.close();
    LOG(INFO) << "save tinker linklist succ.";

    std::string id_mapper_file_path = puck::FLAGS_index_path + "/" + FLAGS_tinker_idmapper_file_name;
    std::ofstream id_mapper_file(id_mapper_file_path, std::ios::binary);
    if(!id_mapper_file.is_open()){
        LOG(ERROR) << "open id mapper file failed";
        return -1;
    }
    id_mapper_file.write(
        reinterpret_cast<char*>(_memory_to_local_idx.data()), 
        sizeof(int) * _total_elements_stored.load(std::memory_order_acquire)
    );
    if(!id_mapper_file.good()){
        LOG(ERROR) << "write id mapper file failed";
        return -1;
    }
    id_mapper_file.close();
    LOG(INFO) << "save tinker id mapper succ.";
    if (_brief != 0 && save_filter_metadata() != 0) {
        LOG(ERROR) << "save filter metadata failed";
        return -1;
    }
    LOG(INFO) << "save tinker index succ.";
    return 0;
}

int TinkerIndex::save_quantizer() {
    if (!_quantizer) {
        LOG(ERROR) << "quantizer is null, cannot save";
        return -1;
    }
    std::string quantizer_path = puck::FLAGS_index_path + "/" + FLAGS_tinker_quantizer_file_name;
    if (_quantizer->save(quantizer_path) != 0) {
        LOG(ERROR) << "save quantizer failed, path=" << quantizer_path;
        return -1;
    }
    LOG(INFO) << "save quantizer succ, path=" << quantizer_path;
    return 0;
}

int TinkerIndex::load_quantizer() {
    std::string quantizer_path = puck::FLAGS_index_path + "/" + FLAGS_tinker_quantizer_file_name;
    if (load_quantizer_dynamic(quantizer_path, _quantizer,
                               _whether_norm) != 0) {
        LOG(ERROR) << "load quantizer failed, path=" << quantizer_path;
        return -1;
    }
    LOG(INFO) << "load quantizer succ, path=" << quantizer_path;
    return 0;
}

int TinkerIndex::save_index(const std::string &auxiliary_quant_file){
    // reorder_index();
    // 1. 读取辅助量化文件
    std::unique_ptr<Quantizer> quantizer_aux;
    int ret = load_quantizer_dynamic(auxiliary_quant_file, quantizer_aux,
                                     _whether_norm);
    if(ret != 0){
        LOG(ERROR) << "load auxiliary quantizer failed";
        return -1;
    }

    // 2. 索引转存
    // 2.1 meta data
    if(save_model_file() != 0){
        LOG(ERROR) << "save model file failed";
        return -1;
    }

    if(!_quantizer){
        LOG(ERROR) << "quantizer is null, please load index first";
        return -1;
    }

    size_t data_length = _quantizer->data_length();         // 当前索引量化器特征下编码长度
    size_t data_length_aux = quantizer_aux->data_length();  // 待保存的量化器下特征编码长度

    std::string linklist_file_name = puck::FLAGS_index_path + "/" + FLAGS_tinker_file_prefix + ".relat";
    std::string feature_code_file_name = puck::FLAGS_index_path + "/" + FLAGS_tinker_file_prefix + ".feat";
    std::ofstream linklist_file(linklist_file_name, std::ios::binary);
    if(!linklist_file.is_open()){
        LOG(ERROR) << "open linklist file failed";
        return -1;
    }
    std::ofstream feature_code_file(feature_code_file_name, std::ios::binary);
    if(!feature_code_file.is_open()){
        LOG(ERROR) << "open feature file failed";
        return -1;
    }

    // 2.2 decode -> encode -> save
    std::vector<std::unique_ptr<char[]>> decoded_buffer_thread_data(omp_get_max_threads());
    for(auto &decoded_buffer : decoded_buffer_thread_data){
        decoded_buffer.reset(new char[_dim * sizeof(float)]);
    }
    std::vector<std::unique_ptr<char[]>> encoded_buffer_thread_data(omp_get_max_threads());
    for(auto &encoded_buffer : encoded_buffer_thread_data){
        encoded_buffer.reset(new char[data_length_aux]);
    }
    
    // 2.3 save feature code in new quantizer
    // 默认参数，最高精度解码
    QuantizerParams common_params{};

    static constexpr size_t process_batch = 1 << 12;
    std::unique_ptr<char[]> write_buffer(std::make_unique<char[]>(data_length_aux * process_batch));
    unsigned int total_stored = _total_elements_stored.load(std::memory_order_acquire);
    for(size_t i = 0; i < total_stored; i += process_batch){
        size_t true_process_num = std::min(process_batch, total_stored - i);
        // 并行编解码 id \in [i, i + true_process_num)
        #pragma omp parallel for schedule(dynamic)
        for(size_t j = 0; j < true_process_num; j++){
            std::unique_ptr<char[]> &decoded_buffer = decoded_buffer_thread_data[omp_get_thread_num()];
            std::unique_ptr<char[]> &encoded_buffer = encoded_buffer_thread_data[omp_get_thread_num()];

            char *single_data = get_node_fea(i + j);
            char *ptr = single_data;    // ptr now refer to old encoded data
            size_t ptr_data_length = data_length;
            if(_quantizer->need_recode()){
                _quantizer->decode(reinterpret_cast<float*>(decoded_buffer.get()), ptr, common_params);
                ptr = decoded_buffer.get();
                ptr_data_length = _dim * sizeof(float);
            }
            // ptr now refer to decoded data
            if(quantizer_aux->need_recode()){
                quantizer_aux->encode(encoded_buffer.get(), reinterpret_cast<float*>(ptr), common_params);
                ptr = encoded_buffer.get();
                ptr_data_length = data_length_aux;
            }
            // ptr now refer to new encoded data
            if(ptr_data_length != data_length_aux){
                throw std::runtime_error("TinkerIndex::save_index: data length mismatch in recoding");
            }
            memcpy(write_buffer.get() + j * data_length_aux, ptr, ptr_data_length);
        }
        // dump to file
        feature_code_file.write(write_buffer.get(), data_length_aux * true_process_num);
        if(!feature_code_file.good()){
            LOG(ERROR) << "write feature file failed";
            return -1;
        }
    }
    feature_code_file.close();
    LOG(INFO) << "save tinker feature data succ.";

    // 2.4 save linklist
    linklist_file.write(
        _linklist_ptr.get(), 
        sizeof(uint32_t) * (_M + 1) * _total_elements_stored.load(std::memory_order_acquire)
    );
    if(!linklist_file.good()){
        LOG(ERROR) << "write linklist file failed";
        return -1;
    }
    linklist_file.close();
    LOG(INFO) << "save tinker linklist data succ.";

    // 3. id mapper与原来一致
    std::string id_mapper_file_path = puck::FLAGS_index_path + "/" + FLAGS_tinker_idmapper_file_name;
    std::ofstream id_mapper_file(id_mapper_file_path, std::ios::binary);
    if(!id_mapper_file.is_open()){
        LOG(ERROR) << "open id mapper file failed";
        return -1;
    }
    id_mapper_file.write(
        reinterpret_cast<char*>(_memory_to_local_idx.data()), 
        sizeof(int) * _total_elements_stored.load(std::memory_order_acquire)
    );

    if(!id_mapper_file.good()){
        LOG(ERROR) << "write id mapper file failed";
        return -1;
    }
    id_mapper_file.close();
    LOG(INFO) << "save tinker id mapper succ.";
    if (_brief != 0 && save_filter_metadata() != 0) {
        LOG(ERROR) << "save filter metadata failed";
        return -1;
    }
    LOG(INFO) << "save tinker index succ.";

    return 0;
}

size_t TinkerIndex::size(){
    // 原子读取已插入元素数量（包含未计算完毕的节点）
    return static_cast<size_t>(_total_elements_inserted.load(std::memory_order_acquire));
}

int TinkerIndex::create_empty_index() {
    LOG(INFO) << "Creating empty TinkerIndex in memory";
    
    // 检查量化器类型，仅支持 FLAT
    if (FLAGS_tinker_quantizer_type != "FLAT") {
        LOG(ERROR) << "create_empty_index only supports FLAT quantizer, "
                   << "but tinker_quantizer_type=" << FLAGS_tinker_quantizer_type;
        return -1;
    }
    
    // 使用 FLAT 量化器（仅在内存中初始化，不保存）
    if (make_quantizer_dynamic("FLAT", _quantizer, _dim, _whether_norm) != 0 ||
        _quantizer->train() != 0) {
        LOG(ERROR) << "initialize FLAT quantizer failed";
        return -1;
    }
    
    // 初始化距离函数
    _quant_sdc_distance_func = get_quant_sdc_distance_func(_quantizer);
    _quant_adc_distance_func = get_quant_adc_distance_func(_quantizer);
    
    // 计算数据宽度
    if (adapt_data_width() != 0) {
        LOG(ERROR) << "adapt data width error";
        return -1;
    }
    
    // 先释放旧内存，避免新旧共存导致内存峰值
    _fea_ptr.reset();
    _linklist_ptr.reset();
    _memory_to_local_idx.clear();
    _memory_to_local_idx.shrink_to_fit();

    // 分配内存
    _fea_ptr.reset(new char[_fea_length * _elements_capacity]());
    _linklist_ptr.reset(new char[_linklist_length * _elements_capacity]());
    _visited_pool.init(_elements_capacity);
    _node_mutex.reset(new std::shared_mutex[NODE_MUTEX_COUNT]);
    _filter_label_mutex.reset(new std::mutex[FILTER_LABEL_MUTEX_COUNT]);
    if (_brief != 0) {
        _node_label_ids.reset(new std::atomic<LabelId>[_elements_capacity]);
        _node_label_id_vectors.reset(
            new std::shared_ptr<const std::vector<LabelId>>[
                _elements_capacity]);
        for (unsigned int i = 0; i < _elements_capacity; ++i) {
            _node_label_ids[i].store(INVALID_LABEL_ID,
                                     std::memory_order_relaxed);
        }
    }
    _memory_to_local_idx.resize(_elements_capacity, -1);
    _label_to_id.clear();
    _id_to_label.clear();
    _label_to_nodes.clear();
    _filter_entrypoints.clear();
    
    // 初始化为空索引
    _total_elements_stored.store(0, std::memory_order_release);
    _total_elements_inserted.store(0, std::memory_order_release);
    _enterpoint_id = 0;
    
    LOG(INFO) << "Empty TinkerIndex created in memory, capacity=" << _elements_capacity;
    return 0;
}

int TinkerIndex::reset() {
    LOG(INFO) << "Resetting TinkerIndex...";

    // 重置计数器
    _total_elements_stored.store(0, std::memory_order_release);
    _total_elements_inserted.store(0, std::memory_order_release);
    _enterpoint_id = 0;

    // 释放内存
    _fea_ptr.reset();
    _linklist_ptr.reset();
    _memory_to_local_idx.clear();
    _memory_to_local_idx.shrink_to_fit();
    _node_label_ids.reset();
    _node_label_id_vectors.reset();
    _filter_label_mutex.reset();
    _label_to_id.clear();
    _id_to_label.clear();
    _label_to_nodes.clear();
    _filter_entrypoints.clear();

    // 重置 visited_pool
    _visited_pool.init(_elements_capacity);

    LOG(INFO) << "TinkerIndex reset completed, capacity=" << _elements_capacity;
    return 0;
}

} // namespace puck
