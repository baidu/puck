//Copyright (c) 2023 Baidu, Inc.  All Rights Reserved.
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
 * @author huangben@baidu.com
 * @author yinjie06@baidu.com
 * @date 2022/5/16 10:34
 * @brief
 *
 **/
#include <queue>
#include <memory>
#include "puck/tinker/tinker_index.h"
#include "puck/gflags/puck_gflags.h"
#include "puck/search_context.h"
#include "puck/hierarchical_cluster/max_heap.h"
namespace puck {
TinkerIndex::TinkerIndex() {
    _conf.index_type = puck::IndexType::TINKER;
    std::vector<std::string> buildParams;
    int tinker_neighborhood = puck::FLAGS_tinker_neighborhood;
    std::string m_str = "M=" + std::to_string(tinker_neighborhood);
    buildParams.push_back(m_str);

    int construction = puck::FLAGS_tinker_construction;
    std::string construction_str = "efConstruction=" + std::to_string(construction);
    buildParams.push_back(construction_str);

    int index_build_thread_count = puck::FLAGS_threads_count;
    buildParams.push_back("indexThreadQty=" + std::to_string(index_build_thread_count));

    _any_params.reset(new similarity::AnyParams(buildParams));
    _space.reset(new similarity::SpaceLp<float>(2));
}

int TinkerIndex::check_index_type() {
    if (_conf.index_type != IndexType::TINKER) {
        LOG(ERROR) << "index_type is not TINKER";
        return -1;
    }

    return 0;
}

int TinkerIndex::search_top1_fine_cluster(puck::SearchContext* context, const float* feature) {
    auto& search_cell_data = context->get_search_cell_data();
    float* cluster_inner_product = search_cell_data.cluster_inner_product;

    matrix_multiplication(_fine_vocab, feature, _conf.fine_cluster_count, 1, _conf.feature_dim,
                  "TN", cluster_inner_product);
    puck::MaxHeap result_heap(_conf.fine_cluster_count, search_cell_data.fine_distance,
                              search_cell_data.fine_tag);

    for (uint32_t k = 0; k < _conf.fine_cluster_count; ++k) {
        result_heap.max_heap_update(-cluster_inner_product[k], k);
    }

    result_heap.reorder();

    std::pair<float, int> nearest_cell;
    nearest_cell.first = 1 << 20;
    //计算一级聚类中心的距离,使用最大堆
    float* coarse_distance = search_cell_data.coarse_distance;
    uint32_t* coarse_tag = search_cell_data.coarse_tag;

    for (uint32_t l = 0; l < _conf.search_coarse_count; ++l) {
        int coarse_id = coarse_tag[l];
        //计算query与当前一级聚类中心下cell的距离
        auto* cur_fine_cluster_list = _coarse_clusters[coarse_id].fine_cell_list;
        float min_dist = _coarse_clusters[coarse_id].min_dist_offset + coarse_distance[l];
        float max_stationary_dist = nearest_cell.first - coarse_distance[l] - search_cell_data.fine_distance[0];

        for (uint32_t idx = 0; idx < _conf.fine_cluster_count; ++idx) {
            if (search_cell_data.fine_distance[idx] + min_dist >= nearest_cell.first) {
                break;
            }

            uint32_t k = search_cell_data.fine_tag[idx];

            if (cur_fine_cluster_list[k].stationary_cell_dist >= max_stationary_dist) {
                continue;
            }

            float temp_dist = coarse_distance[l] + cur_fine_cluster_list[k].stationary_cell_dist +
                              search_cell_data.fine_distance[idx];

            if (temp_dist < nearest_cell.first) {
                nearest_cell.first = temp_dist;
                nearest_cell.second = coarse_id * _conf.fine_cluster_count + k;
            }
        }
    }

    return nearest_cell.second;
}

int TinkerIndex::search(const Request* request, Response* response) {
    if (request->topk > _conf.topk || request->feature == nullptr) {
        LOG(ERROR) << "topk should <= topk, topk = " << _conf.topk << ", or feature is nullptr";
        return -1;
    }

    DataHandler<SearchContext> context(_context_pool);

    if (0 != context->reset(_conf)) {
        return -1;
    }

    const float* feature = normalization(context.get(), request->feature);

    if (feature == nullptr) {
        return -1;
    }

    //输出query与一级聚类中心的top-search-cell个ID和距离
    int ret = search_nearest_coarse_cluster(context.get(), feature,
                                            _conf.search_coarse_count);

    if (ret != 0) {
        return ret;
    }

    int nearest_cell_id = search_top1_fine_cluster(context.get(), feature);
    const auto* cur_fine_cluster = get_fine_cluster(nearest_cell_id);
    std::vector<int> eps;

    for (auto i = cur_fine_cluster->memory_idx_start; i < (cur_fine_cluster + 1)->memory_idx_start; ++i) {
        eps.push_back(i);
    }

    std::priority_queue<std::pair<float, int>> closest_dist_queuei;
    _tinker_index->SearchOld_level0(feature, _conf.feature_dim,
                                    std::max(_conf.tinker_search_range, (uint32_t)request->topk),
                                    eps, closest_dist_queuei);

    while (closest_dist_queuei.size() > request->topk) {
        closest_dist_queuei.pop();
    }

    response->result_num = closest_dist_queuei.size();

    while (!closest_dist_queuei.empty()) {
        int idx = closest_dist_queuei.size() - 1;
        int cur_memory_id = closest_dist_queuei.top().second;
        float top_dist = closest_dist_queuei.top().first;
        response->distance[idx] = top_dist;
        response->local_idx[idx] = _memory_to_local[cur_memory_id];
        closest_dist_queuei.pop();
    }

    return 0;
}

int TinkerIndex::read_feature_index(uint32_t* local_to_memory_idx) {
    (void)local_to_memory_idx;
    std::string tinker_index_file = _conf.index_path + "/" + puck::FLAGS_tinker_file_name;
    LOG(INFO) << "tinker_index_file = " << tinker_index_file;
    similarity::ObjectVector object_vector;
    _tinker_index.reset(new similarity::Hnsw<float>(*_space.get(), object_vector));
    _tinker_index->LoadIndex(tinker_index_file);
    return 0;
}

int TinkerIndex::build() {
    this->HierarchicalClusterIndex::build();
    uint32_t cell_cnt = _conf.coarse_cluster_count * _conf.fine_cluster_count;
    std::vector<uint32_t> cell_start_memory_idx(cell_cnt + 1, _conf.total_point_count);
    std::vector<uint32_t> local_to_memory_idx(_conf.total_point_count);
    //读取local idx，转换memory idx
    convert_local_to_memory_idx(cell_start_memory_idx.data(), local_to_memory_idx.data());
    similarity::ObjectVector object_data(_conf.total_point_count);
    size_t datalength = _conf.feature_dim * sizeof(float);

    std::ifstream data_stream;
    data_stream.open(_conf.feature_file_name.c_str(), std::ios::binary);
    std::vector<float> temp_data_fea(_conf.feature_dim);
    int feature_dim = -1;

    for (uint64_t i = 0; i < local_to_memory_idx.size(); ++i) {
        data_stream.read((char*)&feature_dim, sizeof(int));

        if (feature_dim != (int)_conf.feature_dim) {
            LOG(INFO) << "read " << _conf.feature_file_name << " error.";
            return -1;
        }

        data_stream.read((char*)temp_data_fea.data(), sizeof(float) * _conf.feature_dim);
        uint32_t memory_idx = local_to_memory_idx[i];
        similarity::Object* cur_object = new similarity::Object(memory_idx, memory_idx, datalength,
                (void*)(temp_data_fea.data()));
        object_data[memory_idx] = cur_object;
    }
    data_stream.close();
    
    _tinker_index.reset(new similarity::Hnsw<float>(*_space.get(), object_data));
    _tinker_index->CreateIndex(*_any_params.get());
    std::string tinker_index_file = _conf.index_path + "/" + puck::FLAGS_tinker_file_name;
    _tinker_index->SaveIndex(tinker_index_file);
    return 0;
}
}//tinker
