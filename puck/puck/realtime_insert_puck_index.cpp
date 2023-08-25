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
 * @file realtime_insert_puck_index.cpp
 * @author huangben@baidu.com
 * @author yinjie06@baidu.com
 * @date 2022/12/30 16:06
 * @brief
 *
 **/
#include <algorithm>
#include <glog/logging.h>
#include "puck/hierarchical_cluster/imitative_heap.h"
#include "puck/hierarchical_cluster/max_heap.h"
#include "puck/puck/realtime_insert_puck_index.h"
#include "puck/tinker/method/hnsw_distfunc_opt_impl_inline.h"
namespace puck {
InsertFineCluster::~InsertFineCluster() {
    std::lock_guard<std::mutex> fine_lock(update_mutex);

    while (insert_points != nullptr) {
        auto* temp = insert_points;
        insert_points = insert_points->next;
        delete temp;
    }
}

InsertDataMemory::~InsertDataMemory() {
    for (auto& data : quantizations) {
        delete data;
        data = nullptr;
    }
}

RealtimeInsertPuckIndex::RealtimeInsertPuckIndex(): _insert_id(0) {
}

RealtimeInsertPuckIndex::~RealtimeInsertPuckIndex() {
    for (auto& labels : _insert_labels) {
        delete labels;
        labels = nullptr;
    }
}

int RealtimeInsertPuckIndex::reorganize_inserted_index() {
    uint64_t total_point_count = _conf.total_point_count;

    if (total_point_count == 0) {
        return 0;
    }

    if (read_model_file() != 0) {
        return -1;
    }

    QuantizationParams q_param;
    q_param.init(_conf, true);
    Quantization filter_quantization(q_param, _conf.total_point_count);
    uint64_t file_length = total_point_count * (filter_quantization.get_per_fea_len());

    if (reorganize(_conf.filter_data_file_name, file_length) != 0) {
        return -1;
    }

    if (_conf.whether_pq) {
        QuantizationParams q_param;
        q_param.init(_conf, false);
        Quantization pq_quantization(q_param, _conf.total_point_count);
        file_length = total_point_count * (pq_quantization.get_per_fea_len());

        if (reorganize(_conf.pq_data_file_name, file_length) != 0) {
            return -1;
        }
    } else {
        file_length = total_point_count * (sizeof(uint32_t) + sizeof(float) * _conf.feature_dim);
        LOG(INFO) << file_length << " " << total_point_count << " " << sizeof(uint32_t) + sizeof(
                      float) * _conf.feature_dim;

        if (reorganize(_conf.feature_file_name, file_length) != 0) {
            return -1;
        }
    }

    file_length = total_point_count * sizeof(uint32_t);

    if (reorganize(_conf.cell_assign_file_name, file_length) != 0) {
        return -1;
    }

    save_model_file();
    return 0;
}

int RealtimeInsertPuckIndex::reorganize(const std::string& filename, uint64_t file_length) {
    LOG(INFO) << "RealtimeInsertPuckIndex::reorganize " << filename << " " << file_length;

    if (check_file_length_info(filename, file_length)) {
        LOG(INFO) << "check file " << filename << " length is ok";
        return 0;
    }

    std::unique_ptr<char> buffer(new char[file_length + 1]);
    memset(buffer.get(), 0, file_length + 1);
    std::ifstream file_handle(filename.c_str(), std::ios::binary | std::ios::in);
    file_handle.read(buffer.get(), file_length);
    file_handle.close();

    std::ofstream  file_stream(filename.c_str(), std::ios::binary | std::ios::out);
    file_stream.write(buffer.get(), file_length);
    file_stream.close();
    LOG(INFO) << "updated " << filename;
    return 0;
}

int RealtimeInsertPuckIndex::read_labels() {
    LOG(INFO) << "start load index file  " << _conf.label_file_name;
    std::ifstream in_url(_conf.label_file_name);
    std::string url_line;

    if (!in_url.good()) {
        LOG(FATAL) << "open index file " << _conf.label_file_name << " error.";
        return -1;
    }

    _labels.clear();

    while (std::getline(in_url, url_line)) {
        _labels.push_back(url_line);
    }

    LOG(INFO) << "total_point_count = " << _labels.size();
    in_url.close();

    read_model_file();
    _conf.total_point_count = _labels.size();
    save_model_file();
    return 0;
}

int append_sequential_format(std::ofstream& file_stream, const uint64_t offset, char* data, size_t length) {
    file_stream.seekp(offset, std::ios::beg);
    file_stream.write(data, length);

    if (file_stream.good()) {
        file_stream.flush();
        return 0;
    }

    return -1;
}

int append_fvec_format(std::ofstream& file_stream, const uint64_t offset, char* data, uint32_t dim) {
    file_stream.seekp(offset, std::ios::beg);
    file_stream.write((char*)&dim, sizeof(uint32_t));
    file_stream.write(data, dim * sizeof(float));

    if (file_stream.good()) {
        file_stream.flush();
        return 0;
    }

    return -1;
}

int RealtimeInsertPuckIndex::append_insert_memory(uint32_t group_id) {
    std::lock_guard<std::mutex> update_lock(_update_memory_mutex);

    //内存分配以group为单位，一个group可容纳_max_insert_point个point
    if (group_id >= _insert_labels.size()) {
        _insert_labels.push_back(new std::vector<std::string>());
        _insert_labels.back()->resize(_max_insert_point);
    }

    if (group_id >= _insert_data_memorys.size()) {
        InsertDataMemory* new_data = new InsertDataMemory();
        new_data->quantizations.clear();
        new_data->quantizations.push_back(new Quantization(
                                              _filter_quantization->get_quantization_params(), _max_insert_point));

        if (new_data->quantizations.back()->load_codebooks(_conf.filter_codebook_file_name) != 0) {
            LOG(ERROR) << "read filter quantization codebook fail";
            return -2;
        }

        if (new_data->quantizations.back()->init_quantized_feature_memory() != 0) {
            LOG(ERROR) << "get_filter_quantization init_quantized_feature_memory error";
            return -1;
        }

        if (_conf.whether_pq) {
            new_data->quantizations.push_back(new Quantization(
                                                  _pq_quantization->get_quantization_params(), _max_insert_point));

            if (new_data->quantizations.back()->load_codebooks(_conf.pq_codebook_file_name) != 0) {
                LOG(ERROR) << "read pq quantization codebook fail";
                return -2;
            }

            if (new_data->quantizations.back()->init_quantized_feature_memory() != 0) {
                LOG(ERROR) << "get_filter_quantization init_quantized_feature_memory error";
                return -1;
            }
        } else {
            new_data->init_feature.resize(_conf.feature_dim * _max_insert_point * sizeof(float));
        }

        new_data->memory_to_local.resize(_max_insert_point);
        _insert_data_memorys.push_back(new_data);
    }

    return 0;
}

int RealtimeInsertPuckIndex::init() {
    //获取样本总数
    if (read_labels() != 0) {
        return -1;
    }

    //文件长度检查
    if (reorganize_inserted_index() != 0) {
        return -1;
    }

    //加载存量索引
    if (this->PuckIndex::init() != 0) {
        return -1;
    }

    uint32_t cell_cnt = _conf.coarse_cluster_count * _conf.fine_cluster_count;
    _insert_fine_cluster.reset(new InsertFineCluster[cell_cnt]);

    for (uint32_t idx = 0; idx < cell_cnt; ++idx) {
        _insert_fine_cluster.get()[idx].point_cnt = get_fine_cluster(idx)->get_point_cnt();
    }

    //insert数据落盘句柄
    _index_file_handle.reset(new IndexFileHandle(_conf));

    if (_index_file_handle->init() != 0) {
        return -1;
    }

    if (append_insert_memory(0) != 0) {
        return -1;
    }

    if (_fine_norms == nullptr) {
        _fine_norms = new float[_conf.fine_cluster_count];

        for (uint32_t j = 0; j < _conf.fine_cluster_count; ++j) {
            _fine_norms[j] = cblas_sdot(_conf.feature_dim, _fine_vocab + _conf.feature_dim * j, 1,
                                        _fine_vocab + _conf.feature_dim * j, 1) / 2;

            _fine_norms[j] = 0 - std::sqrt(_fine_norms[j]);
        }
    }

    return 0;
}

int RealtimeInsertPuckIndex::insert(const InsertRequest* request) {
    if (request == nullptr || request->feature == nullptr) {
        return -1;
    }

    puck::BuildInfo build_info;
    build_info.feature.resize(_conf.feature_dim);
    //LOG(INFO) << "_conf.feature_dim = " << _conf.feature_dim;
    memcpy(build_info.feature.data(), request->feature, sizeof(float) * _conf.feature_dim);

    //特征归一 + 计算最近的cell
    if (this->HierarchicalClusterIndex::single_build(&build_info) != 0) {
        return -1;
    }

    //计算量化特征 + 写内存，cur_insert_id是insert部分内存的idx，在文件中的顺序可能会不一样（前面可能有样本写失败）
    uint32_t cur_insert_id = _insert_id.fetch_add(1);
    uint32_t group_id = cur_insert_id / _max_insert_point;
    uint32_t offset_id = cur_insert_id % _max_insert_point;

    if (append_insert_memory(group_id) != 0) {
        return -1;
    }

    auto* data_memory = _insert_data_memorys[group_id];

    if (this->PuckIndex::puck_single_assign(&build_info, data_memory->quantizations, offset_id) != 0) {
        return -1;
    }

    //先写文件
    if (append_index(request, build_info, cur_insert_id) != 0) {
        return -1;
    }

    //LOG(INFO) << request->label << " " << build_info.nearest_cell.cell_id << " " <<build_info.nearest_cell.distance;
    InsertFineCluster* cur_fine_cluster = _insert_fine_cluster.get() + build_info.nearest_cell.cell_id;
    std::lock_guard<std::mutex> fine_lock(cur_fine_cluster->update_mutex);
    InsertPoint* point = new InsertPoint(cur_insert_id);

    if (point == nullptr) {
        return -1;
    }

    point->next = cur_fine_cluster->insert_points;
    cur_fine_cluster->insert_points = point;
    cur_fine_cluster->point_cnt ++;

    //之前cell是空的
    if (cur_fine_cluster->point_cnt == 1) {
        uint32_t coarse_id = build_info.nearest_cell.cell_id / _conf.fine_cluster_count;
        uint32_t fine_id = build_info.nearest_cell.cell_id % _conf.fine_cluster_count;

        float stationary_cell_dist =  cblas_sdot(_conf.feature_dim,
                                      _coarse_vocab + coarse_id * _conf.feature_dim, 1,
                                      _fine_vocab + fine_id * _conf.feature_dim, 1);
        stationary_cell_dist -= cblas_sdot(_conf.feature_dim,
                                           _fine_vocab + fine_id * _conf.feature_dim, 1,
                                           _fine_vocab + fine_id * _conf.feature_dim, 1);
        get_fine_cluster(build_info.nearest_cell.cell_id)->stationary_cell_dist =
            stationary_cell_dist;
    }

    return 0;
}

int RealtimeInsertPuckIndex::append_index(const InsertRequest* request, puck::BuildInfo& build_info,
        uint32_t insert_id) {
    uint32_t group_idx = insert_id / _max_insert_point;
    uint32_t memory_idx = insert_id % _max_insert_point;
    InsertDataMemory* data = _insert_data_memorys[group_idx];
    //写文件加锁
    std::lock_guard<std::mutex> lk(_index_file_handle->update_file_mutex);
    //计算文件偏移量、写文件
    auto* filter_quantization = data->quantizations[0];
    uint64_t file_offset = _conf.total_point_count * (filter_quantization->get_per_fea_len());

    if (append_sequential_format(_index_file_handle->filter_quantization_file, file_offset,
                                 (char*)filter_quantization->get_quantized_feature(memory_idx),
                                 (size_t)filter_quantization->get_per_fea_len()) != 0) {
        return -1;
    }

    if (_conf.whether_pq) {
        auto* pq_quantization = data->quantizations[1];
        uint64_t file_offset = _conf.total_point_count * (pq_quantization->get_per_fea_len());

        if (append_sequential_format(_index_file_handle->pq_quantization_file, file_offset,
                                     (char*)pq_quantization->get_quantized_feature(memory_idx),
                                     (size_t)pq_quantization->get_per_fea_len()) != 0) {
            return -1;
        }
    } else {
        uint64_t file_offset = _conf.total_point_count * (sizeof(uint32_t) + sizeof(float) * _conf.feature_dim);
        float* cur_feature = data->init_feature.data() + (uint64_t)memory_idx * _conf.feature_dim;
        memcpy(cur_feature, build_info.feature.data(), sizeof(float) * _conf.feature_dim);

        if (append_fvec_format(_index_file_handle->all_feature_file, file_offset, (char*)cur_feature,
                               _conf.feature_dim) != 0) {
            return -1;
        }
    }

    file_offset = _conf.total_point_count * sizeof(uint32_t);

    if (append_sequential_format(_index_file_handle->cell_assign_file, file_offset,
                                 (char*)&build_info.nearest_cell.cell_id, sizeof(uint32_t)) != 0) {
        return -1;
    }

    std::string temp = request->label + "\n";
    _index_file_handle->label_file.write(temp.c_str(), temp.length());

    if (!_index_file_handle->label_file.good()) {
        return -1;
    }

    _index_file_handle->label_file.flush();
    //label_file文件写成功，落盘成功，更新label的内存
    _insert_labels[group_idx]->data()[memory_idx] = request->label;
    data->memory_to_local[memory_idx] = _labels.size() + insert_id;
    ++_conf.total_point_count;
    return 0;
}

int RealtimeInsertPuckIndex::compute_quantized_distance(SearchContext* context,
        const FineCluster* fine_cluster,
        const float cell_dist, MaxHeap& result_heap) {
    uint32_t  point_cnt = this->PuckIndex::compute_quantized_distance(context, fine_cluster, cell_dist,
                          result_heap);
    const float* pq_dist_table = context->get_search_point_data().pq_dist_table;
    SearchCellData& search_cell_data = context->get_search_cell_data();
    //LOG(INFO)<<(uint64_t)search_cell_data.cell_distance[cell_idx].second<<" "<<(uint64_t)_insert_fine_cluster.get()<<" "<<sizeof(InsertFineCluster);
    int cur_cell_id = fine_cluster -
                      (FineCluster*)get_fine_cluster(0);

    if (cur_cell_id < 0 || cur_cell_id > _conf.coarse_cluster_count * _conf.fine_cluster_count) {
        LOG(ERROR) << "compute_quantized_distance for insert cell has error, cell offset is " << cur_cell_id;
        return -1;
    }

    InsertFineCluster* cur_fine_cluster = _insert_fine_cluster.get() + cur_cell_id;
    auto* point = cur_fine_cluster->insert_points;
    float* result_distance = result_heap.get_top_addr();
    ///point info存储的量化特征对应的参数
    uint32_t* query_sorted_tag = context->get_search_point_data().query_sorted_tag;

    while (point != nullptr) {
        uint32_t insert_id = point->insert_id;
        point = point->next;
        ++point_cnt;
        auto* data_memory = _insert_data_memorys[insert_id / _max_insert_point];
        uint32_t offset_id = insert_id % _max_insert_point;
        auto* filter_quantization = data_memory->quantizations.front();
        const unsigned char* feature = filter_quantization->get_quantized_feature(offset_id);
        float temp_dist = 2.0 * cell_dist + ((float*)feature)[0];
        auto& quantization_params = filter_quantization->get_quantization_params();
        const unsigned char* pq_feature = (unsigned char*)feature + filter_quantization->get_fea_offset();
#ifdef __SSE__
        temp_dist += lookup_dist_table(pq_feature, pq_dist_table, quantization_params.ks, quantization_params.nsq);
#else

        for (uint32_t m = 0; m < (uint32_t)quantization_params.nsq; ++m) {
            uint32_t idx = query_sorted_tag[m];
            temp_dist += (pq_dist_table + idx * quantization_params.ks)[pq_feature[idx]];

            //当PQ子空间累计距离已经大于当前最大值，不再计算
            if (temp_dist > result_distance[0]) {
                break;
            }
        }

#endif

        if (temp_dist < result_distance[0]) {
            result_heap.max_heap_update(temp_dist, insert_id + _labels.size());
        }
    }

    return point_cnt;
}

int RealtimeInsertPuckIndex::rank_topN_points(SearchContext* context, const float* feature,
        const uint32_t filter_topk,
        MaxHeap& result_heap) {
    //LOG(INFO) << "RealtimeInsertPuckIndex::rank_topN_points";
    auto& search_point_data = context->get_search_point_data();
    float* result_distance = search_point_data.result_distance;
    uint32_t* result_tag = search_point_data.result_tag;
    float query_norm = cblas_sdot(_conf.feature_dim, feature, 1, feature, 1);
    ////真正会返回的结果存储的位置
    float* true_result_distance = result_heap.get_top_addr();

    if (_conf.whether_pq) {
        const Quantization* pq_quantization = _pq_quantization.get();
        float* pq_dist_table = context->get_search_point_data().pq_dist_table;
        pq_quantization->get_dist_table(feature, pq_dist_table);

        for (uint32_t idx = 0; idx < filter_topk; ++idx) {
            int point_id = -1;
            const unsigned char* pq_feature = nullptr;

            if (result_tag[idx] < _labels.size()) {
                pq_feature = (unsigned char*)(_pq_quantization->get_quantized_feature(result_tag[idx]));
                point_id = _memory_to_local[result_tag[idx]];
            } else {
                uint32_t group_id = (result_tag[idx] - _labels.size()) / _max_insert_point;
                uint32_t offset_id = (result_tag[idx] - _labels.size()) % _max_insert_point;
                auto* data_memory = _insert_data_memorys[group_id];
                auto* pq_quantization = data_memory->quantizations[1];
                pq_feature = (unsigned char*)pq_quantization->get_quantized_feature(offset_id);
                point_id = data_memory->memory_to_local[offset_id];
            }

            float temp_dist = result_distance[idx] - query_norm + ((float*)pq_feature)[0];
            pq_feature += pq_quantization->get_fea_offset();

            for (uint32_t m = 0; m < (uint32_t)pq_quantization->get_quantization_params().nsq
                    && temp_dist < true_result_distance[0]; ++m) {
                temp_dist += (pq_dist_table + m * pq_quantization->get_quantization_params().ks)[pq_feature[m]];
            }

            if (temp_dist < true_result_distance[0]) {
                result_heap.max_heap_update(temp_dist, point_id);
            }
        }
    } else {
        size_t qty_16 = 16;
        _mm_prefetch((char*)(feature), _MM_HINT_T0);

        for (uint32_t idx = 0; idx < filter_topk; ++idx) {
            uint32_t m = 0;
            float PORTABLE_ALIGN32 TmpRes[8];
            const float* exhaustive_feature = nullptr;
            int point_id = -1;

            if (result_tag[idx] < _labels.size()) {
                exhaustive_feature = _all_feature + (uint64_t)result_tag[idx] * _conf.feature_dim;
                point_id = _memory_to_local[result_tag[idx]];
            } else {
                uint32_t group_id = (result_tag[idx] - _labels.size()) / _max_insert_point;
                auto* data_memory = _insert_data_memorys[group_id];
                uint32_t offset_id = (result_tag[idx] - _labels.size()) % _max_insert_point;
                exhaustive_feature = data_memory->init_feature.data() + offset_id * _conf.feature_dim;
                point_id = data_memory->memory_to_local[offset_id];
            }

            float temp_dist = 0;

            for (; (m + qty_16) <= _conf.feature_dim && temp_dist < true_result_distance[0];) {
                temp_dist += similarity::L2Sqr16Ext(exhaustive_feature + m, feature + m, qty_16, TmpRes);
                m += qty_16;
            }

            if (temp_dist < true_result_distance[0]) {
                size_t left_dim = _conf.feature_dim - m;

                if (left_dim > 0) {
                    temp_dist += similarity::L2SqrExt(exhaustive_feature + m, feature + m, left_dim, TmpRes);
                }

                result_heap.max_heap_update(temp_dist, point_id);
            }
        }
    }

    result_heap.reorder();
    return 0;
}

int RealtimeInsertPuckIndex::get_label(const uint32_t label_id, std::string& label) {
    if (label_id < _labels.size()) {
        //LOG(INFO) << "label_id = " << label_id << ", " << _labels[label_id];
        label = _labels[label_id];
        return 0;
    }

    uint32_t true_label_id = label_id - _labels.size();
    uint32_t group_id = true_label_id / _max_insert_point;

    if (group_id >= _insert_labels.size()) {
        return -1;
    }

    label = _insert_labels[group_id]->data()[true_label_id % _max_insert_point];
    return 0;
}

IndexFileHandle::~IndexFileHandle() {
    close_handle(cell_assign_file);
    close_handle(label_file);
    close_handle(pq_quantization_file);
    close_handle(filter_quantization_file);
    close_handle(all_feature_file);
}

int IndexFileHandle::init() {
    if (open_handle(cell_assign_file, conf.cell_assign_file_name) == false) {
        return -1;
    }

    if (open_handle(label_file, conf.label_file_name) == false) {
        return -1;
    }

    if (open_handle(filter_quantization_file, conf.filter_data_file_name) == false) {
        return -1;
    }

    if (conf.whether_pq && open_handle(pq_quantization_file, conf.pq_data_file_name) == false) {
        return -1;
    }

    if (conf.whether_pq == false && open_handle(all_feature_file, conf.feature_file_name) == false) {
        return -1;
    }

    return 0;
}

bool IndexFileHandle::open_handle(std::ofstream& file_stream, std::string& file_name) {
    file_stream.open(file_name, std::ios::binary | std::ios::app);

    if (file_stream.is_open()) {
        return true;
    }

    return false;
}

bool IndexFileHandle::close_handle(std::ofstream& file_stream) {
    if (file_stream.is_open()) {
        file_stream.close();
    }

    return true;
}

}//namespace puck
