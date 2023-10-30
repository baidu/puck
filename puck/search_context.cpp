// Copyright (c) 2023 Baidu, Inc.  All Rights Reserved.
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
 * @file search_context.cpp
 * @author huangben@baidu.com
 * @date 2018/8/13 19:50
 * @brief
 *
 **/
#include <malloc.h>
#include <unistd.h>
#include <glog/logging.h>
#include "puck/search_context.h"
//#define _aligned_malloc(size, alignment) aligned_alloc(alignment, size)

namespace puck {
extern const u_int64_t cache_offset_size;
SearchContext::SearchContext() :
        _logid(0),
        //_topk(0),
        _debug(false),
        _inited(false),
        _model(nullptr),
        _selector(nullptr),
        _visited_list(nullptr),
        _log_string("") {}

SearchContext::~SearchContext() {
    if (_model) {
        free(_model);
        _search_cell_data.init();
        _search_point_data.init();
    }
}

int SearchContext::reset(const IndexConf& conf) {
    _logid = 0;
    _log_string = "";
    _debug = false;

    if (_inited == false || _init_conf.filter_topk < conf.filter_topk ||
        _init_conf.search_coarse_count < conf.search_coarse_count ||
        _init_conf.neighbors_count < conf.neighbors_count) {
        _inited = false;
        _init_conf = conf;

        if (_model) {
            free(_model);
            _search_cell_data.init();
            _search_point_data.init();
        }
    }

    //_topk = conf.topk;
    //二级聚类中心最多需要top-neighbors_count个cell，空间多申请一些可避免频繁更新tag idx
    unsigned int all_cells_cnt = 1;

    // if (conf.index_type == IndexType::PUCK ||  conf.index_type ==
    // IndexType::HIERARCHICAL_CLUSTER) {
    if (conf.index_type == IndexType::HIERARCHICAL_CLUSTER) {
        all_cells_cnt = conf.neighbors_count * 1.1;

        if (all_cells_cnt > conf.search_coarse_count * conf.fine_cluster_count) {
            all_cells_cnt = conf.search_coarse_count * conf.fine_cluster_count;
        }

        if (_search_cell_data.cell_distance.size() != all_cells_cnt) {
            _search_cell_data.cell_distance.resize(all_cells_cnt);
        }
    }

    if (_inited) {
        return 0;
    }
    if (_visited_list == nullptr) {
        _visited_list = new similarity::VisitedList(
                conf.coarse_cluster_count * conf.fine_cluster_count + conf.coarse_cluster_count +
                conf.fine_cluster_count);
    }
    size_t model_size = 0;
    //每个过程需要的内存
    // coarse
    size_t coarse_ip_dist = sizeof(float) * conf.coarse_cluster_count;
    size_t coarse_heap_size = (sizeof(float) + sizeof(uint32_t)) * conf.search_coarse_count;
    size_t stage_coarse = coarse_ip_dist + coarse_heap_size;

    model_size = std::max(model_size, stage_coarse);
    // fine
    size_t fine_ip_dist = sizeof(float) * conf.fine_cluster_count;
    size_t fine_ip_heap_size = (sizeof(float) + sizeof(uint32_t)) * conf.fine_cluster_count;

    size_t stage_fine = coarse_heap_size + fine_ip_dist + fine_ip_heap_size;

    // model_size = std::max(model_size, stage_fine);
    if (conf.whether_filter) {
        // filter
        size_t filter_dist_table = sizeof(float) * conf.filter_nsq * conf.ks;
        size_t filter_heap = (sizeof(float) + sizeof(uint32_t)) * conf.filter_topk;
        size_t pq_reorder = (sizeof(float) + sizeof(uint32_t)) * conf.filter_nsq;
        size_t stage_filter = pq_reorder + filter_dist_table + filter_heap;
        // model_size = std::max(model_size, stage_filter);

        if (conf.whether_pq) {
            // rank
            size_t pq_dist_table = sizeof(float) * conf.nsq * conf.ks;
            size_t pq_stage = pq_dist_table + filter_heap;
            // model_size = std::max(model_size, pq_stage);
            stage_filter = std::max(stage_filter, pq_stage);
        }

        stage_fine += stage_filter;
    }

    // LOG(INFO)<<"stage_fine="<<stage_fine;
    model_size = std::max(model_size, stage_fine);
    // LOG(INFO)<<"model_size="<<model_size;
    model_size += sizeof(float) * conf.feature_dim;

    void* memb = nullptr;
    int32_t pagesize = getpagesize();

    size_t size = model_size + (pagesize - model_size % pagesize);
    // LOG(INFO) << pagesize << " " << model_size << " " << size;
    int err = posix_memalign(&memb, pagesize, size);

    if (err != 0) {
        std::runtime_error("alloc_aligned_mem_failed errno=" + errno);
        return -1;
    }

    _model = reinterpret_cast<char*>(memb);
    _search_cell_data.query_norm = (float*)_model;
    char* temp = _model + sizeof(float) * conf.feature_dim;

    _search_cell_data.coarse_distance = (float*)temp;
    temp += sizeof(float) * conf.search_coarse_count;

    _search_cell_data.coarse_tag = (uint32_t*)temp;
    temp += sizeof(uint32_t) * conf.search_coarse_count;

    _search_cell_data.cluster_inner_product = (float*)temp;
    temp += sizeof(float) * std::max(conf.fine_cluster_count, conf.coarse_cluster_count);

    _search_cell_data.fine_distance = (float*)temp;
    temp += sizeof(float) * conf.fine_cluster_count;

    _search_cell_data.fine_tag = (uint32_t*)temp;

    temp += sizeof(uint32_t) * conf.fine_cluster_count;
    // temp = _model + sizeof(float) * conf.feature_dim;

    if (conf.whether_filter) {
        _search_point_data.result_distance = (float*)temp;
        temp += sizeof(float) * conf.filter_topk;

        _search_point_data.result_tag = (uint32_t*)temp;
        temp += sizeof(uint32_t) * conf.filter_topk;

        _search_point_data.pq_dist_table = (float*)temp;
        temp += sizeof(uint32_t) * conf.filter_nsq * conf.ks;

        _search_point_data.query_sorted_tag = (uint32_t*)temp;
        temp += sizeof(uint32_t) * conf.filter_nsq;

        _search_point_data.query_sorted_dist = (float*)temp;
    }

    _inited = true;

    return 0;
}

}  // namespace puck
/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
