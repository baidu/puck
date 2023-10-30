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
 * @file search_context.h
 * @author huangben@baidu.com
 * @date 2018/8/13 19:50
 * @brief
 *
 **/
#pragma once
#include <vector>
#include "puck/index_conf.h"
#include "puck/index.h"
#include "puck/puck/multi_brief_puck_index.h"
#include "puck/tinker/method/hnsw.h"

namespace puck {
class FineCluster;
//typedef std::vector<std::pair<float, FineCluster*>>
//        DistanceInfo; //存储query到二级类聚中心距离的结构
typedef std::vector<std::pair<float, std::pair<FineCluster*, uint32_t>>> DistanceInfo;

struct SearchCellData {
    float* query_norm;                  //存储归一后的query,长度=feature_dim
    float* cluster_inner_product;       //query与聚类中心的内积,长度=max(coarse_cluster_count, fine_cluster_count)
    float* coarse_distance;             //query与一级聚类中心的距离，和coarse_tag一起在最大堆调整时使用,长度=search_coarse_count
    uint32_t* coarse_tag;              //与query距离最近的一级聚类中心的id,长度=search_coarse_count
    float* fine_distance;
    uint32_t* fine_tag;
    DistanceInfo cell_distance;
    SearchCellData() {
        init();
    }
    void init() {
        query_norm = nullptr;
        cluster_inner_product = nullptr;
        coarse_distance = nullptr;
        coarse_tag = nullptr;
        fine_distance = nullptr;
        fine_tag = nullptr;
    }
};

struct SearchPointData {
    float* result_distance;         //query与point的距离,长度=topk
    uint32_t* result_tag;          //point的local index,通过local index查cnts,长度=topk
    float* pq_dist_table;
    uint32_t* query_sorted_tag;
    float* query_sorted_dist;
    SearchPointData() {
        init();
    }
    void init() {
        result_distance = nullptr;
        result_tag = nullptr;
        pq_dist_table = nullptr;
        query_sorted_tag = nullptr;
        query_sorted_dist = nullptr;

    }
};
struct Request;
class SearchContext {
public:
    SearchContext();
    virtual ~SearchContext();

    uint64_t get_logid() {
        return _logid;
    }
    uint64_t set_logid(uint64_t logid) {
        return _logid = logid;
    }
    void set_request(const Request* request) {
        _request = request;
    }
    const Request* get_request() {
        return _request;
    }
    /**
     * @brief push notice
     */
    inline void log_push(const char* key, const char* fmt, ...);

    const std::string& get_log_string() {
        return _log_string;
    }

    /**
     * @brief 使Context对象恢复初始状态（以便重复使用）
     *
     * @return  void
     **/
    int reset(const IndexConf& conf);


    SearchCellData& get_search_cell_data() {
        return _search_cell_data;
    }

    SearchPointData& get_search_point_data() {
        return _search_point_data;
    }

    void set_debug_mode(bool debug) {
        _debug = debug;
    }
    bool debug() {
        return _debug;
    }
    IDSelector* get_selector(){
        return _selector;
    }

    void init_selector(size_t nb, const int32_t *lims, const int32_t *indices){
        if(_selector == nullptr){
            _selector = new IDSelector(nb, lims, indices);
        }
    }
    similarity::VisitedList* get_visited_list(){
        return _visited_list;
    }
private:
    uint64_t _logid;
    const Request* _request;
    bool _debug;
    bool _inited;
    char* _model;
    IDSelector* _selector;
    similarity::VisitedList* _visited_list;
    IndexConf _init_conf;
    std::string _log_string;
    SearchCellData _search_cell_data;
    SearchPointData _search_point_data;
    //DISALLOW_COPY_AND_ASSIGN(SearchContext);
};
/*
inline void SearchContext::log_push(const char* key, const char* fmt, ...) {
    if (!_debug) {
        return;
    }

    char tmp[256];
    tmp[0] = '\0';
    snprintf(tmp, 256, "[%s: %s] ", key, fmt);

    va_list args;
    va_start(args, fmt);
    base::string_vappendf(&_log_string, tmp, args);
    va_end(args);
}
*/
} //namesapce puck
/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */

