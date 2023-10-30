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
 * @file py_puck_api_wrapper.cpp
 * @author huangben@baidu.com
 * @author yinjie06@baidu.com
 * @date 2021/8/18 14:30
 * @brief
 *
 **/
#include <gflags/gflags.h>
#include <glog/logging.h>
#include "pyapi_wrapper/py_api_wrapper.h"
#include "puck/gflags/puck_gflags.h"
#include "puck/puck/puck_index.h"
#include "puck/tinker/tinker_index.h"
#include "puck/hierarchical_cluster/hierarchical_cluster_index.h"
#include "puck/puck/multi_brief_puck_index.h"

namespace py_puck_api {

DEFINE_int32(index_type, 1, "");
void update_gflag(const char* gflag_key, const char* gflag_val) {
    google::SetCommandLineOption(gflag_key, gflag_val);
}

void PySearcher::show() {

    std::cout << "1\n";
}
PySearcher::PySearcher() {};

int PySearcher::build(uint32_t total_cnt) {
    std::cout << "start to train\n";

    if (FLAGS_index_type == int(puck::IndexType::TINKER)) { //Tinker
        LOG(INFO) << "init index of Tinker";
        _index.reset(new puck::TinkerIndex());
    } else if (FLAGS_index_type == int(puck::IndexType::PUCK)) { //PUCK
        LOG(INFO) << "init index of Puck";
        _index.reset(new puck::PuckIndex());
    } else if (FLAGS_index_type == int(puck::IndexType::HIERARCHICAL_CLUSTER)) {
        _index.reset(new puck::HierarchicalClusterIndex());
        LOG(INFO) << "init index of Flat";
    } else if (FLAGS_index_type == int(puck::IndexType::MULTI_BRIEF_PUCK_INDEX)) {
        _index.reset(new puck::MultiBriefPuckIndex());
        LOG(INFO) << "init index of MultiBriefPuckIndex";
    } 
    else {
        LOG(INFO) << "init index of Error, Nan type";
        return -1;
    }

    LOG(INFO) << "start to train";

    if (_index->train() != 0) {
        LOG(ERROR) << "train Faild";
        return -1;
    }

    LOG(INFO) << "train Suc.\n";

    LOG(INFO) << "start to build\n";

    if (_index->build() != 0) {
        LOG(ERROR) << "build Faild";
        return -1;

    }

    return 0;
}

int PySearcher::init() {

    puck::IndexType index_type = puck::load_index_type();
    index_type = puck::IndexType::MULTI_BRIEF_PUCK_INDEX;
    if (index_type == puck::IndexType::TINKER) { //Tinker
        LOG(INFO) << "init index of Tinker";
        _index.reset(new puck::TinkerIndex());
    } else if (index_type == puck::IndexType::PUCK) { //PUCK
        LOG(INFO) << "init index of Puck";
        _index.reset(new puck::PuckIndex());
    } else if (index_type == puck::IndexType::HIERARCHICAL_CLUSTER) {
        _index.reset(new puck::HierarchicalClusterIndex());
        LOG(INFO) << "init index of Flat";
    } else if (index_type == puck::IndexType::MULTI_BRIEF_PUCK_INDEX) {
        _index.reset(new puck::MultiBriefPuckIndex());
        LOG(INFO) << "init index of MultiBriefPuckIndex";
    } 
    
     else {
        LOG(INFO) << "init index of Error, Nan type";
        return -1;
    }

    if (_index->init() != 0) {
        LOG(ERROR) << "load index Faild";
        return -1;
    }

    puck::IndexConf conf = puck::load_index_conf_file();
    _dim = conf.feature_dim;

    if (conf.ip2cos) {
        --_dim;
    }

    return 0;
}

template <class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
    std::vector<std::thread> threads;
    std::atomic<size_t> current(start);

    std::exception_ptr lastException = nullptr;
    std::mutex lastExceptMutex;

    for (size_t threadId = 0; threadId < numThreads; ++threadId) {
        threads.push_back(std::thread([&, threadId] {
            while (true) {
                size_t id = current.fetch_add(1);

                if ((id >= end)) {
                    break;
                }

                try {
                    fn(id, threadId);
                } catch (...) {
                    std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                    lastException = std::current_exception();
                    /*
                    * This will work even when current is the largest value that
                    * size_t can fit, because fetch_add returns the previous value
                    * before the increment (what will result in overflow
                    * and produce 0 instead of current + 1).
                    */
                    current = end;
                    break;
                }
            }
        }));
    }

    for (auto& thread : threads) {
        thread.join();
    }

    if (lastException) {
        std::rethrow_exception(lastException);
    }
}

int PySearcher::search(uint32_t n, const float* query_fea, const uint32_t topk, float* distance,
                       uint32_t* labels) {

    ParallelFor(0, n, puck::FLAGS_context_initial_pool_size, [&](int id, int threadId) {
        (void)threadId;
        puck::Request request;
        puck::Response response;
        request.topk = topk;
        request.feature = query_fea + id * _dim;

        response.distance = distance + id * topk;
        response.local_idx = labels + id * topk;
        _index->search(&request, &response);
    });

    return 0;
}
int PySearcher::filter_search(uint32_t n, const float* query_fea, const uint32_t topk, float* distances,
                              uint32_t* labels,
                              int* indptr,   int* indices) {
    
    ParallelFor(0, n, puck::FLAGS_context_initial_pool_size, [&](int id, int threadId) {
        (void)threadId;
        puck::BriefRequest request;
        puck::Response response;
        size_t cur_id = id;
        request.topk = topk;
        request.feature = query_fea + cur_id * _dim;
        request.briefs = indices + indptr[cur_id];
        request.brief_size = indptr[cur_id + 1] - indptr[cur_id];

        response.distance = distances + cur_id * topk;
        response.local_idx = labels + cur_id * topk;
        _index->search(&request, &response);
    });
    return 0;
}
PySearcher::~PySearcher() {};
};//namespace py_puck_api
