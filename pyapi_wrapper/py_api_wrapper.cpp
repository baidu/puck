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
#include <algorithm>
#include "pyapi_wrapper/py_api_wrapper.h"
#include "gflags/puck_gflags.h"
#include "puck/puck_index.h"
#include "tinker/tinker_index.h"
#include "hierarchical_cluster/hierarchical_cluster_index.h"
namespace py_puck_api {

bool update_gflag(const char* gflag_key, const char* gflag_val) {
    return !google::SetCommandLineOption(gflag_key, gflag_val).empty();
}

PySearcher::PySearcher(int index_type)
        : _dim(puck::FLAGS_feature_dim), _index_type(puck::IndexType::UNKNOWN) {
    create_index(index_type);
}

int PySearcher::create_index(int index_type) {
    if (index_type == int(puck::IndexType::TINKER)) {
        LOG(INFO) << "init index of Tinker";
        _index.reset(new puck::TinkerIndex());
    } else if (index_type == int(puck::IndexType::PUCK)) {
        LOG(INFO) << "init index of Puck";
        _index.reset(new puck::PuckIndex());
    } else if (index_type == int(puck::IndexType::HIERARCHICAL_CLUSTER)) {
        _index.reset(new puck::HierarchicalClusterIndex());
        LOG(INFO) << "init index of HierarchicalCluster";
    } else {
        LOG(ERROR) << "invalid index type: " << index_type;
        return -1;
    }
    _index_type = static_cast<puck::IndexType>(index_type);
    return 0;
}

int PySearcher::train() {
    if (!_index) {
        return -1;
    }
    LOG(INFO) << "start to train";
    if (_index->train() != 0) {
        LOG(ERROR) << "train failed";
        return -1;
    }
    _dim = puck::FLAGS_feature_dim;
    return 0;
}

int PySearcher::build() {
    if (!_index) {
        return -1;
    }
    LOG(INFO) << "start to build";
    if (_index->build() != 0) {
        LOG(ERROR) << "build failed";
        return -1;
    }
    return 0;
}

int PySearcher::load() {

    puck::IndexType index_type = puck::load_index_type();

    if (create_index(static_cast<int>(index_type)) != 0) {
        return -1;
    }

    if (_index->init() != 0) {
        LOG(ERROR) << "load index Faild";
        return -1;
    }

    if (index_type == puck::IndexType::TINKER) {
        _dim = static_cast<puck::TinkerIndex*>(_index.get())->dim();
    } else {
        puck::IndexConf conf = puck::load_index_conf_file();
        _dim = conf.feature_dim;
        if (conf.ip2cos) {
            --_dim;
        }
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

int PySearcher::search(uint32_t n, const float* query_fea, uint32_t topk,
                       int64_t ef_search, float* distance, uint32_t* labels) {
    if (!_index || !query_fea || !distance || !labels || n == 0 || topk == 0 || _dim == 0) {
        return -1;
    }
    if (ef_search < 0) {
        LOG(ERROR) << "ef_search must be non-negative";
        return -1;
    }

    std::atomic<int> search_result{0};
    const size_t thread_count = std::max(1, puck::FLAGS_context_initial_pool_size);
    ParallelFor(0, n, thread_count, [&](int id, int threadId) {
        (void)threadId;
        puck::Request request;
        puck::Response response;
        request.topk = topk;
        request.feature = query_fea + id * _dim;
        request.ef_search = ef_search;

        response.distance = distance + id * topk;
        response.local_idx = labels + id * topk;
        if (_index->search(&request, &response) != 0) {
            search_result.store(-1, std::memory_order_release);
        }
    });

    return search_result.load(std::memory_order_acquire);
}
uint32_t PySearcher::dim() const { return _dim; }
PySearcher::~PySearcher() {};
};//namespace py_puck_api
