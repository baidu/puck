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
 * @file py_puck_api_wrapper.h
 * @author huangben@baidu.com
 * @author yinjie06@baidu.com
 * @date 2021/8/18 14:30
 * @brief
 *
 **/
#pragma once

#include <iostream>
#include <memory>
#include "puck/index.h"
namespace py_puck_api {

void update_gflag(const char* gflag_key, const char* gflag_val);
class PySearcher {
public:
    PySearcher();
    void show();
    int init();
    int build(uint32_t n);
    int search(uint32_t n, const float* query_fea, const uint32_t topk, float* distance, uint32_t* labels);
    int filter_search(uint32_t n, const float* x, const uint32_t topk, float* distances, uint32_t* labels,
                      int* indptr,  int* indices);
    ~PySearcher();
private:
    std::unique_ptr<puck::Index> _index;
    uint32_t _dim;
};
};//namespace py_puck_api

