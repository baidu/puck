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
#include "interface/index.h"
namespace py_puck_api {

bool update_gflag(const char* gflag_key, const char* gflag_val);
class PySearcher {
public:
    explicit PySearcher(int index_type);
    int load();
    int train();
    int build();
    int search(uint32_t n, const float* query_fea, uint32_t topk,
               int64_t ef_search, float* distance, uint32_t* labels);
    uint32_t dim() const;
    ~PySearcher();
private:
    int create_index(int index_type);
    std::unique_ptr<puck::Index> _index;
    uint32_t _dim;
    puck::IndexType _index_type;
};
};//namespace py_puck_api
