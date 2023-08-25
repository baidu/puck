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
 * @file    test_index.cpp
 * @author  yinjie06(yinjie06@baidu.com)
 * @date    2023/04/13 16:18
 * @brief
 *
 **/
#pragma once
#include <memory>
#include "puck/index.h"

namespace puck {
class TestIndex {
public:
    TestIndex() {}
    ~TestIndex() {}
    int download_data();
    int build_index(int index_type);
    int insert_index(int thread_cnt);
    float cmp_search_recall();
private:
    std::unique_ptr<Index> _index;
    std::string _query_filename;
    std::string _groundtruth_filename;
};

}
