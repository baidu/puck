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
 * @file build.cpp
 * @author huangben@baidu.com
 * @author yinjie06@baidu.com
 * @date 2019/4/16 15:36
 * @brief
 *
 **/

#include <iostream>
#include <string>
#include <cstdlib>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include "puck/tinker/tinker_index.h"
#include "puck/gflags/puck_gflags.h"
#include "puck/puck/puck_index.h"

DEFINE_int32(index_type, 1, "");
int main(int argc, char** argv) {
    //com_loadlog("./conf", "puck_log.conf");
    google::ParseCommandLineFlags(&argc, &argv, true);
    std::unique_ptr<puck::Index> index;
    
    if (FLAGS_index_type == int(puck::IndexType::TINKER)) {
        index.reset(new puck::TinkerIndex());
    } else if (FLAGS_index_type == int(puck::IndexType::PUCK)) {
        index.reset(new puck::PuckIndex());
    } else if (FLAGS_index_type == int(puck::IndexType::HIERARCHICAL_CLUSTER)) {
        index.reset(new puck::HierarchicalClusterIndex());
    } else {
        LOG(ERROR) << "index type error.\n";
        return -1;
    }

    if(index->build() != 0){
        LOG(ERROR) << "build Fail.\n";
        return -1;
    }
    LOG(INFO) << "build Suc.\n";
    return 0;
}
/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */

