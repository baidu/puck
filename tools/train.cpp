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
 * @file  train.cpp
 * @author huangben@baidu.com
 * @author yinjie06@baidu.com
 * @date 2019/4/16 15:35
 * @brief
 *
 **/

#include <iostream>
#include <string>
#include <cstdlib>
#include <glog/logging.h>
#include <fcntl.h>
#include <sys/types.h> 
#include <sys/stat.h>
#include "puck/tinker/tinker_index.h"
#include "puck/gflags/puck_gflags.h"
#include "puck/puck/puck_index.h"

DEFINE_int32(index_type, 1, "");
//获取文件行数，index初始化时候通过key file确定样本总个数
int getFileLineCnt(const char* fileName) {
    struct stat st;

    if (stat(fileName, &st) != 0) {
        return 0;
    }

    char buff[1024];
    sprintf(buff, "wc -l %s", fileName);

    FILE* fstream = nullptr;
    fstream = popen(buff, "r");
    int total_line_cnt = -1;

    if (fstream) {
        memset(buff, 0x00, sizeof(buff));

        if (fgets(buff, sizeof(buff), fstream)) {
            int index = strchr((const char*)buff, ' ') - buff;
            buff[index] = '\0';
            total_line_cnt =  atoi(buff);
        }
    }

    if (fstream) {
        pclose(fstream);
    }

    return total_line_cnt;
}

int main(int argc, char** argv) {
#ifndef _OPENMP
    LOG(INFO)<<"not found openmp, train & build will cost more time.";
#endif
    google::ParseCommandLineFlags(&argc, &argv, true);
    std::cout << "start to train\n";
    //puck::InitializeLogger(0);
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

    if(index->train() != 0){
        LOG(ERROR) << "train Fail.\n";
        return -1;
    }

    LOG(INFO) << "train Suc.\n";
    return 0;
}
/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */

