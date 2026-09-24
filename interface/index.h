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
 * @file index.h
 * @author huangben@baidu.com
 * @author yinjie06@baidu.com
 * @date 2022/10/25  11:49
 * @brief
 *
 **/
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "gflags/puck_gflags.h"
namespace puck {

struct Request;
struct Response;
struct InsertRequest;

void InitializeLogger(int choice = 0);
class Index {
public:
    Index() {}
    virtual ~Index() {}
    /*
    * @brief 根据配置文件修改conf、初始化内存、加载索引文件
    * @@return (int) : 正常返回0，错误返回值<0
    **/
    virtual int init() = 0;

    /*
     * @brief 检索最近的topk个样本
     * @@param [in] request : request
     * @@param [out] response : response
     * @@return (int) : 正常返回0，错误返回值<0
     **/
    virtual int search(const Request* request, Response* response) = 0;

    // Incremental insertion is optional for index implementations.
    virtual int insert(const InsertRequest* /*request*/) {
        return -1;
    }

    /*
    * @brief 训练
    * @@return (int) : 正常返回0，错误返回值<0
    **/
    virtual int train() = 0;
    /*
     * @brief 建库
     * @@return (int) : 正常返回0，错误返回值<0
     **/
    virtual int build() = 0;
};

struct Request {
public:
    uint32_t topk;         //最大不超过GFLAG_topk
    const float* feature;  //query feature
    int64_t ef_search;     // Tinker：0使用topk，正数使用max(ef_search, topk)，负数拒绝；HC/Puck忽略
    const std::vector<std::string>* labels;  // nullptr或空表示普通搜索
    Request() : topk(100), feature(nullptr), ef_search(0), labels(nullptr) {
    }
    virtual ~Request() {
        feature = nullptr;
    }
};

struct InsertRequest {
    std::string label;               // Realtime Puck使用的单值label
    const float* feature;            // 待插入的特征向量
    uint32_t dim;
    size_t local_id;                 // Tinker返回给调用方的本地ID
    const std::vector<std::string>* labels;  // Tinker过滤标签
    InsertRequest()
        : label(), feature(nullptr), dim(0), local_id(0), labels(nullptr) {
    }
};

struct Response {
    float* distance;
    uint32_t* local_idx;
    uint32_t result_num;   //实际返回的结果数,极其特殊场景下会小于设置的topk
    Response(): distance(nullptr), local_idx(nullptr) {}
    virtual ~Response() {
        distance = nullptr;
        local_idx = nullptr;
        result_num = 0;
    }
};

enum class IndexType {
    UNKNOWN = -1,
    HIERARCHICAL_CLUSTER = 0,
    PUCK = 1,
    TINKER = 2
};

IndexType load_index_type();

}//namespace puck
