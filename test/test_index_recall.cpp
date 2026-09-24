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
 * @file    test_index_recall.cpp
 * @author  yinjie06(yinjie06@baidu.com)
 * @date    2023/04/13 16:18
 * @brief
 *
 **/

#include <gtest/gtest.h>
#include <gflags/gflags.h>
#include "test/test_index.h"

DEFINE_double(base_recall_acc, 0.99, "");
DEFINE_int32(insert_thread_cnt, 50, "concurrent insert threads count");
using namespace puck;

// ============ train 阶段 ============
TEST(IndexTest, TrainPuckIndex) {
    TestIndex index;

    EXPECT_EQ(index.download_data(), 0);
    EXPECT_EQ(index.train_index(int(puck::IndexType::PUCK)), 0);
}

// ============ build 阶段：多线程并发插入建库 ============
TEST(IndexTest, BuildPuckIndex) {
    TestIndex index;

    EXPECT_EQ(index.download_data(), 0);
    EXPECT_EQ(index.insert_index(FLAGS_insert_thread_cnt), 0);
    float recall_rate = index.cmp_search_recall();
    EXPECT_GT(recall_rate, FLAGS_base_recall_acc);
}

// ============ search 阶段 ============
// pq 版 puck 召回
TEST(IndexTest, SearchPuckIndex) {
    TestIndex index;

    EXPECT_EQ(index.download_data(), 0);
    google::SetCommandLineOption("whether_pq", "true");
    EXPECT_EQ(index.build_index(int(puck::IndexType::PUCK)), 0);
    float recall_rate = index.cmp_search_recall();
    EXPECT_GT(recall_rate, FLAGS_base_recall_acc);
}

// 关 pq 的 puck 召回
TEST(IndexTest, SearchPuckFlatIndex) {
    TestIndex index;

    EXPECT_EQ(index.download_data(), 0);
    google::SetCommandLineOption("whether_pq", "false");
    EXPECT_EQ(index.build_index(int(puck::IndexType::PUCK)), 0);
    float recall_rate = index.cmp_search_recall();
    EXPECT_GT(recall_rate, FLAGS_base_recall_acc);
}

// 近零查询拒绝
TEST(IndexTest, AllZeroSearch) {
    TestIndex index;

    EXPECT_EQ(index.download_data(), 0);

    EXPECT_EQ(index.build_index(int(puck::IndexType::PUCK)), 0);
    Request req;
    std::vector<float> vec(FLAGS_feature_dim, 0);
    req.feature = vec.data();
    req.topk = FLAGS_topk;
    Response resp;
    std::vector<float> distance(FLAGS_topk);
    std::vector<uint32_t> local_idx(FLAGS_topk);
    resp.distance = distance.data();
    resp.local_idx = local_idx.data();
    EXPECT_EQ(index.search(&req, &resp), -1);
}

