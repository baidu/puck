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
using namespace puck;

TEST(IndexTest, InsertIndex1) {
    TestIndex index;

    EXPECT_EQ(index.download_data(), 0);
    EXPECT_EQ(index.insert_index(1), 0);
    float recall_rate = index.cmp_search_recall();
    EXPECT_GT(recall_rate, FLAGS_base_recall_acc);
}

TEST(IndexTest, InsertIndex50) {
    TestIndex index;

    EXPECT_EQ(index.download_data(), 0);
    EXPECT_EQ(index.insert_index(50), 0);
    float recall_rate = index.cmp_search_recall();
    EXPECT_GT(recall_rate, FLAGS_base_recall_acc);
}

TEST(IndexTest, InsertIndex100) {
    TestIndex index;

    EXPECT_EQ(index.download_data(), 0);
    EXPECT_EQ(index.insert_index(100), 0);
    float recall_rate = index.cmp_search_recall();
    EXPECT_GT(recall_rate, FLAGS_base_recall_acc);
}

TEST(IndexTest, FlatIndex) {
    TestIndex index;

    EXPECT_EQ(index.download_data(), 0);
    EXPECT_EQ(index.build_index(0), 0);
    float recall_rate = index.cmp_search_recall();
    EXPECT_GT(recall_rate, FLAGS_base_recall_acc);
}

TEST(IndexTest, PuckIndex) {
    TestIndex index;

    EXPECT_EQ(index.download_data(), 0);

    EXPECT_EQ(index.build_index(1), 0);
    float recall_rate = index.cmp_search_recall();
    EXPECT_GT(recall_rate, FLAGS_base_recall_acc);
}

TEST(IndexTest, PuckFlatIndex) {
    TestIndex index;

    EXPECT_EQ(index.download_data(), 0);
    google::SetCommandLineOption("whether_pq", "false");

    EXPECT_EQ(index.build_index(1), 0);
    float recall_rate = index.cmp_search_recall();
    EXPECT_GT(recall_rate, FLAGS_base_recall_acc);
}

TEST(IndexTest, TinkerIndex) {
    TestIndex index;

    EXPECT_EQ(index.download_data(), 0);

    EXPECT_EQ(index.build_index(2), 0);
    float recall_rate = index.cmp_search_recall();
    EXPECT_GT(recall_rate, FLAGS_base_recall_acc);
}

