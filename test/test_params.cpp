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
 * @file    test_params.cpp
 * @author  yinjie06(yinjie06@baidu.com)
 * @date    2023/04/13 16:18
 * @brief
 *
 **/

#include <gtest/gtest.h>
#include <gflags/gflags.h>
#include "puck/index_conf.h"
#include "puck/gflags/puck_gflags.h"

TEST(IndexConfTest, FlatIndex) {
    puck::IndexConf conf;
    conf.index_type = puck::IndexType::HIERARCHICAL_CLUSTER;
    conf.adaptive_train_param();

    EXPECT_EQ(conf.whether_pq, false);
    EXPECT_EQ(conf.whether_filter, false);
    EXPECT_EQ(conf.index_type, puck::IndexType::HIERARCHICAL_CLUSTER);
}

TEST(IndexConfTest, PuckIndex) {
    google::SetCommandLineOption("whether_pq", "true");
    puck::IndexConf conf;
    conf.index_type = puck::IndexType::PUCK;
    
    conf.adaptive_train_param();

    EXPECT_EQ(conf.whether_pq, true);
    EXPECT_EQ(conf.whether_filter, true);
    EXPECT_EQ(conf.index_type, puck::IndexType::PUCK);
    EXPECT_GE(conf.radius_rate, 1.0);
}

TEST(IndexConfTest, PuckFlatIndex) {
    google::SetCommandLineOption("whether_pq", "false");

    puck::IndexConf conf;
    conf.index_type = puck::IndexType::PUCK;
    conf.adaptive_train_param();

    EXPECT_EQ(conf.whether_pq, false);
    EXPECT_EQ(conf.whether_filter, true);
    EXPECT_EQ(conf.index_type, puck::IndexType::PUCK);
    EXPECT_GE(conf.radius_rate, 1.0);
}

TEST(IndexConfTest, TinkerIndex) {
    puck::IndexConf conf;
    conf.index_type = puck::IndexType::TINKER;
    conf.adaptive_train_param();

    EXPECT_EQ(conf.whether_pq, false);
    EXPECT_EQ(conf.whether_filter, false);
    EXPECT_EQ(conf.index_type, puck::IndexType::TINKER);
}

TEST(IndexConfTest, Dataset) {
    puck::IndexConf conf;
    EXPECT_LE(conf.ip2cos, 1);

    if (conf.ip2cos == 1) {
        EXPECT_EQ(conf.feature_dim, puck::FLAGS_feature_dim + 1);
    } else if (conf.ip2cos == 0) {
        EXPECT_EQ(conf.feature_dim, puck::FLAGS_feature_dim);
    }

    EXPECT_EQ(conf.ks, 256);
    EXPECT_EQ(conf.ip2cos, puck::FLAGS_ip2cos);
    EXPECT_EQ(conf.feature_dim, puck::FLAGS_feature_dim);
    EXPECT_EQ(conf.nsq, puck::FLAGS_nsq);
    EXPECT_EQ(conf.lsq, puck::FLAGS_feature_dim / puck::FLAGS_nsq);
    EXPECT_EQ(conf.whether_pq, puck::FLAGS_whether_pq);
    EXPECT_EQ(conf.whether_norm, puck::FLAGS_whether_norm);
    EXPECT_EQ(conf.topk, puck::FLAGS_topk);
    EXPECT_EQ(conf.filter_nsq, puck::FLAGS_filter_nsq);
}

