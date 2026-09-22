// Copyright (c) 2026 Baidu, Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Regression tests for the PQ quantization subspace partition.
//
// Covers the guard restored from commit 68409db: when nsq is configured so
// that the last subspace would be 0-dimensional (lsq * (nsq - 1) >= dim),
// QuantizationParams::init must converge nsq down to a legal value. A
// 0-dim subspace made the per-subspace residual copy/clamp in puck_assign
// and PQ training underflow and read/write out of bounds.

#include <cstdint>

#include <gtest/gtest.h>

#include "interface/index_conf.h"
#include "quantizer/PQ.h"

namespace puck {
namespace {

IndexConf make_conf(uint32_t feature_dim, uint32_t nsq, uint32_t filter_nsq) {
    IndexConf conf;
    conf.ks = 256;
    conf.feature_dim = feature_dim;
    conf.nsq = nsq;
    conf.filter_nsq = filter_nsq;
    return conf;
}

// 收敛后任意子空间都不得为0维：lsq * (nsq - 1) >= dim 意味着末个子空间为0维
void expect_no_empty_subspace(const QuantizationParams& params) {
    EXPECT_LT(params.lsq * (params.nsq - 1), params.dim);
}

TEST(QuantizationParamsTest, ConvergesNsqWhenLastSubspaceWouldBeEmpty) {
    QuantizationParams params;
    const IndexConf conf = make_conf(128, 100, 64);  // lsq=2, 2*99>=128 → 需收敛
    ASSERT_EQ(params.init(conf), 0);
    expect_no_empty_subspace(params);
    EXPECT_LT(params.nsq, 100);  // nsq 从100收敛到合法值
}

TEST(QuantizationParamsTest, KeepsValidPartitionsUnchanged) {
    for (uint32_t nsq : {64u, 128u}) {  // dim=128: lsq=2/1，末子空间均≥1维
        QuantizationParams params;
        const IndexConf conf = make_conf(128, nsq, 64);
        ASSERT_EQ(params.init(conf), 0);
        expect_no_empty_subspace(params);
        EXPECT_EQ(params.nsq, nsq);
    }
}

TEST(QuantizationParamsTest, ConvergesFilterNsq) {
    QuantizationParams params;
    const IndexConf conf = make_conf(256, 64, 200);  // filter 路径：filter_nsq=200 → 需收敛
    ASSERT_EQ(params.init(conf, /*is_filter=*/true), 0);
    expect_no_empty_subspace(params);
    EXPECT_LT(params.nsq, 200);
}

}  // namespace
}  // namespace puck
