// Copyright (c) 2024 Baidu, Inc. All Rights Reserved.
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

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "interface/index.h"

namespace puck {
namespace {

class IndexTypeTest : public testing::Test {
protected:
    void SetUp() override {
        _index_path = std::filesystem::temp_directory_path()
                / "puck_index_type_test";
        std::filesystem::remove_all(_index_path);
        std::filesystem::create_directories(_index_path);
        google::SetCommandLineOption("index_path", _index_path.c_str());
        google::SetCommandLineOption("index_file_name", "index.dat");
    }

    void TearDown() override {
        std::filesystem::remove_all(_index_path);
    }

    void write_framed_model(IndexType index_type) const {
        const std::filesystem::path model_path = _index_path / "index.dat";
        const size_t payload_size = sizeof(IndexType);
        std::ofstream output(model_path, std::ios::binary);
        output.write(reinterpret_cast<const char*>(&payload_size),
                     sizeof(payload_size));
        output.write(reinterpret_cast<const char*>(&index_type),
                     sizeof(index_type));
    }

    void write_unframed_model(IndexType index_type) const {
        const std::filesystem::path model_path = _index_path / "index.dat";
        std::ofstream output(model_path, std::ios::binary);
        output.write(reinterpret_cast<const char*>(&index_type),
                     sizeof(index_type));
    }

    std::filesystem::path _index_path;
};

TEST_F(IndexTypeTest, LoadsFramedPuckModel) {
    write_framed_model(IndexType::PUCK);
    EXPECT_EQ(load_index_type(), IndexType::PUCK);
}

TEST_F(IndexTypeTest, LoadsFramedHierarchicalClusterModel) {
    write_framed_model(IndexType::HIERARCHICAL_CLUSTER);
    EXPECT_EQ(load_index_type(), IndexType::HIERARCHICAL_CLUSTER);
}

TEST_F(IndexTypeTest, RejectsUnframedModel) {
    write_unframed_model(IndexType::PUCK);
    EXPECT_EQ(load_index_type(), IndexType::UNKNOWN);
}

}  // namespace
}  // namespace puck
