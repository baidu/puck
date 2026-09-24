// Copyright (c) 2026 Baidu, Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
/**
 * @file normalization.h
 * @author Baidu
 * @brief Shared input-vector normalization for Tinker.
 */

#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

#include <glog/logging.h>

namespace puck {

/**
 * @brief Copies and optionally L2-normalizes one vector.
 * @param input Source vector, which is never modified.
 * @param dim Number of float elements in input.
 * @param whether_norm Whether L2 normalization is enabled.
 * @param output Destination vector.
 * @return 0 on success and a negative value for invalid input.
 */
inline int preprocess_vector(const float* input, size_t dim, bool whether_norm,
                             std::vector<float>* output) {
    if (input == nullptr || output == nullptr || dim == 0) {
        LOG(ERROR) << "invalid vector preprocessing arguments";
        return -1;
    }
    output->assign(input, input + dim);
    if (!whether_norm) {
        return 0;
    }

    double squared_norm = 0.0;
    for (float value : *output) {
        if (!std::isfinite(value)) {
            LOG(ERROR) << "cannot normalize vector containing non-finite value";
            return -2;
        }
        squared_norm += static_cast<double>(value) * value;
    }
    static constexpr double MIN_SQUARED_NORM = 1e-12;
    if (!std::isfinite(squared_norm) || squared_norm <= MIN_SQUARED_NORM) {
        LOG(ERROR) << "cannot normalize vector with invalid L2 norm: "
                   << std::sqrt(squared_norm);
        return -2;
    }
    const float inverse_norm = static_cast<float>(1.0 / std::sqrt(squared_norm));
    for (float& value : *output) {
        value *= inverse_norm;
    }
    return 0;
}

}  // namespace puck
