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
 * @file utils.h
 * @author hananoyuuki@qq.com
 * @date 2024/9/23 20:06
 * @brief common utils for tinker
 *
 **/

#pragma once

#include <random>
#include <mutex>
#include <cassert>
#include <glog/logging.h>

namespace puck {
namespace utils{

class RandomGenerator {
public:
    // @brief 返回[0, max)区间内的随机整数
    static unsigned int randint(unsigned int max) {
        return getInstance().randint_impl(0, max);
    }

private:
    RandomGenerator() : mt_(std::random_device{}()) {}
    RandomGenerator(const RandomGenerator&) = delete;
    void operator=(const RandomGenerator&) = delete;

    static RandomGenerator& getInstance() {
        static RandomGenerator instance;
        return instance;
    }

    unsigned int randint_impl(unsigned int min, unsigned int max) {
        std::lock_guard<std::mutex> lock(mtx_);
        unsigned int ret = mt_() % (max - min) + min;
        return ret;
    }

    std::mt19937 mt_;
    std::mutex mtx_;
};

inline bool begin_with(const std::string &str, const std::string &prefix){
    return str.size() >= prefix.size() &&
           str.substr(0, prefix.size()) == prefix;
}

inline std::vector<std::string> ssplit(const std::string &str, const char delim){
    std::vector<std::string> res;
    if (str.empty()) {
        return res;
    }
    size_t pos = 0, lastPos = 0;
    while ((pos = str.find(delim, pos)) != std::string::npos) {
        res.push_back(str.substr(lastPos, pos - lastPos));
        pos++;
        lastPos = pos;
    }
    res.push_back(str.substr(lastPos));
    return res;
}

inline ssize_t read_fvecs_format(const std::string &filepath, size_t n, float *data){
    ssize_t read_num = 0;
    FILE *fd = fopen(filepath.c_str(), "r");
    if(fd == nullptr){
        LOG(ERROR) << "open file " << filepath << " failed";
        return -1;
    }
    int dim = 0;
    fread(&dim, sizeof(int), 1, fd);
    fseek(fd, 0, SEEK_SET);
    for(size_t i = 0; i < n && !feof(fd); i++){
        int dim_confirm = 0;
        fread(&dim_confirm, sizeof(int), 1, fd);
        if(dim_confirm != dim){
            LOG(ERROR) << "broken file, mismatch dimension at line " << i
                       << ", expect " << dim << ", but got " << dim_confirm;
            break;
        }
        fread(data + i * dim, sizeof(float), dim, fd);
        read_num++;
    }
    fclose(fd);
    return read_num;
}

inline void prefetch_blk(const char* addr, size_t len) noexcept {
    constexpr size_t LEN_CACHE_LINE = 64;
    size_t i = 0;
    for(; i + 4 * LEN_CACHE_LINE < len; i += 4 * LEN_CACHE_LINE){
        __builtin_prefetch(addr + i, 0, 3);
        __builtin_prefetch(addr + i + LEN_CACHE_LINE, 0, 3);
        __builtin_prefetch(addr + i + 2 * LEN_CACHE_LINE, 0, 3);
        __builtin_prefetch(addr + i + 3 * LEN_CACHE_LINE, 0, 3);
    }
    for(; i < len; i += LEN_CACHE_LINE){
        __builtin_prefetch(addr + i, 0, 3);
    }
}

} // namesapce utils
} // namespace puck
