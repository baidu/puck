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
 * @file visited_list.h
 * @author hananoyuuki@qq.com
 * @date 2024/10/9 20:06
 * @brief dynamic bitmap for visited list
 *
 **/

#pragma once

#include <memory>
#include <vector>
#include <algorithm>
#include <limits>
#include <mutex>
#include <concepts>
#include <omp.h>
#include <glog/logging.h>

namespace puck {
class VisitedList{
public:
    VisitedList(size_t init_size = 0) : val(0){
        if(init_size){
            _data.resize(init_size + 1);
        }
    };
    ~VisitedList(){};

    bool visit(uint32_t id){
        if(id >= _data.size()){
            throw std::runtime_error("visitedlist::visit out of range");
        }
        bool ret = (_data[id] == val);
        _data[id] = val;
        return ret;
    }

    void reset(){
        val++; // 自然溢出
        if(val == 0){
            val = 1;
            memset(_data.data(), 0, _data.size() * sizeof(unsigned char));
        }
    }

    bool get(uint32_t id){
        if(id >= _data.size()){
            throw std::runtime_error("visitedlist::get out of range");
        }
        bool ret = (_data[id] == val);
        return ret;
    }

    void resize(uint32_t n){
        _data.resize(n);
    }

    void clear(){
        val = 0;
        _data.clear();
        _data.shrink_to_fit();
    }

    size_t size() const {
        return _data.size();
    }
private:
    unsigned char val;
    std::vector<unsigned char> _data;
};

// bitmap形式的vl，内存占用较小，初始化速度较慢
class VisitedBitmap{
public:
    VisitedBitmap(size_t init_size){
        _data.resize((init_size + compressed_size - 1) / compressed_size);
    }
    ~VisitedBitmap(){};

    bool visit(uint32_t id){
        size_t compressed_id  = id / compressed_size;
        size_t compressed_off = id % compressed_size;
        if(compressed_id >= _data.size()) {
            throw std::runtime_error("visitedlist::visit out of range");
        }
        bool ret = _data[compressed_id] & (1ul << compressed_off);

        _data[compressed_id] |= (1ul << compressed_off);
        return ret;
    }

    void reset(){
        std::fill(_data.begin(), _data.end(), 0);
    }

    void clear(){
        _data.clear();
        _data.shrink_to_fit();
    }

    bool get(uint32_t id){
        size_t compressed_id = id / compressed_size;
        size_t compressed_off = id % compressed_size;
        if(compressed_id >= _data.size()){
            throw std::runtime_error("visitedlist::get out of range");
        }
        bool ret = _data[compressed_id] & (1ul << compressed_off);
        
        return ret;
    }

    void resize(uint32_t n){
        _data.resize((n + compressed_size - 1) / compressed_size);
    }

    size_t size() const {
        return _data.size() * compressed_size;
    }

private:
    static constexpr uint32_t compressed_size = sizeof(uint64_t) << 3;    

    std::vector<uint64_t> _data;
};

template<typename T>
concept VL_t = requires(T t){
    { t.visit(0) } -> std::same_as<bool>;
    { t.get(0) } -> std::same_as<bool>;
    { t.resize(0) };
    { t.clear() };
    { t.reset() };
    { t.size() } -> std::same_as<size_t>;
};

template <VL_t T>
class VisitedPool{
public:
    VisitedPool(){};
    ~VisitedPool(){
        while(!_pool.empty()){
            T *vl = _pool.front();
            delete vl;
            _pool.pop_front();
        }
    };

    void init(size_t init_size = 0){
        _init_size = init_size;
    }

    T* get_free_visitedlist(size_t expect_size = 0){
        T *ret;
        std::unique_lock<std::mutex> lock(_mtx);
        if(!_pool.empty()){
            ret = _pool.front();
            _pool.pop_front();
        }
        else{
            ret = new T(_init_size);
        }
        if(expect_size > 0 && ret->size() < expect_size){
            ret->resize(expect_size);
        }
        return ret;
    }

    void release_visitedlist(T *vl){
        std::unique_lock<std::mutex> lock(_mtx);
        _pool.push_front(vl);
    }

private:
    std::deque<T*> _pool;
    std::mutex _mtx;
    size_t _init_size;
};

}  // namespace puck
