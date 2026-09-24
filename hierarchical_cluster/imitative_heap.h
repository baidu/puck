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
 * @file imitative_heap.h
 * @author yinjie06@baidu.com
 * @date 2019/7/12 14:54:36
 * @brief
 *
 **/
#pragma once
#include "puck/search_context.h"
namespace puck {
inline bool dist_cmp(const std::pair<float, FineCluster*>& x, const std::pair<float, FineCluster*>& y) {
    return x.first < y.first;
}

class ImitativeHeap {
public:
    /*
     * @brief 初始化
     * @@param [in] neighbors_count : 需要保留的样本个数
     * @@param [in] cell_distance : 记录最近的top-N个cell信息的vector
     **/
    ImitativeHeap(const uint32_t neighbors_count, DistanceInfo& cell_distance);
    ~ImitativeHeap() {}
    /*
     * @brief 插入队列，如果大于堆顶元素直接返回0，否则插入队尾并判断是否进行调整，返回1
     * @@param [in] distance : 距离
     * @@param [in] cell : cell的指针
     * @@return （uint32_t）:入队的cell个数
     **/
    uint32_t push(const float distance, FineCluster* cell, uint32_t point_cnt);
    /*
     * @brief 返回当前堆的包含的元素个数
     * @@return （uint32_t）队列长度
    **/
    inline uint32_t get_top_idx() {
        return _top_idx;
    }
    /*
    * @brief 设置队列的最大距离值
    * @@param [in] pivot : 最大的距离值
    **/
    inline void set_pivot(float pivot) {
        _pivot = pivot;
    }
    /*
    * @brief 返回当前队列的最大距离值
    * @@return : 最大的距离值
    **/
    inline float get_pivot() {
        return _pivot;
    }
private:
    /*
     * @brief 调整队列
     * @@return : 返回堆顶位置idx+1，[0,idx]内包含>=_neighbors_count个point，[0,idx)范围的_cell_distance[i].first无序
     **/
    uint32_t imitative_heap_partition();
private:
    DistanceInfo& _cell_distance;
    uint32_t _top_idx;                 //当前堆内包含元素总个数
    uint32_t _heap_size;               //堆的容量，可包含元素的最大个数
    uint32_t _contain_point_cnt;         //入队cell内包含的point个数
    uint32_t _neighbors_count;
    float _pivot;
    float _min_distance;
};
}
