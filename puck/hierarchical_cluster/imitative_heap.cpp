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
 * @file imitative_heap.cpp
 * @author yinjie06@baidu.com
 * @date 2019/7/12 14:54:36
 * @brief
 *
 **/
#include <algorithm>
#include <cmath>
#include <glog/logging.h>
#include "puck/hierarchical_cluster/imitative_heap.h"
#include "puck/hierarchical_cluster/hierarchical_cluster_index.h"
namespace puck {
ImitativeHeap::ImitativeHeap(const uint32_t neighbors_count, DistanceInfo& cell_distance) :
    _cell_distance(cell_distance) {
    _top_idx = 0;
    _heap_size = _cell_distance.size();
    _contain_point_cnt = 0;
    _neighbors_count = neighbors_count;
    _pivot = std::sqrt(std::numeric_limits<float>::max());
    _min_distance = _pivot;
}

uint32_t ImitativeHeap::push(const float distance, FineCluster* cell, uint32_t point_cnt) {
    //当前距离大于堆顶，返回0
    if (_pivot < distance) {
        return 0;
    }

    _cell_distance[_top_idx] = {distance, std::make_pair(cell, point_cnt)};
    _min_distance = std::min(_min_distance, distance);
    ++_top_idx;
    _contain_point_cnt += point_cnt;

    //新入堆的point个数>=_neighbors_count * 1.4 或 有越界风险,进行堆调整
    if (_contain_point_cnt >= _neighbors_count * 1.4
            || _top_idx >= _heap_size) {
        //尽量快的找到合适的堆顶
        _top_idx = imitative_heap_partition();
    }

    //入队1个cell
    return 1;
}

uint32_t ImitativeHeap::imitative_heap_partition() {
    if (_contain_point_cnt < _neighbors_count) {
        return _top_idx;
    }

    auto first = _cell_distance.begin();
    auto last = _cell_distance.begin() + _top_idx;
    float pivot = _min_distance + (_neighbors_count * 1.0 / _contain_point_cnt) * (_pivot - _min_distance);

    auto middle = std::partition(first,
    last, [pivot](const std::pair<float, std::pair<FineCluster*, uint32_t>>& a) {
        return a.first <= pivot;
    });

    //LOG(INFO)<<"imitative_heap_partition "<<_pivot;
    std::sort(middle, last);
    uint32_t tail_idx = _top_idx - 1;
    uint32_t tail_min = std::distance(first, middle);

    while (tail_idx > tail_min) {
        auto cur_point_cnt = _cell_distance[tail_idx].second.second;

        if (_contain_point_cnt - cur_point_cnt >= _neighbors_count) {
            _contain_point_cnt -= cur_point_cnt;
            --tail_idx;
        } else {
            _pivot = _cell_distance[tail_idx].first;
            return tail_idx + 1;
        }
    }

    _pivot = pivot;
    return std::distance(first, middle);
}

}
