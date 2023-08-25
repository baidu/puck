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
 * @file max_heap.cpp
 * @author huangben@baidu.com
 * @author yinjie06@baidu.com
 * @date 2019/8/20 10:43
 * @brief
 *
 **/
#include <memory>
#include <iostream>
#include <glog/logging.h>
#include "puck/hierarchical_cluster/max_heap.h"
namespace puck {

MaxHeap::MaxHeap(uint32_t size, float* val, uint32_t* tag) :
    _heap_val(val), _heap_tag(tag), _heap_size(size) {
    _default_point_cnt = size;
    //初始值为极大值
    memset(_heap_val, 0x7f, sizeof(float) * _heap_size);
    //数组下标从1开始
    _heap_val--;
    _heap_tag--;
}

void MaxHeap::pop_top(uint32_t heap_size) {
    insert(heap_size - 1, 1, _heap_val[heap_size], _heap_tag[heap_size]);
}

void MaxHeap::reorder() {
    for (uint32_t i = 0; i < _heap_size; i++) {
        float val_bk = _heap_val[1];
        uint32_t tag_bk = _heap_tag[1];
        pop_top(_heap_size - i);
        _heap_val[_heap_size - i] = val_bk;
        _heap_tag[_heap_size - i] = tag_bk;
    }
}

void MaxHeap::max_heap_update(const float new_val, const uint32_t new_tag) {
    uint32_t father_idx = 1;

    if (new_val >= _heap_val[father_idx]) {
        return;
    }

    //当插入元素小于_heap_size时候，从可能会影响到的那个节点开始查找插入位置
    if (_default_point_cnt > 0) {
        father_idx = _default_point_cnt--;
    }

    insert(_heap_size, father_idx, new_val, new_tag);
}

void MaxHeap::insert(uint32_t heap_size, uint32_t father_idx, float new_val, uint32_t new_tag) {
    while (1) {
        uint32_t left_ch = father_idx << 1;

        if (left_ch > heap_size) {
            break;
        }

        uint32_t right_ch = left_ch + 1;

        //继续检查左分支
        if (right_ch > heap_size || _heap_val[left_ch] > _heap_val[right_ch]) {
            //更新的val大于左节点，放在左节点的父节点
            if (_heap_val[left_ch] < new_val) {
                break;
            }

            //左节点上移
            _heap_val[father_idx] = _heap_val[left_ch];
            _heap_tag[father_idx] = _heap_tag[left_ch];
            father_idx = left_ch;
        } else {
            //更新的val大于右节点，放在右节点的父节点
            if (_heap_val[right_ch] < new_val) {
                break;
            }

            _heap_val[father_idx] = _heap_val[right_ch];
            _heap_tag[father_idx] = _heap_tag[right_ch];
            father_idx = right_ch;
        }
    }

    _heap_val[father_idx] = new_val;
    _heap_tag[father_idx] = new_tag;
}
}