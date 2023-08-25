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
 * @file   tinker_distfunc_opt.cpp
 * @author huangben@baidu.com
 * @author yinjie06@baidu.com
 * @date   2022/5/20 10:43
 * @brief
 *
 **/
#include <algorithm> // std::min
#include <limits>
#include <vector>
#include "puck/tinker/method/hnsw.h"
#include "puck/tinker/method/hnsw_distfunc_opt_impl_inline.h"
#include "puck/tinker/portable_prefetch.h"
#include "puck/tinker/space.h"

namespace similarity {
template <typename dist_t>
void
Hnsw<dist_t>::SearchOld_level0(const float* pVectq, const size_t feature_dim, const int topk,
                               const std::vector<int>& enterpointIds, std::priority_queue<std::pair<float, int>>& closestDistQueuei) {
    TMP_RES_ARRAY(TmpRes);
    size_t qty = feature_dim;
    VisitedList* vl = visitedlistpool->getFreeVisitedList();
    vl_type* massVisited = vl->mass;
    vl_type currentV = vl->curV;
    std::priority_queue<EvaluatedMSWNodeInt<dist_t>>
            candidateQueuei;

    int distance_computations = 0;

    for (auto& enterpointId : enterpointIds) {
        int curNodeNum = enterpointId;
        dist_t curdist = (fstdistfunc_(
                              pVectq, (float*)(data_level0_memory_ + curNodeNum * memoryPerObject_ + offsetData_ + objectDataOffset_), qty,
                              TmpRes));
        ++distance_computations;
        massVisited[curNodeNum] = currentV;

        if (closestDistQueuei.size() < (size_t)topk || curdist < closestDistQueuei.top().first) {
            candidateQueuei.emplace(-curdist, curNodeNum);
            closestDistQueuei.emplace(curdist, curNodeNum);

            if (closestDistQueuei.size() > (size_t)topk) {
                closestDistQueuei.pop();
            }
        }
    }

    while (!candidateQueuei.empty()) {
        EvaluatedMSWNodeInt<dist_t> currEv = candidateQueuei.top(); // This one was already compared to the query

        if (closestDistQueuei.size() >= (size_t)topk && (-currEv.getDistance()) > closestDistQueuei.top().first) {
            break;
        }

        candidateQueuei.pop();
        int curNodeNum = currEv.element;
        int* data = (int*)(data_level0_memory_ + curNodeNum * memoryPerObject_ + offsetLevel0_);
        int size = *data;
        PREFETCH((char*)(massVisited + * (data + 1)), _MM_HINT_T0);
        PREFETCH((char*)(massVisited + * (data + 1) + 64), _MM_HINT_T0);
        PREFETCH(data_level0_memory_ + (*(data + 1)) * memoryPerObject_ + offsetData_, _MM_HINT_T0);
        PREFETCH((char*)(data + 2), _MM_HINT_T0);

        for (int j = 1; j <= size; j++) {
            int tnum = *(data + j);
            PREFETCH((char*)(massVisited + * (data + j + 1)), _MM_HINT_T0);
            PREFETCH(data_level0_memory_ + (*(data + j + 1)) * memoryPerObject_ + offsetData_, _MM_HINT_T0);

            if (!(massVisited[tnum] == currentV)) {
                ++distance_computations;
                massVisited[tnum] = currentV;
                char* currObj1 = (data_level0_memory_ + tnum * memoryPerObject_ + offsetData_);
                dist_t d = (fstdistfunc_(pVectq, (float*)(currObj1 + objectDataOffset_), qty, TmpRes));

                if (closestDistQueuei.top().first > d || closestDistQueuei.size() < (size_t)topk) {
                    candidateQueuei.emplace(-d, tnum);
                    closestDistQueuei.emplace(d, tnum);

                    if (closestDistQueuei.size() > (size_t)topk) {
                        closestDistQueuei.pop();
                    }
                }
            }
        }
    }

    visitedlistpool->releaseVisitedList(vl);
}

template class Hnsw<float>;
}
