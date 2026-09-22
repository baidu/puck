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
 * @file puck_gflags.h
 * @author huangben(com@baidu.com)
 * @date 2018/7/07 15:59:36
 * @brief
 *
 **/

#pragma once

#include <gflags/gflags.h>

namespace puck {

DECLARE_bool(need_log_file);
DECLARE_string(puck_log_file);

/*****训练&建库参数******/
//通用参数
DECLARE_string(index_path);
DECLARE_string(index_file_name);
DECLARE_string(feature_file_name);
DECLARE_string(coarse_codebook_file_name);
DECLARE_string(fine_codebook_file_name);

DECLARE_string(cell_assign_file_name);
DECLARE_string(label_file_name);

DECLARE_int32(feature_dim);
DECLARE_bool(whether_norm);

DECLARE_int32(coarse_cluster_count);
DECLARE_int32(fine_cluster_count);
DECLARE_int32(threads_count);

DECLARE_int32(ip2cos);

//puck
DECLARE_bool(whether_pq);
DECLARE_int32(nsq);
DECLARE_string(pq_codebook_file_name);
DECLARE_string(pq_data_file_name);

DECLARE_int32(filter_nsq);
DECLARE_string(filter_codebook_file_name);
DECLARE_string(filter_data_file_name);

/***********检索参数*********/
//检索时，初始化内存池的size
DECLARE_int32(context_initial_pool_size);
//检索通用参数
DECLARE_int32(search_coarse_count);
DECLARE_int32(topk);

//HierarchicalClusterIndex
DECLARE_int32(neighbors_count);

//puck
DECLARE_int32(filter_topk);
DECLARE_double(radius_rate);
//tinker

/// <tinker_file_prefix>.relat
/// <tinker_file_prefix>.feat
DECLARE_string(tinker_file_prefix);   // tinker索引文件前缀
DECLARE_string(tinker_idmapper_file_name);  // tinker memory id -> local id的映射文件
DECLARE_string(tinker_quantizer_file_name);        // 量化器文件名
DECLARE_string(tinker_quantizer_type);             // 量化器类型
DECLARE_int32(tinker_neighborhood_count);         // 节点邻居数量
DECLARE_int32(tinker_ef_construction);         // ef construction
DECLARE_int64(tinker_memory_capacity);      // 最大内存容量，会影响单图分片中的最大可容纳节点数量，根据经验设置为预期实际使用的2/3
DECLARE_bool(tinker_brief);                  // 是否启用过滤增量索引
DECLARE_string(brief_file_name);            // Tinker brief建库标签文件，每行以tab分隔过滤标签
DECLARE_bool(tinker_cartesian);              // brief索引是否使用笛卡尔标签

}

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100 */
