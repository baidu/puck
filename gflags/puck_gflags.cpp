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
 * @file puck_gflags.cpp
 * @author huangben@baidu.com
 * @date 2018/7/07 15:59:36
 * @brief
 *
 **/
#include <algorithm>
#include <thread>
#include "gflags/puck_gflags.h"

namespace puck {

DEFINE_bool(need_log_file, false, "need log file");
DEFINE_string(puck_log_file, "log/puck.log", "log file name");

/*****训练&建库参数******/
//通用参数
DEFINE_string(index_path, "puck_index", "lib of index files");
DEFINE_string(index_file_name, "index.dat", "store index format information");
DEFINE_string(feature_file_name, "all_data.feat.bin", "feature data of all points");
DEFINE_string(coarse_codebook_file_name, "coarse.dat", "coarse codebook");
DEFINE_string(fine_codebook_file_name, "fine.dat", "fine codebook");

DEFINE_string(cell_assign_file_name, "cell_assign.dat", "cell assign");
//realtime insert和分布式建库的索引需要这个索引文件，search返回的local id是这个样本在该文件的行数
DEFINE_string(label_file_name, "all_data.url", "label of points");

DEFINE_int32(feature_dim, 256, "feature dim");
DEFINE_bool(whether_norm, true, "whether norm");

DEFINE_int32(coarse_cluster_count, 2000, "the number of coarse clusters");
DEFINE_int32(fine_cluster_count, 2000, "the number of fine clusters");
DEFINE_int32(threads_count, std::max(1u, std::thread::hardware_concurrency()), "threads count");

DEFINE_int32(ip2cos, 0, "Convert ip to cos, 0-NA, 1-directly，2-need transform");

//puck
DEFINE_bool(whether_pq, true, "whether pq");
DEFINE_int32(nsq, FLAGS_feature_dim, "the count of pq sub space, default valus is 1/4");
DEFINE_string(pq_codebook_file_name, "learn_codebooks.dat", "pq codebook");
DEFINE_string(pq_data_file_name, "pq_data.dat", "pq data of all points");

DEFINE_int32(filter_nsq, FLAGS_feature_dim / 4, "the count of pq sub space");
DEFINE_string(filter_codebook_file_name, "filter_codebook.dat", "filter codebook");
DEFINE_string(filter_data_file_name, "filter_data.dat", "filter data of points");

/***********检索参数*********/
//检索相关
//检索时，初始化内存池的size
DEFINE_int32(context_initial_pool_size, std::thread::hardware_concurrency(), "search context pool size");
//检索通用参数
DEFINE_int32(search_coarse_count, 200, "restrict the retrieval range in top-n nearest coarse clusters");
DEFINE_int32(topk, 100, "return top-k nearest points");

//HierarchicalClusterIndex
DEFINE_int32(neighbors_count, 40000, "search points count, default value is 4w");

//puck
DEFINE_int32(filter_topk, FLAGS_topk * 11, "filter top-k");
DEFINE_double(radius_rate, 1.0, "radius_rate");

//tinker

DEFINE_string(tinker_file_prefix, "tinker", "prefix of tinker index file name");
DEFINE_string(tinker_idmapper_file_name, "idmapper.dat", "tinker idmapper file name");
DEFINE_string(tinker_quantizer_file_name, "quantizer.dat", "quantizer file name");
DEFINE_string(tinker_quantizer_type, "FLAT", "quantizer type");
DEFINE_int32(tinker_neighborhood_count, 16, "neighborhood conut");
DEFINE_int32(tinker_ef_construction, 600, "tinker_ef_construction");
DEFINE_int64(tinker_memory_capacity, 42949672960ul, "tinker_memory_capacity (in B)");
DEFINE_bool(tinker_brief, false, "enable filtered incremental graph index");
DEFINE_string(brief_file_name, "all_data.brief", "tab-separated filter labels per vector for Tinker brief build");
DEFINE_bool(tinker_cartesian, true, "use cartesian labels for brief index");

}

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100 */
