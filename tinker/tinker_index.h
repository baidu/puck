//Copyright (c) 2024 Baidu, Inc.  All Rights Reserved.
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
 * @file tinker_index.h
 * @author hananoyuuki@qq.com
 * @date 2024/9/23 20:05
 * @brief tinker_index.h version: 2.0 after refactoring
 *
 **/
#pragma once
#include <atomic>
#include <vector>
#include <shared_mutex>
#include <queue>
#include <chrono>
#include <unordered_map>
#include <functional>
#include <string>
#include <cstdint>
#include <mutex>
#include <limits>
#include "gflags/puck_gflags.h"
#include "interface/index.h"
#include "tinker/internal/visited_list.h"
#include "quantizer/quantizer.h"

namespace puck {

#ifndef DISALLOW_COPY_AND_ASSIGN_AND_MOVE
    #define DISALLOW_COPY_AND_ASSIGN_AND_MOVE(TypeName) \
        TypeName(const TypeName&) = delete;             \
        void operator=(const TypeName&) = delete;       \
        TypeName(const TypeName&&) = delete
#endif
#ifndef DISALLOW_COPY_AND_ASSIGN
    #define DISALLOW_COPY_AND_ASSIGN(TypeName) \
        TypeName(const TypeName&) = delete;      \
        void operator=(const TypeName&) = delete
#endif

using LabelId = uint32_t;

//内存索引结构
class TinkerIndex : public Index {
public:
    /*
     * @brief 默认构造函数，索引配置根据gflag参数确定；检索宽度由Request显式提供
     **/
    TinkerIndex();
    ~TinkerIndex() override {}
    /*
    * @brief 读取配置文件、初始化内存
    * @@return (int) : 正常返回0，错误返回值<0
    **/
    int init() override;
    /*
    * @brief 训练量化器
    * @@return (int) : 正常返回0，错误返回值<0
    **/
    int train() override;
    /*
     * @brief 读取索引配置文件（index.dat）、初始化内存、建库（计算样本最近的1个聚类中心）、写索引文件
     * @@return (int) : 正常返回0，错误返回值<0
     **/
    int build() override;
    /*
     * @brief 向索引插入一条数据，线程安全
     * @@return (int) : 正常返回值0，错误返回值<0
     **/
    int insert(const InsertRequest* request) override;
    /*
     * @brief 检索最近的topk个样本
     * @@param [in] request : ef_search默认为0（宽度topk），正数宽度max(ef_search, topk)，负数拒绝
     * @@param [out] response : response
     * @@return (int) : 正常返回0，错误返回值<0
     **/
    int search(const Request* request, Response* response) override;
    /*
     * @brief 写索引文件(建库的产出，与建库样本相关)，提供一种alternative实现，在提供quant索引的情况下，可以转存为量化后的索引文件
     * @@return (int) : 正常返回0，错误返回值<0
     **/
    int save_index();
    int save_index(const std::string &auxiliary_quant_file);
    /*
     * @brief 初始化内存，读索引文件到内存，(建库的产出，与建库样本相关)
     * @@return (int) : 正常返回0，错误返回值<0
     **/
    int read_index();

    /*
     * @brief 查询索引大小（注意有些节点可能仍在计算插入，所以这个值不是已建库的个数）
     * @@return (size_t) : 返回索引内已插入的节点个数
     **/
    size_t size();
    uint32_t dim() const { return static_cast<uint32_t>(_dim); }

    /*
     * @brief 创建空索引，用于从零开始的增量构建场景
     *        要求 gflags 中 tinker_quantizer_type 必须为 "FLAT"，否则报错返回
     *        仅在内存中初始化索引结构，不保存到磁盘（由用户决定是否调用 save_index）
     * @@return (int) : 正常返回0，非FLAT量化器或其他错误返回值<0
     **/
    int create_empty_index();

    /*
     * @brief 保存量化器索引到默认路径
     * @@return (int) : 正常返回0，错误返回值<0
     **/
    int save_quantizer();
    /*
     * @brief 从默认路径加载量化器索引
     * @@return (int) : 正常返回0，错误返回值<0
     **/
    int load_quantizer();
    /*
     * @brief 重置索引，清空内存中的数据
     * @@return (int) : 正常返回0，错误返回值<0
     **/
    int reset();

    bool is_brief_enabled() const;
    bool is_cartesian_enabled() const;
protected:
    /*
     * @brief 读取索引配置文件（index.dat）
     * @@return (int) : 正常返回0，错误返回值<0
     **/
    int read_model_file();
    /*
     * @brief 写索引配置文件（index.dat）
     * @@return (int) : 正常返回0，错误返回值<0
     **/
    int save_model_file();
    /*
     * @brief 校验样本特征数据文件 FLAGS_index_path/FLAGS_feature_file_name (fvecs格式)的维度合法性，
     *        并统计样本条数，分块读取校验
     * @@return (int) : 正常返回读取到的样本条数，错误返回值<0
     **/
    int check_feature_data();
    

    /*
     * @brief 检索距离query最近的topk个样本，以堆结果返回
     * @@return (int) : 正常返回值0，错误返回值<0
     **/
    int knn_search(const float* query, size_t topk, std::priority_queue<std::pair<float, unsigned int>>& result_heap);
    int knn_search(const float* query, const std::vector<unsigned int> enterpoint, size_t topk, std::priority_queue<std::pair<float, unsigned int>>& result_heap);
    int filtered_knn_search(const float* query,
                            const std::vector<LabelId>& query_labels,
                            size_t topk,
                            bool require_all_labels,
                            std::priority_queue<std::pair<float, unsigned int>>& result_heap);

    /*
     * @brief 根据vamana rng规则裁剪id号节点的邻居
     * @return (int) : 正常返回值0，错误返回值<0
     * @cite https://www.cs.utexas.edu/~rak/papers/neurips19-diskann.pdf
     **/
    int purge_link(const unsigned int id, size_t NN, std::priority_queue<std::pair<float, unsigned int>> &candidate_neighbors);
    int filtered_purge_link(const unsigned int id,
                            size_t max_degree,
                            std::priority_queue<std::pair<float, unsigned int>>& candidate_neighbors);

    /*
     * @brief 添加连边from->to，如果超出from的出边上限则进行裁边
     * @@return (int) : 正常返回值0，错误返回值<0
     **/
    int add_neighbor(const unsigned int from, const unsigned int to);

    /*
     * @brief 根据图拓扑结构重排节点，提高局部性
     * @@return (int) : 正常返回值0，错误返回值<0
     **/
    int reorder_index();
    
    int check_index_type();

private:
    // @brief 获取第id号节点的向量数据头指针
    // @return (char*) 返回向量数据头指针
    inline char* get_node_fea(const unsigned int id);

    // @brief 获取第id号节点的邻接表头指针 - linksize + linklist
    // @return (char*) 返回邻接表头指针
    inline char* get_node_linklist(const unsigned int id);
    
    // @brief 获取第id号节点的读写锁
    // @return (shared_mutex&) : 返回读写锁的引用
    inline std::shared_mutex& get_node_mutex(const unsigned int id);
    inline std::mutex& get_filter_label_mutex(LabelId label_id);

    // @brief 计算当前加载量化器意义下的数据宽度和pad
    // @return 正常返回0，错误返回值<0
    int adapt_data_width();
    int load_filter_metadata();
    int save_filter_metadata() const;
    int get_or_create_label_id(const std::vector<std::string>& labels,
                               std::vector<LabelId>* label_ids);
    int get_label_ids(const std::vector<std::string>& labels,
                      std::vector<LabelId>* label_ids) const;
    std::string canonicalize_labels(const std::vector<std::string>& labels) const;
    int update_filter_entrypoints(const std::vector<LabelId>& label_ids,
                                  unsigned int node_id);
    int insert(const float* feature, size_t local_id);
    int insert_filtered(const InsertRequest* request);
    int insert_preprocessed(const float* feature, size_t local_id);
    int insert_filtered_preprocessed(const InsertRequest* request);
    int search_unfiltered(const Request* request, Response* response);
    int search_filtered(const Request* request, Response* response);
    size_t resolve_search_width(const Request* request) const;
    bool node_matches_any_label(
        unsigned int node_id,
        const std::vector<LabelId>& label_ids) const;
    bool node_matches_all_labels(
        unsigned int node_id,
        const std::vector<LabelId>& label_ids) const;
    std::vector<unsigned int> get_filter_entrypoints(
        const std::vector<LabelId>& label_ids) const;


private:
    DISALLOW_COPY_AND_ASSIGN_AND_MOVE(TinkerIndex);
    // parameters
    // 构建时必须设置的参数，gflags指定
    // TODO: 加个裁边松弛参数_alpha
    puck::IndexType _index_type;
    uint8_t _brief;
    uint8_t _cartesian;
    bool _whether_norm;
    int _dim;
    size_t _M;
    size_t _ef_construction;
    size_t _thread_num;
    // ---

    // 获取节点读写锁的时候只考虑节点id的低位，同hash值的节点会共享一个锁，增大这个值可以有效提高插入并发
    static constexpr unsigned int NODE_HASH_LOWBIT = 0xfffff;
    static constexpr unsigned int NODE_MUTEX_COUNT = NODE_HASH_LOWBIT + 1;
    // 获取属性锁时只考虑属性id的低位，同hash值的属性会共享一个锁
    static constexpr unsigned int FILTER_LABEL_HASH_LOWBIT = 0xfff;
    static constexpr unsigned int FILTER_LABEL_MUTEX_COUNT = FILTER_LABEL_HASH_LOWBIT + 1;
    static constexpr LabelId INVALID_LABEL_ID =
        std::numeric_limits<LabelId>::max();

    unsigned int _enterpoint_id;
    // 已经计算完毕的节点数
    std::atomic<unsigned int> _total_elements_stored;
    // 已经插入的节点数（有可能还没计算完毕）
    std::atomic<unsigned int> _total_elements_inserted;
    // Serializes reservation through publication so failed inserts can safely roll back the tail slot.
    std::mutex _insert_mutex;
    // 内存中可容纳的节点数
    size_t _ram_capacity;
    unsigned int _elements_capacity;

    float _alpha;
    // _data_length = _linklist_length + _fea_length
    size_t _linklist_length, _fea_length, _data_length;

    // 近邻图相关内存
    std::unique_ptr<char[]> _fea_ptr, _linklist_ptr;

    // 节点级别的哈希读写锁
    std::unique_ptr<std::shared_mutex[]> _node_mutex;
    std::unique_ptr<std::mutex[]> _filter_label_mutex;

    // 内存节点序 -> 数据集节点序
    std::vector<int> _memory_to_local_idx;

    // brief 过滤索引元数据
    std::unordered_map<std::string, LabelId> _label_to_id;
    std::vector<std::string> _id_to_label;
    std::unique_ptr<std::atomic<LabelId>[]> _node_label_ids;
    std::unique_ptr<std::shared_ptr<const std::vector<LabelId>>[]>
        _node_label_id_vectors;
    std::unordered_map<LabelId, std::vector<unsigned int>> _label_to_nodes;
    std::unordered_map<LabelId, std::vector<unsigned int>> _filter_entrypoints;
    mutable std::shared_mutex _label_dictionary_mutex;
    mutable std::shared_mutex _filter_entrypoint_mutex;

    // 量化器相关内存   
    std::unique_ptr<Quantizer> _quantizer;
    std::function<float(const char*, const float*, QuantizerParams&)> _quant_adc_distance_func;
    std::function<float(const char*, const char*, QuantizerParams&)> _quant_sdc_distance_func;
    
    VisitedPool<VisitedList> _visited_pool;

public:
    std::chrono::high_resolution_clock::duration timer;
    size_t distance_computations;
};

}//tinker
