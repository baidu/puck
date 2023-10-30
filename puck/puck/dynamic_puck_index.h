/**
 * @file    dynamic_puck_index.h
 * @author  yinjie06(yinjie06@baidu.com)
 * @date    2023/10/27 10:22
 * @brief
 *
 **/
#pragma once
#include <vector>
#include "puck/puck/puck_index.h"
#include "puck/tinker/method/hnsw.h"
#include "puck/tinker/space/space_lp.h"
namespace puck {
//内存索引结构
class DynamicPuckIndex : public puck::PuckIndex {
public:
    /*
     * @brief 默认构造函数，检索配置根据gflag参数确定(推荐使用)
     **/
    DynamicPuckIndex();
    virtual ~DynamicPuckIndex() {}
    /*
     * @brief 检索最近的topk个样本
     * @@param [in] request : request
     * @@param [out] response : response
     * @@return (int) : 正常返回0，错误返回值<0
     **/
    virtual int search(const Request* request, Response* response) override;
    /*
    * @brief 读取索引配置文件（index.dat）、初始化内存、加载索引文件，检索前需要先调用该函数
    * @@return (int) : 正常返回0，错误返回值<0
    **/
    virtual int init() override;

    int batch_add(const uint32_t n, const uint32_t dim, const float* features, const uint32_t* labels);
    int batch_delete(const uint32_t n, const uint32_t* labels) {
        #pragma omp parallel for schedule(dynamic) num_threads(_conf.threads_count)
        for (size_t i = 0; i < n; ++i) {
            _cell_assign[labels[i]] = -1;
            size_t memory_id = _label_to_memory[labels[i]];
            _memory_to_label[memory_id] = -1;
        }

        update_cell_assign();
        return 0;
    }
protected:
    virtual int check_index_type() override;

    int init_model_memory();
    /*
    * @brief 计算query与某个cell下所有样本的距离（样本的filter量化特征）
    * @@param [in\out] context : context由内存池管理
    * @@param [in] FineCluster : cell指针
    * @@param [in] cell_dist : query和cell的距离
    * @@param [in] result_heap : 堆结构，存储query与样本的topk
    * @@return (int) : 正常返回0，错误返回值<0
    **/
    virtual int compute_quantized_distance(SearchContext* context, const FineCluster*, const float cell_dist,
                                           MaxHeap& result_heap);
    /*
     * @brief 计算query与部分样本的距离(query与filter特征的topN个样本）
     * @@param [in\out] context : context由内存池管理
     * @@param [in] feature : query的特征向量
     * @@param [in] filter_topk : cell指针
     * @@param [in] result_heap : 堆结构，存储query与样本的topk
     * @@return (int) : 正常返回0，错误返回值<0
     **/
    virtual int rank_topN_points(SearchContext* context, const float* feature, const uint32_t filter_topk,
                                 MaxHeap& result_heap);
    void update_cell_assign();
    void update_stationary_cell_dist();
private:
    DISALLOW_COPY_AND_ASSIGN_AND_MOVE(DynamicPuckIndex);
    std::unique_ptr<int[]> _cell_assign;
    std::unique_ptr<int[]> _label_to_memory;
    std::unique_ptr<float[]> _stationary_cell_dist;
    int* _memory_to_label;

    std::unique_ptr<Quantization> _dynamic_filter_quantization;
    std::vector<std::vector<int>> _cell_has_points;
    
    bool _is_init;
    size_t _max_point_count;
};
}
