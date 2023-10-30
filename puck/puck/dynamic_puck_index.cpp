/**
 * @file    dynamic_puck_index.cpp
 * @author  yinjie06(yinjie06@baidu.com)
 * @date    2023/10/27 10:22
 * @brief
 *
 **/

#include <queue>
#include <memory>
#include "puck/puck/dynamic_puck_index.h"
#include "puck/gflags/puck_gflags.h"
#include "puck/search_context.h"
#include "puck/hierarchical_cluster/max_heap.h"
namespace puck {

DEFINE_int32(max_point_stored, 10000000, "max_point_stored");
DynamicPuckIndex::DynamicPuckIndex() {
    _conf.index_type = puck::IndexType::PUCK;
    _is_init = false;
    _conf.show();
    save_model_file();
}

int DynamicPuckIndex::check_index_type() {
    if (_conf.index_type != IndexType::PUCK) {
        LOG(ERROR) << "index_type is not PUCK";
        return -1;
    }

    return 0;
}

int DynamicPuckIndex::search(const Request* request, Response* response) {
    if (request->topk > _conf.topk || request->feature == nullptr) {
        LOG(ERROR) << "topk should <= topk, topk = " << _conf.topk << ", or feature is nullptr";
        return -1;
    }

    DataHandler<SearchContext> context(_context_pool);

    if (0 != context->reset(_conf)) {
        return -1;
    }

    const float* feature = normalization(context.get(), request->feature);

    if (feature == nullptr) {
        return -1;
    }

    //输出query与一级聚类中心的top-search-cell个ID和距离
    int ret = search_nearest_coarse_cluster(context.get(), feature, _conf.search_coarse_count);

    if (ret != 0) {
        LOG(ERROR) << "search nearest coarse cluster error " << ret;
        return ret;
    }

    //计算query与二级聚类中心的距离，并根据filter特征，筛选子集
    int search_point_cnt = search_nearest_filter_points(context.get(), feature);

    if (search_point_cnt < 0) {
        LOG(ERROR) << "search filter points has error.";
        return -1;
    }

    MaxHeap result_heap(request->topk, response->distance, response->local_idx);
    ret = rank_topN_points(context.get(), feature, search_point_cnt, result_heap);

    if (ret == 0) {
        response->result_num = result_heap.get_heap_size();
    } else {
        LOG(ERROR) << "rank points after filter has error.";
    }

    // LOG(INFO)<<response->result_num;
    return 0;
}

int DynamicPuckIndex::init_model_memory() {
    LOG(INFO) << "DynamicPuckIndex::init_model_memory, " << _conf.total_point_count;
    this->PuckIndex::init_model_memory();

    _dynamic_filter_quantization.reset(
            new Quantization(_filter_quantization->get_quantization_params(), _max_point_count));
    _dynamic_filter_quantization->init_quantized_feature_memory();

    _cell_assign.reset(new int[_max_point_count]);
    memset(_cell_assign.get(), -1, _max_point_count * sizeof(int));
    _label_to_memory.reset(new int[_max_point_count]);
    memset(_cell_assign.get(), -1, _max_point_count * sizeof(int));

    _memory_to_label = (int*)_memory_to_local;
    memset(_memory_to_label, -1, _conf.total_point_count * sizeof(int));

    void* memb = nullptr;
    int32_t pagesize = getpagesize();
    size_t all_feature_length = (size_t)_conf.total_point_count * _conf.feature_dim * sizeof(float);
    size_t size = all_feature_length + (pagesize - all_feature_length % pagesize);
    int err = posix_memalign(&memb, pagesize, size);

    if (err != 0) {
        std::runtime_error("alloc_aligned_mem_failed errno=" + errno);
        return -1;
    }

    _all_feature = reinterpret_cast<float*>(memb);
    return 0;
}

int DynamicPuckIndex::init() {
    LOG(INFO) << "start init index.";
    _conf.adaptive_train_param();

    if (check_index_type() != 0) {
        LOG(ERROR) << "check_index_type has error.";
        return -1;
    }

    _conf.total_point_count = FLAGS_max_point_stored;
    _max_point_count = FLAGS_max_point_stored * 3;
    _cell_has_points.resize(_conf.coarse_cluster_count * _conf.fine_cluster_count);
    _conf.show();

    //初始化内存
    if (init_model_memory() != 0) {
        LOG(ERROR) << "init_model_memory has error.";
        return -1;
    }

    init_context_pool();
    _is_init = false;
    //调整默认的检索参数 & 检索参数检查
    return _conf.adaptive_search_param();
}

int DynamicPuckIndex::batch_add(
        const uint32_t n,
        const uint32_t dim,
        const float* features,
        const uint32_t* labels) {
    write_fvec_format(_conf.feature_file_name.c_str(), dim, n, features);

    if (_is_init == false) {
        _conf.feature_dim = dim;
        PuckIndex index;
        index.train();

        //读码本
        if (read_coodbooks() != 0) {
            LOG(ERROR) << "read_coodbooks has error.";
            return -1;
        }
        {
            _stationary_cell_dist.reset(
                    new float[_conf.fine_cluster_count * _conf.coarse_cluster_count]);
            std::vector<float> fine_norms(_conf.fine_cluster_count);

            for (uint32_t i = 0; i < _conf.fine_cluster_count; ++i) {
                fine_norms[i] = cblas_sdot(
                                        _conf.feature_dim,
                                        _fine_vocab + _conf.feature_dim * i,
                                        1,
                                        _fine_vocab + _conf.feature_dim * i,
                                        1) /
                        2;
            }

            std::vector<float> coarse_fine_products(
                    _conf.fine_cluster_count * _conf.coarse_cluster_count);
            //矩阵乘
            matrix_multiplication(
                    _fine_vocab,
                    _coarse_vocab,
                    _conf.fine_cluster_count,
                    _conf.coarse_cluster_count,
                    _conf.feature_dim,
                    "TN",
                    coarse_fine_products.data());
            for (uint32_t i = 0; i < _conf.fine_cluster_count * _conf.coarse_cluster_count; ++i) {
                size_t fine_id = i % _conf.fine_cluster_count;
                _stationary_cell_dist[i] = fine_norms[fine_id] + coarse_fine_products[i];
            }
        }
        _is_init = true;
    }

    _conf.show();
    std::unique_ptr<uint32_t[]> cell_assign(new uint32_t[n]);
    this->batch_assign(n, _conf.feature_file_name, cell_assign.get());
    int memory_ids[n];
    size_t idx = 0;
    for (size_t i = 0; i < _conf.total_point_count && idx < n; ++i) {
        if (_memory_to_label[i] < 0) {
            memory_ids[idx++] = i;
        }
    }

    for (size_t i = 0; i < n; ++i) {
        size_t memory_id = memory_ids[i];
        size_t cur_local_id = labels[i];

        _label_to_memory[cur_local_id] = memory_id;
        _memory_to_label[memory_id] = cur_local_id;

        _cell_assign.get()[labels[i]] = (int)cell_assign[i];
        auto* cur_filter_feature =
                _dynamic_filter_quantization->get_quantized_feature(cur_local_id);
        auto* temp_filter_feature = _filter_quantization->get_quantized_feature(i);
        memcpy(cur_filter_feature,
               temp_filter_feature,
               _dynamic_filter_quantization->get_per_fea_len());
        memcpy(_all_feature + memory_id * dim, features + i * dim, sizeof(float) * dim);
        _cell_has_points[cell_assign[i]].push_back(cur_local_id);
    }
    update_stationary_cell_dist();

    return 0;
}

int DynamicPuckIndex::compute_quantized_distance(
        SearchContext* context,
        const FineCluster* cur_fine_cluster,
        const float cell_dist,
        MaxHeap& result_heap) {
    float* result_distance = result_heap.get_top_addr();
    const float* pq_dist_table = context->get_search_point_data().pq_dist_table;

    // point info存储的量化特征对应的参数
    auto& quantization_params = _dynamic_filter_quantization->get_quantization_params();
    uint32_t* query_sorted_tag = context->get_search_point_data().query_sorted_tag;
    size_t cur_cell_id = cur_fine_cluster - get_fine_cluster(0);
    auto point_cnt = _cell_has_points[cur_cell_id].size();
    uint32_t updated_cnt = 0;

    for (uint32_t i = 0; i < point_cnt; ++i) {
        size_t local_id = _cell_has_points[cur_cell_id][i];

        const unsigned char* feature =
                _dynamic_filter_quantization->get_quantized_feature(local_id);
        float temp_dist = 2.0 * cell_dist + ((float*)feature)[0];

        if (temp_dist >= result_distance[0]) {
            continue;
            ;
            // break;
        }

        const unsigned char* pq_feature =
                (unsigned char*)feature + _dynamic_filter_quantization->get_fea_offset();
#ifdef __SSE__
        temp_dist += lookup_dist_table(
                pq_feature, pq_dist_table, quantization_params.ks, quantization_params.nsq);
#else

        for (uint32_t m = 0; m < (uint32_t)quantization_params.nsq; ++m) {
            uint32_t idx = query_sorted_tag[m];
            temp_dist += (pq_dist_table + idx * quantization_params.ks)[pq_feature[idx]];

            //当PQ子空间累计距离已经大于当前最大值，不再计算
            if (temp_dist > result_distance[0]) {
                break;
            }
        }

#endif

        if (temp_dist < result_distance[0]) {
            result_heap.max_heap_update(temp_dist, local_id);
            ++updated_cnt;
        }
    }

    return updated_cnt;
}

int DynamicPuckIndex::rank_topN_points(
        SearchContext* context,
        const float* feature,
        const uint32_t filter_topk,
        MaxHeap& result_heap) {
    auto& search_point_data = context->get_search_point_data();
    float* result_distance = search_point_data.result_distance;
    uint32_t* result_tag = search_point_data.result_tag;
    // float query_norm = cblas_sdot(_conf.feature_dim, feature, 1, feature, 1);
    //堆顶
    float* true_result_distance = result_heap.get_top_addr();

    {
        size_t qty_16 = _conf.feature_dim;
        _mm_prefetch((char*)(feature), _MM_HINT_T0);
        float PORTABLE_ALIGN32 TmpRes[8];

        for (uint32_t idx = 0; idx < filter_topk; ++idx) {
            size_t memory_id = _label_to_memory[result_tag[idx]];
            const float* exhaustive_feature =
                    _all_feature + (uint64_t)memory_id * _conf.feature_dim;
            float temp_dist = similarity::L2SqrExt(exhaustive_feature, feature, qty_16, TmpRes);

            if (temp_dist < true_result_distance[0]) {
                result_heap.max_heap_update(temp_dist, result_tag[idx]);
            }
        }
    }

    result_heap.reorder();
    // LOG(INFO)<<result_heap.get_heap_size();
    return 0;
}
void DynamicPuckIndex::update_cell_assign() {
#pragma omp parallel for schedule(dynamic) num_threads(_conf.threads_count)
    for (uint32_t i = 0; i < _cell_has_points.size(); ++i) {
        size_t start = 0;

        for (size_t j = 0; j < _cell_has_points[i].size(); ++j) {
            if (_cell_assign[j] >= 0) {
                _cell_has_points[i][start] = _cell_has_points[i][j];
                ++start;
            }
        }

        _cell_has_points[i].resize(start);
        _cell_has_points[i].shrink_to_fit();
    }

    update_stationary_cell_dist();
}

void DynamicPuckIndex::update_stationary_cell_dist() {
#pragma omp parallel for schedule(dynamic) num_threads(_conf.threads_count)
    for (uint32_t i = 0; i < _conf.coarse_cluster_count; ++i) {
        _coarse_clusters[i].min_dist_offset = std::numeric_limits<float>::max();
    }

    for (uint32_t i = 0; i < _conf.fine_cluster_count * _conf.coarse_cluster_count; ++i) {
        FineCluster* cur_fine_cluster = get_fine_cluster(i);
        size_t fine_id = i % _conf.fine_cluster_count;

        if (_cell_has_points[i].size() == 0) {
            cur_fine_cluster->stationary_cell_dist = std::sqrt(std::numeric_limits<float>::max());
        } else {
            cur_fine_cluster->stationary_cell_dist = _stationary_cell_dist[i];
        }

        size_t coarse_id = i / _conf.fine_cluster_count;
        _coarse_clusters[coarse_id].min_dist_offset = std::min(
                _coarse_clusters[coarse_id].min_dist_offset,
                cur_fine_cluster->stationary_cell_dist);
    }
}
}  // namespace puck
