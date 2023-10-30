/**
 * @file    multi_brief_puck_index.cpp
 * @author  yinjie06(yinjie06@baidu.com)
 * @date    2023/09/12 16:49
 * @brief
 *
 **/
#include <glog/logging.h>

#include "puck/puck/multi_brief_puck_index.h"
#include "puck/hierarchical_cluster/max_heap.h"
#include "puck/search_context.h"
namespace puck {

MultiBriefPuckIndex::MultiBriefPuckIndex() {
    _conf.index_type = IndexType::MULTI_BRIEF_PUCK_INDEX;
}

MultiBriefPuckIndex::~MultiBriefPuckIndex() {}

int MultiBriefPuckIndex::check_index_type() {
    if (_conf.index_type != IndexType::MULTI_BRIEF_PUCK_INDEX) {
        LOG(ERROR) << "index_type is not MULTI_BRIEF_PUCK_INDEX";
        return -1;
    }

    return 0;
}

int MultiBriefPuckIndex::convert_local_to_memory_idx(
        uint32_t* cell_start_memory_idx,
        uint32_t* local_to_memory_idx) {
    LOG(INFO) << "convert_local_to_memory_idx";

    if (this->PuckIndex::convert_local_to_memory_idx(cell_start_memory_idx, local_to_memory_idx) !=
        0) {
        return -1;
    }

    std::unique_ptr<uint32_t[]> cell_assign(new uint32_t[_conf.total_point_count]);

    if (load_cell_assign(cell_assign.get()) != 0) {
        LOG(INFO) << "load_cell_assign has error.";
        return -1;
    }

    std::string brief_indptr_file_name = _conf.index_path + "/indptr.dat";
    std::ifstream brief_indptr_file(
            brief_indptr_file_name.c_str(), std::ios::binary | std::ios::in);
    int group_cnt = 0;
    brief_indptr_file.read((char*)&group_cnt, sizeof(int));
    _briefs_indptr.reset(new int32_t[group_cnt]);
    brief_indptr_file.read((char*)(_briefs_indptr.get()), sizeof(_briefs_indptr[0]) * group_cnt);
    brief_indptr_file.close();
    LOG(INFO) << "group cnt = " << group_cnt;

    if (group_cnt != _conf.total_point_count + 1) {
        LOG(ERROR) << "group cnt error";
        return -1;
    }

    std::string brief_indices_file_name = _conf.index_path + "/indices.dat";
    std::ifstream brief_indices_file(
            brief_indices_file_name.c_str(), std::ios::binary | std::ios::in);
    group_cnt = 0;
    brief_indices_file.read((char*)&group_cnt, sizeof(int));
    _briefs_indices.reset(new int32_t[group_cnt]);
    brief_indices_file.read((char*)(_briefs_indices.get()), sizeof(_briefs_indices[0]) * group_cnt);
    brief_indices_file.close();
    LOG(INFO) << "brief point relation pair = " << group_cnt;

    std::vector<std::pair<int, int>> brief_point;
    std::vector<std::pair<int, int>> point_brief;

    for (size_t j = 0; j < _conf.total_point_count; ++j) {
        int local_id = j;

        for (size_t i = _briefs_indptr[j]; i < _briefs_indptr[j + 1]; ++i) {
            int brief_id = _briefs_indices[i];

            brief_point.push_back(std::make_pair(brief_id, local_to_memory_idx[local_id]));
            point_brief.push_back(std::make_pair(local_to_memory_idx[local_id], brief_id));
        }
    }

    std::stable_sort(brief_point.begin(), brief_point.end());

    int brief_count = brief_point.back().first + 1;
    LOG(INFO) << "brief count = " << brief_count;
    _briefs_cell_indptr.reset(new int32_t[brief_count + 1]);
    std::vector<uint32_t> briefs_cell_indices;
    std::vector<uint32_t> cell_point_indptr;
    std::vector<uint32_t> cell_point_indices;

    int pre_brief = -1;

    for (size_t i = 0; i < brief_point.size(); ++i) {
        int memory_id = brief_point[i].second;
        int cur_brief = brief_point[i].first;
        int cur_local_id = _memory_to_local[memory_id];
        int cur_cell_id = cell_assign[cur_local_id];

        while (pre_brief < cur_brief) {
            ++pre_brief;
            _briefs_cell_indptr[pre_brief] = briefs_cell_indices.size();
        }

        if (briefs_cell_indices.size() == 0 || briefs_cell_indices.back() != cur_cell_id) {
            briefs_cell_indices.push_back(cur_cell_id);
            cell_point_indptr.push_back(cell_point_indices.size());
        }

        cell_point_indices.push_back(memory_id);
    }

    _briefs_cell_indptr[brief_count] = briefs_cell_indices.size();
    cell_point_indptr.push_back(cell_point_indices.size());

    _briefs_cell_indices.reset(new int32_t[briefs_cell_indices.size()]);
    memcpy(_briefs_cell_indices.get(),
           briefs_cell_indices.data(),
           sizeof(briefs_cell_indices[0]) * briefs_cell_indices.size());

    _cell_point_indptr.reset(new int32_t[cell_point_indptr.size()]);
    memcpy(_cell_point_indptr.get(),
           cell_point_indptr.data(),
           sizeof(cell_point_indptr[0]) * cell_point_indptr.size());

    _cell_point_indices.reset(new int32_t[cell_point_indices.size()]);
    memcpy(_cell_point_indices.get(),
           cell_point_indices.data(),
           sizeof(cell_point_indices[0]) * cell_point_indices.size());

    _briefs_coarse.reset(new bool[brief_count * _conf.coarse_cluster_count]);
    memset(_briefs_coarse.get(), 0, sizeof(bool) * brief_count * _conf.coarse_cluster_count);

    for (size_t j = 0; j < _conf.total_point_count; ++j) {
        int coarse_id = cell_assign[j] / _conf.fine_cluster_count;

        for (size_t i = _briefs_indptr[j]; i < _briefs_indptr[j + 1]; ++i) {
            int brief_id = _briefs_indices[i];
            _briefs_coarse[brief_id * _conf.coarse_cluster_count + coarse_id] = true;
        }
    }

    int pre_memory_id = -1;

    // 按memory idx的顺序存储
    std::stable_sort(point_brief.begin(), point_brief.end());

    for (size_t j = 0; j < point_brief.size(); ++j) {
        int brief_id = point_brief[j].second;
        int memory_id = point_brief[j].first;
        _briefs_indices[j] = brief_id;

        while (pre_memory_id < memory_id) {
            ++pre_memory_id;
            _briefs_indptr[pre_memory_id] = j;
        }
    }

    _briefs_indptr[_conf.total_point_count] = point_brief.size();

    LOG(INFO) << "convert_local_to_memory_idx Suc.";
    return 0;
}

int MultiBriefPuckIndex::search_nearest_coarse_cluster(
        SearchContext* context,
        const float* feature,
        const uint32_t top_coarse_cnt,
        uint32_t& true_top_coarse) {
    const BriefRequest* request = dynamic_cast<const BriefRequest*>(context->get_request());

    SearchCellData& search_cell_data = context->get_search_cell_data();
    float* cluster_inner_product = search_cell_data.cluster_inner_product;
    matrix_multiplication(
            _coarse_vocab,
            feature,
            _conf.coarse_cluster_count,
            1,
            _conf.feature_dim,
            "TN",
            cluster_inner_product);

    //计算一级聚类中心的距离,使用最大堆
    float* coarse_distance = search_cell_data.coarse_distance;
    uint32_t* coarse_tag = search_cell_data.coarse_tag;
    //初始化最大堆。
    MaxHeap max_heap(top_coarse_cnt, coarse_distance, coarse_tag);

    for (uint32_t c = 0; c < _conf.coarse_cluster_count; ++c) {
        if (!_briefs_coarse[request->briefs[0] * _conf.coarse_cluster_count + c]) {
            // LOG(INFO)<<request->briefs[0]<<" "<<c<<" skip";
            continue;
        }

        if (request->brief_size == 2 &&
            !_briefs_coarse[request->briefs[1] * _conf.coarse_cluster_count + c]) {
            // LOG(INFO)<<request->briefs[0]<<" "<<c<<" skip";
            continue;
        }

        float temp_dist = _coarse_norms[c] - cluster_inner_product[c];

        if (temp_dist < coarse_distance[0]) {
            max_heap.max_heap_update(temp_dist, c);
        }
    }

    true_top_coarse = max_heap.get_heap_size();
    //堆排序
    max_heap.reorder();
    return 0;
}

int MultiBriefPuckIndex::search_nearest_filter_points(
        SearchContext* context,
        const float* feature,
        const uint32_t true_coarse_cnt) {
    if (pre_filter_search(context, feature) != 0) {
        LOG(ERROR) << "cmp filter dist table failed";
        return -1;
    }

    SearchCellData& search_cell_data = context->get_search_cell_data();
    float* cluster_inner_product = search_cell_data.cluster_inner_product;

    matrix_multiplication(
            _fine_vocab,
            feature,
            _conf.fine_cluster_count,
            1,
            _conf.feature_dim,
            "TN",
            cluster_inner_product);
    //一级聚类中心的排序结果
    float* coarse_distance = search_cell_data.coarse_distance;
    uint32_t* coarse_tag = search_cell_data.coarse_tag;
    //过滤阈值
    float pivot = coarse_distance[_conf.search_coarse_count - 1];

    //堆结构
    float* result_distance = context->get_search_point_data().result_distance;
    uint32_t* result_tag = context->get_search_point_data().result_tag;
    MaxHeap filter_heap(_conf.filter_topk, result_distance, result_tag);

    float query_norm = cblas_sdot(_conf.feature_dim, feature, 1, feature, 1);

    auto* visited_list = context->get_visited_list();
    visited_list->reset();
    auto& cur_V = visited_list->curV;
    auto* coarse_mass = visited_list->mass + _conf.coarse_cluster_count * _conf.fine_cluster_count;
    auto* fine_mass = coarse_mass + _conf.coarse_cluster_count;

    for (uint32_t l = 0; l < true_coarse_cnt; ++l) {
        coarse_mass[coarse_tag[l]] = cur_V;
    }

    const BriefRequest* request = dynamic_cast<const BriefRequest*>(context->get_request());
    std::vector<int> cell_point_start(_conf.coarse_cluster_count * _conf.fine_cluster_count, -1);

    if (request->brief_size == 1) {
        for (size_t i = _briefs_cell_indptr[request->briefs[0]];
             i < _briefs_cell_indptr[request->briefs[0] + 1];
             ++i) {
            if (coarse_mass[_briefs_cell_indices[i] / _conf.fine_cluster_count] != cur_V) {
                continue;
            }

            visited_list->mass[_briefs_cell_indices[i]] = cur_V;
            cell_point_start[_briefs_cell_indices[i]] = i;
            fine_mass[_briefs_cell_indices[i] % _conf.fine_cluster_count] = cur_V;
        }
    } else {
        int left_start = _briefs_cell_indptr[request->briefs[0]];
        int left_end = _briefs_cell_indptr[request->briefs[0] + 1];

        int right_start = _briefs_cell_indptr[request->briefs[1]];
        int right_end = _briefs_cell_indptr[request->briefs[1] + 1];

        while (true) {
            while (left_start < left_end &&
                   coarse_mass[_briefs_cell_indices[left_start] / _conf.fine_cluster_count] !=
                           cur_V) {
                ++left_start;
            }

            if (left_start >= left_end) {
                break;
            }

            while (right_start < right_end &&
                   _briefs_cell_indices[right_start] < _briefs_cell_indices[left_start]) {
                ++right_start;
            }

            if (right_start >= right_end) {
                break;
            }

            if (_briefs_cell_indices[right_start] == _briefs_cell_indices[left_start]) {
                visited_list->mass[_briefs_cell_indices[left_start]] = cur_V;
                cell_point_start[_briefs_cell_indices[left_start]] = left_start;
                ++right_start;
                fine_mass[_briefs_cell_indices[left_start] % _conf.fine_cluster_count] = cur_V;
            }

            ++left_start;
        }
    }

    MaxHeap max_heap(
            _conf.fine_cluster_count, search_cell_data.fine_distance, search_cell_data.fine_tag);

    for (uint32_t k = 0; k < _conf.fine_cluster_count; ++k) {
        if (fine_mass[k] == cur_V) {
            max_heap.max_heap_update(-cluster_inner_product[k], k);
        } else {
            max_heap.max_heap_update(std::sqrt(std::numeric_limits<float>::max()), k);
        }
    }

    max_heap.reorder();

    int cmp_cnt = 0;

    for (uint32_t l = 0; l < true_coarse_cnt; ++l) {
        int coarse_id = coarse_tag[l];
        //计算query与当前一级聚类中心下cell的距离
        FineCluster* cur_fine_cluster_list = _coarse_clusters[coarse_id].fine_cell_list;
        float min_dist = _coarse_clusters[coarse_id].min_dist_offset + coarse_distance[l];
        float max_stationary_dist = pivot - coarse_distance[l] - search_cell_data.fine_distance[0];

        for (uint32_t idx = 0; idx < _conf.fine_cluster_count; ++idx) {
            uint32_t k = search_cell_data.fine_tag[idx];
            int cell_id = coarse_id * _conf.fine_cluster_count + k;

            if (visited_list->mass[cell_id] != cur_V) {
                continue;
            }

            if (search_cell_data.fine_distance[idx] + min_dist >= pivot) {
                // LOG(INFO)<<l<<" "<<idx<<" break;";
                break;
            }

            if (cur_fine_cluster_list[k].stationary_cell_dist >= max_stationary_dist) {
                continue;
            }

            float temp_dist = coarse_distance[l] + cur_fine_cluster_list[k].stationary_cell_dist +
                    search_cell_data.fine_distance[idx];
            // int updated_cnt = compute_quantized_distance(context, cur_fine_cluster_list + k,
            // temp_dist, filter_heap);
            int updated_cnt = compute_quantized_distance(
                    context, cell_point_start[cell_id], temp_dist, filter_heap);

            if (updated_cnt > 0) {
                pivot = (filter_heap.get_top_addr()[0] - query_norm) / _conf.radius_rate / 2.0;
            }

            max_stationary_dist = std::min(
                    max_stationary_dist,
                    pivot - coarse_distance[l] - search_cell_data.fine_distance[idx]);
        }
    }

    if (filter_heap.get_heap_size() < _conf.filter_topk) {
        filter_heap.reorder();
    }

    return filter_heap.get_heap_size();
}

int MultiBriefPuckIndex::compute_quantized_distance(
        SearchContext* context,
        const int cell_point_idx,
        const float cell_dist,
        MaxHeap& result_heap) {
    float* result_distance = result_heap.get_top_addr();
    const float* pq_dist_table = context->get_search_point_data().pq_dist_table;

    // point info存储的量化特征对应的参数
    auto& quantization_params = _filter_quantization->get_quantization_params();
    uint32_t* query_sorted_tag = context->get_search_point_data().query_sorted_tag;

    uint32_t updated_cnt = 0;
    auto* id_selector = context->get_selector();
    const BriefRequest* request = dynamic_cast<const BriefRequest*>(context->get_request());

    for (uint32_t i = _cell_point_indptr[cell_point_idx];
         i < _cell_point_indptr[cell_point_idx + 1];
         ++i) {
        int memory_id = _cell_point_indices[i];

        if (request->brief_size == 2) {
            // if (id_selector && !id_selector->is_member(_briefs_encode[local_id])){
            if (!id_selector->is_member(memory_id)) {
                continue;
            }
        }

        const unsigned char* feature = _filter_quantization->get_quantized_feature(memory_id);
        float temp_dist = 2.0 * cell_dist + ((float*)feature)[0];

        if (temp_dist >= result_distance[0]) {
            break;
        }

        const unsigned char* pq_feature =
                (unsigned char*)feature + _filter_quantization->get_fea_offset();
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
        ++updated_cnt;

        if (temp_dist < result_distance[0]) {
            result_heap.max_heap_update(temp_dist, memory_id);
        }
    }

    return updated_cnt;
}

int MultiBriefPuckIndex::search(const Request* brief_request, Response* response) {
    const BriefRequest* request = dynamic_cast<const BriefRequest*>(brief_request);

    if (request == nullptr || request->topk > _conf.topk || request->feature == nullptr) {
        LOG(ERROR) << "topk should <= topk, topk = " << _conf.topk << ", or feature is nullptr";
        return -1;
    }

    if (request->brief_size == 2) {
        if (get_brief_points_cnt(request->briefs[0]) > get_brief_points_cnt(request->briefs[1])) {
            int reorder_bries[2] = {request->briefs[1], request->briefs[0]};
            BriefRequest new_request;
            new_request.brief_size = 2;
            new_request.briefs = reorder_bries;
            new_request.feature = request->feature;
            new_request.topk = request->topk;
            return search(&new_request, response);
        }
    }

    DataHandler<SearchContext> context(_context_pool);

    if (0 != context->reset(_conf)) {
        LOG(ERROR) << "init search context has error.";
        return -1;
    }

    if (request->brief_size == 2) {
        context.get()->init_selector(
                _conf.total_point_count, _briefs_indptr.get(), _briefs_indices.get());
        auto* id_selector = context.get()->get_selector();
        id_selector->set_query_words(request->briefs[1], -1);
    }

    context.get()->set_request(request);
    const float* feature = normalization(context.get(), request->feature);
    uint32_t true_top_coarse = 0;
    //输出query与一级聚类中心的top-search-cell个ID和距离
    int ret = search_nearest_coarse_cluster(
            context.get(), feature, _conf.search_coarse_count, true_top_coarse);

    if (ret < 0) {
        LOG(ERROR) << "search nearest coarse cluster error " << ret;
        return ret;
    }

    //计算query与二级聚类中心的距离，并根据filter特征，筛选子集
    int search_point_cnt = search_nearest_filter_points(context.get(), feature, true_top_coarse);

    if (search_point_cnt < 0) {
        LOG(ERROR) << "search filter points has error.";
        return -1;
    }

    // LOG(INFO)<<"search_nearest_filter_points "<<search_point_cnt;
    MaxHeap result_heap(request->topk, response->distance, response->local_idx);
    ret = rank_topN_points(context.get(), feature, search_point_cnt, result_heap);

    if (ret == 0) {
        response->result_num = result_heap.get_heap_size();
    } else {
        LOG(ERROR) << "rank points after filter has error.";
    }

    return ret;
}
};  // namespace puck