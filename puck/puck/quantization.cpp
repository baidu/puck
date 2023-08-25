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
 * @file quantization.cpp
 * @author huangben@baidu.com
 * @author yinjie06@baidu.com
 * @date 2021/7/20 11:17
 * @brief
 *
 **/
#include<thread>
#include <numeric>
#include <fstream>
#include <functional>
#include <unistd.h>
#include <mkl_cblas.h>
#include "puck/puck/quantization.h"
#include "puck/hierarchical_cluster/hierarchical_cluster_index.h"
#include <glog/logging.h>
namespace faiss {
void fvec_L2sqr_ny(float* dis, const float* x,
                   const float* y, size_t d, size_t ny);
};

namespace puck {
void QuantizationParams::show() {
    LOG(INFO) << "QuantizationParams.dim = " << dim << ", QuantizationParams.ks = " << ks
              << ", QuantizationParams.lsq = " << lsq << ", QuantizationParams.nsq = " << nsq;
}

Quantization::Quantization(const QuantizationParams& params,
                           uint32_t point_count) : _per_subspace_len(sizeof(char)),
    _fea_offset(std::ceil(1.0 * sizeof(float) / _per_subspace_len)) {
    _params = params;
    _coodbooks.reset(nullptr);
    _quantized_feature.reset(nullptr);

    if (_params.ks != 256) {
        return;
    }

    _total_point_count = point_count;
    _per_fea_len = _params.nsq * _per_subspace_len;
    _per_fea_len += _fea_offset;
    LOG(INFO) << "_per_fea_len=" << _per_fea_len;
    LOG(INFO) << "_per_subspace_len=" << _per_subspace_len;
    LOG(INFO) << "_params.nsq=" << _params.nsq;
    init_codebooks_memory();
}

struct ThreadReaderParam {
    std::string file_name;
    uint32_t thread_idx;
    uint32_t start_id;
    uint32_t count;
    uint32_t read_len;
    uint64_t file_offset;
    uint32_t read_offset;
};

void load_bytes_array_thread(const ThreadReaderParam reader, unsigned char* code_bytes,
                             const uint32_t* local_2memory_idx) {
    std::ifstream input_file(reader.file_name.c_str(), std::ios::binary | std::ios::in);
    u_int64_t curr_file_offset = (u_int64_t)reader.start_id * reader.read_len + reader.file_offset;
    input_file.seekg(curr_file_offset);

    for (uint32_t i = 0; i < reader.count; ++i) {
        uint32_t memory_idx = reader.start_id + i;

        if (local_2memory_idx != nullptr) {
            memory_idx = local_2memory_idx[reader.start_id + i];
        }

        u_int64_t pq_curr_offset = (u_int64_t)memory_idx * (reader.read_len + reader.read_offset) +
                                   reader.read_offset;
        input_file.read((char*)(code_bytes + pq_curr_offset), reader.read_len);

        if ((i + 1) % 1000000 == 0) {
            LOG(INFO) << "loading index file " << reader.file_name << " thread " << reader.thread_idx << " processed "
                      << 1.0 * i / reader.count;
        }
    }

    input_file.close();
}

bool check_file_length_info(const std::string& file_name,
                            const uint64_t file_length);

int Quantization::init_codebooks_memory() {
    u_int64_t pq_codebook_length = (u_int64_t)_params.nsq * _params.ks * _params.lsq;
    _coodbooks.reset(new float[pq_codebook_length]);

    if (_coodbooks.get() == nullptr) {
        LOG(FATAL) << "malloc memory quantization codebooks " << pq_codebook_length << " error.";
        return -1;
    }

    return 0;
}

int Quantization::load_codebooks(const std::string& codebook_file) {
    //int ret = fvecs_read(codebook_file.c_str(), _params.lsq,
    //                     _params.nsq * _params.ks, _coodbooks.get());
    int ret = read_fvec_format(codebook_file.c_str(), _params.lsq,
                               _params.nsq * _params.ks, _coodbooks.get());

    if (ret != int(_params.nsq * _params.ks)) {
        LOG(FATAL) << "load file error, file : " << codebook_file << " feature_dim : " <<
                   _params.lsq
                   << " number : " << _params.nsq* _params.ks << " return code : " << ret;
        return -1;
    }

    return 0;
}

int Quantization::init_quantized_feature_memory() {
    uint64_t pq_feature_length = (u_int64_t)_total_point_count * _per_fea_len;
    LOG(INFO) << "init quantized feature memory, length = " << pq_feature_length;
    void* memb = nullptr;
    int32_t pagesize = getpagesize();
    pq_feature_length =  pq_feature_length + (pagesize - pq_feature_length % pagesize);
    int err = posix_memalign(&memb, pagesize, pq_feature_length);

    if (err != 0) {
        std::runtime_error("alloc_aligned_mem_failed errno=" + errno);
        return -1;
    }

    _quantized_feature.reset(reinterpret_cast<unsigned char*>(memb));

    if (_quantized_feature.get() == nullptr) {
        LOG(FATAL) << "malloc memory quantized feature vector " << pq_feature_length << " error.";
        return -1;
    }

    return 0;
}

int Quantization::load_quantized_feature(const std::string& quantization_file,
        const uint32_t* local_2memory_idx) {
    if (_params.ks != 256) {
        LOG(FATAL) << "only for ks = 256";
        return -1;
    }

    LOG(INFO) << "load quantized feature " << quantization_file << "; _total_point_count = " <<
              _total_point_count;
    //文件长度检查
    u_int64_t pq_feature_length = (u_int64_t)_total_point_count * _per_fea_len;
    bool is_ok = check_file_length_info(quantization_file, pq_feature_length);

    if (is_ok == false) {
        return -1;
    }

    if (init_quantized_feature_memory() != 0) {
        return -1;
    }

    //多线程加载文件
    uint32_t threads_count = std::thread::hardware_concurrency();
    uint32_t per_thread_count = ceil(1.0 * _total_point_count / threads_count);
    std::vector<std::thread> threads;
    ThreadReaderParam params;
    params.file_name = quantization_file;
    params.read_len = _per_fea_len;
    params.file_offset = 0;
    params.read_offset = 0;
    LOG(INFO) << "params.read_len = " << params.read_len << " " << params.read_offset << " " <<
              (local_2memory_idx == nullptr);

    for (uint32_t thread_id = 0; thread_id < threads_count; ++thread_id) {
        params.start_id = thread_id * per_thread_count;
        params.count = std::min(per_thread_count, _total_point_count - params.start_id);
        params.thread_idx = thread_id;
        threads.push_back(std::thread(std::bind(load_bytes_array_thread,
                                                params, _quantized_feature.get(), local_2memory_idx)));
    }

    for (uint32_t thread_id = 0; thread_id < threads_count; ++thread_id) {
        threads[thread_id].join();
    }

    LOG(INFO) << "load quantized feature " << quantization_file << " suc.";
    return 0;
}

int Quantization::load(const std::string& codebook_file, const std::string& quantized_feafile,
                       const uint32_t* local_2memory_idx) {
    if (_params.ks != 256) {
        LOG(FATAL) << "only for ks = 256";
        return -1;
    }

    if (load_codebooks(codebook_file) != 0) {
        return -1;
    }

    if (local_2memory_idx != nullptr) {
        if (load_quantized_feature(quantized_feafile, local_2memory_idx) != 0) {
            return -1;
        }
    } else {
        std::vector<uint32_t> temp_cnts_index(_total_point_count);
        std::iota(temp_cnts_index.begin(), temp_cnts_index.end(), 0);

        if (load_quantized_feature(quantized_feafile, temp_cnts_index.data()) != 0) {
            return -1;
        }
    }

    return 0;
}

std::unique_ptr<float[]> Quantization::decode(uint64_t idx) const {
    unsigned char* pq_feature = get_quantized_feature(idx);
    pq_feature += _fea_offset;
    std::unique_ptr<float[]> residual(new float[_params.nsq * _params.lsq]);

    for (uint32_t m = 0; m < _params.nsq; ++m) {
        const float* codeword = get_coodbooks() + m * _params.ks * _params.lsq + pq_feature[m] * _params.lsq;

        for (uint32_t d = 0; d < _params.lsq; ++d) {
            residual.get()[m * _params.lsq + d] = codeword[d];
        }
    }

    return std::move(residual);
}

int Quantization::set_static_value_of_formula(uint64_t idx, float* vocab) {
    if (idx >= _total_point_count || !vocab) {
        return -1;
    }

    float* static_dist = (float*)get_quantized_feature(idx);
    std::unique_ptr<float[]> residual = decode(idx);
    static_dist[0] = 2 * cblas_sdot(_params.dim, residual.get(), 1, vocab, 1);

    for (uint32_t idx = 0; idx < _params.dim; ++idx) {
        vocab[idx] += residual[idx];
    }

    return 0;
}

int Quantization::get_dist_table(const float*  feature, float* dist_table) const {
    for (uint32_t m = 0; m < _params.nsq; ++m) {
        faiss::fvec_L2sqr_ny(dist_table + m * _params.ks,
                             feature + m * _params.lsq,
                             _coodbooks.get() + (u_int64_t)m * _params.ks * _params.lsq,
                             _params.lsq,
                             _params.ks);
    }

    return 0;
}
int write_fvec_format(const char* file_name, const uint32_t dim, const uint64_t n, const float* fea_vocab);
int Quantization::save_coodbooks(const std::string& file_name) const {
    return write_fvec_format(file_name.c_str(), _params.lsq, _params.nsq * _params.ks, get_coodbooks());
}

int Quantization::save_index(const std::string& file_name) const {
    FILE* f = fopen(file_name.c_str(), "wb");

    if (f == nullptr) {
        LOG(ERROR) << "cannot open " << file_name << " for writing";
        return -1;
    }

    unsigned char* feature_data = get_quantized_feature(0);
    long ret = fwrite(feature_data, get_per_fea_len(), _total_point_count, f);
    fclose(f);

    if (ret != _total_point_count) {
        LOG(ERROR) << "writint to " << file_name << " has error";
        return -1;
    }

    return 0;
}

}//namespace puck