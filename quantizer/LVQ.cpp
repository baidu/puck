#include <algorithm>
#include <iostream>
#include <iterator>
#include <thread>
#include <mutex>
#include <cmath>
#include <stdexcept>
#include <x86intrin.h>
#include <glog/logging.h>

#include "gflags/puck_gflags.h"
#include "tinker/internal/distances.h"
#include "common/normalization.h"
#include "tinker/internal/utils.h"
#include "quantizer/LVQ.h"

namespace puck {

int QuantizerLVQ::train(){
    std::exception_ptr lastException = nullptr;
    std::mutex lastExceptMutex;
    std::string feature_file_path = puck::FLAGS_index_path + "/" + puck::FLAGS_feature_file_name;

    std::ifstream learn_stream(feature_file_path, std::ios::binary | std::ios::ate);
    if(!learn_stream.is_open()){
        LOG(ERROR) << "LVQ train open file error, filename = " << feature_file_path;
        return -1;
    }
    const std::streamoff file_size = learn_stream.tellg();
    if (file_size < 0) {
        LOG(ERROR) << "LVQ train cannot determine feature file size";
        return -1;
    }
    const size_t record_size = _feature_dim * sizeof(float) + sizeof(int);
    const size_t total_point_count = static_cast<size_t>(file_size) / record_size;
    if(total_point_count * record_size != static_cast<size_t>(file_size)){
        LOG(ERROR) << "broken file, "
                   <<"expect size " << total_point_count * (_feature_dim * sizeof(float) + sizeof(int))
                   << ", actual size " << learn_stream.tellg();
        return -1;
    }
    if (total_point_count == 0) {
        LOG(ERROR) << "LVQ train does not support an empty feature file";
        return -1;
    }
    const size_t configured_threads = static_cast<size_t>(std::max(1, puck::FLAGS_threads_count));
    const size_t thread_count = std::min(configured_threads, total_point_count);
    std::vector<std::thread> threads;
    std::vector<std::vector<float>> sum(thread_count, std::vector<float>(_feature_dim, 0.0f));
    // map threads
    for(size_t threadId = 0; threadId < thread_count; ++threadId){
        threads.push_back(std::thread([&, threadId]{
            const size_t thread_chunk_size = 100000;
            const size_t points_per_thread = (total_point_count + thread_count - 1) / thread_count;
            const size_t start_id = threadId * points_per_thread;
            const size_t points_count = std::min(points_per_thread, total_point_count - start_id);
            if (points_count > 0) {
                try {
                    LOG(INFO) << "LVQ train, start_id = " << start_id << " points_count = " <<
                                points_count << " feature_file_name = " << feature_file_path << " threadId = " << threadId;
                    std::ifstream thread_stream(feature_file_path, std::ios::binary);
                    if (!thread_stream.is_open()) {
                        throw std::runtime_error("open feature file failed");
                    }
                    thread_stream.seekg(static_cast<std::streamoff>(start_id * record_size), std::ios::beg);
                    if (!thread_stream.good()) {
                        throw std::runtime_error("seek feature file failed");
                    }
                    std::unique_ptr<float[]> chunk_points(new float[thread_chunk_size * _feature_dim]);
                    for(size_t chunk_start = 0; chunk_start < points_count; chunk_start += thread_chunk_size){
                        const size_t real_thread_chunk_size =
                                std::min(thread_chunk_size, points_count - chunk_start);
                        for (size_t i = 0; i < real_thread_chunk_size; ++i) {
                            int dim = 0;
                            thread_stream.read(reinterpret_cast<char*>(&dim), sizeof(dim));
                            if (!thread_stream || dim != static_cast<int>(_feature_dim)) {
                                throw std::runtime_error("invalid feature dimension");
                            }
                            thread_stream.read(reinterpret_cast<char*>(chunk_points.get() + i * _feature_dim),
                                               static_cast<std::streamsize>(_feature_dim * sizeof(float)));
                            if (!thread_stream) {
                                throw std::runtime_error("read feature data failed");
                            }
                        }
                        for(size_t feaId = 0; feaId < real_thread_chunk_size; feaId++){
                            std::vector<float> normalized;
                            const float* feature = chunk_points.get() + feaId * _feature_dim;
                            if (preprocess_vector(feature, _feature_dim, _whether_norm,
                                                  &normalized) != 0) {
                                LOG(ERROR) << "LVQ train vector preprocessing failed, feature_id="
                                           << start_id + chunk_start + feaId;
                                throw std::runtime_error("LVQ train vector preprocessing failed");
                            }
                            for(uint32_t dimId = 0; dimId < _feature_dim; dimId++){
                                sum[threadId][dimId] += normalized[dimId];
                            }
                        }
                    }
                } catch (...) {
                    std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                    lastException = std::current_exception();
                }
            }
        }));
    }
    for(auto& t : threads){
        t.join();
    }
    if(lastException){
        LOG(ERROR) << "LVQ train failed while processing feature vectors";
        return -1;
    }
    // reduce sum
    for(size_t threadId = 1; threadId < thread_count; threadId++){
        for(uint32_t dimId = 0; dimId < _feature_dim; dimId++){
            sum[0][dimId] += sum[threadId][dimId];
        }
    }
    _mean.resize(_feature_dim, 0);
    for(uint32_t dimId = 0; dimId < _feature_dim; dimId++){
        _mean[dimId] = sum[0][dimId] / total_point_count;
    }
    return 0;
}

int QuantizerLVQ::save(const std::string &filename){
    std::ofstream out(filename);
    if(!out.is_open()){
        LOG(ERROR) << "open file error, filename = " << filename;
        return -1;
    }
    size_t dim = _mean.size();
    puck::QuantType quant_t = quant_type();
    out.write(reinterpret_cast<const char*>(&quant_t), sizeof(puck::QuantType));
    out.write(reinterpret_cast<const char*>(&dim), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(&_bitx), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(&_bity), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(_mean.data()), dim * sizeof(float));

    if(!out.good()){
        LOG(ERROR) << "save LVQ error!";
        return -1;
    }
    LOG(INFO) << "save LVQ success! filename = " << filename;
    return 0;
}

int QuantizerLVQ::load(const std::string &filename){
    std::ifstream in(filename);
    if(!in.is_open()){
        LOG(ERROR) << "open file error, filename = " << filename;
        return -1;
    }
    puck::QuantType quant_t;
    in.read(reinterpret_cast<char*>(&quant_t), sizeof(puck::QuantType));
    if(quant_t != quant_type()){
        LOG(ERROR) << "quant type not match! read typeID = " << static_cast<int>(quant_t) 
                   << ", expect typeID = " << static_cast<int>(quant_type());
        return -1;
    }
    in.read(reinterpret_cast<char*>(&_feature_dim), sizeof(size_t));
    _mean.resize(_feature_dim);

    in.read(reinterpret_cast<char*>(&_bitx), sizeof(_bitx));
    in.read(reinterpret_cast<char*>(&_bity), sizeof(_bity));
    LOG(INFO) << "loading Quantizer LVQ" << _bitx << "x" << _bity;
    _support_rerank = (_bity > 0);
    _data_length = DIV_UP(_feature_dim * _bitx, 8) + DIV_UP(_feature_dim * _bity, 8) + 2 * sizeof(float);

    in.read(reinterpret_cast<char*>(_mean.data()), _feature_dim * sizeof(float));
    if(!in.good()){
        LOG(ERROR) << "load LVQ error!";
        return -1;
    }

    LOG(INFO) << "load LVQ success! filename = " << filename;

    return 0;
}

int QuantizerLVQ::encode(char *des, const float *src, QuantizerParams &params){
    std::vector<float> residual(_feature_dim);
    for(uint32_t dimId = 0; dimId < _feature_dim; dimId++){
        residual[dimId] = src[dimId] - _mean[dimId];
    }

    memset(des, 0, _data_length);

    float minval = std::numeric_limits<float>::max(), maxval = std::numeric_limits<float>::lowest();
    for(uint32_t dimId = 0; dimId < _feature_dim; dimId++){
        minval = std::min(residual[dimId], minval);
        maxval = std::max(residual[dimId], maxval);
    }
    // avoid divide by zero
    if(std::fabs(maxval - minval) < 1e-6){
        maxval = minval + 1e-6;
    }

    // 1st level quant encode
    unsigned char *compressed = reinterpret_cast<unsigned char*>(des);
    encode_impl(compressed, residual.data(), _bitx, minval, maxval);

    // 2st level quant encode
    if(_bity > 0 || support_rerank()){
        float max_code = (1ull << _bitx) - 1;
        float lvq_scale = (maxval - minval) / max_code;
        compressed = reinterpret_cast<unsigned char*>(des + DIV_UP(_feature_dim * _bitx, 8));
        encode_impl(compressed, residual.data(), _bity, -lvq_scale / 2, lvq_scale / 2);
    }

    float *codebook = reinterpret_cast<float*>(
            des + 
            DIV_UP(_feature_dim * _bitx, 8) + 
            DIV_UP(_feature_dim * _bity, 8));
    codebook[0] = minval;
    codebook[1] = maxval;

    return 0;
}

int QuantizerLVQ::decode(float *des, const char *src, QuantizerParams &params){
    memcpy(des, _mean.data(), sizeof(float) * _feature_dim);

    // 1st level decoder
    const unsigned char *compressed = reinterpret_cast<const unsigned char*>(src);
    const float *codebook = 
            reinterpret_cast<const float*>(src + 
                                     DIV_UP(_feature_dim * _bitx, 8) + 
                                     DIV_UP(_feature_dim * _bity, 8));
    float minval = codebook[0];
    float maxval = codebook[1];
    float lvq_scale = (maxval - minval) / ((1ull << _bitx) - 1);
    decode_impl(des, compressed, _bitx, minval, maxval);

    // 2nd level decoder
    if(params.code_level > 1 && support_rerank()){
        compressed = reinterpret_cast<const unsigned char*>(src + DIV_UP(_feature_dim * _bitx, 8));
        decode_impl(des, compressed, _bity, -lvq_scale / 2, lvq_scale / 2);
    }
    else if(params.code_level > 1 && !support_rerank()){
        LOG(ERROR) << "LVQ" << _bitx << "x" << _bity << " does not support rerank";
    }

    return 0;
}

float QuantizerLVQ::sdc_distance(const char *c_u8, const char *v_u8, QuantizerParams &params) noexcept {
    char *cbuffer = params.decoded_buffer;
    char *fbuffer = params.decoded_buffer + _feature_dim * sizeof(float);
    decode(reinterpret_cast<float*>(cbuffer), c_u8, params);
    decode(reinterpret_cast<float*>(fbuffer), v_u8, params);
    return puck::distanceL2(reinterpret_cast<float*>(cbuffer), reinterpret_cast<float*>(fbuffer), _feature_dim);
}

float QuantizerLVQ::adc_distance(const char *c_u8, const float *v_fp32, QuantizerParams &params) noexcept {
    char* cbuffer = params.decoded_buffer;
    decode(reinterpret_cast<float*>(cbuffer), c_u8, params);
    return puck::distanceL2(v_fp32, reinterpret_cast<float*>(cbuffer), _feature_dim);
}

////////////////////////////////////
//      decode simd implement     //
////////////////////////////////////

#if defined(__AVX512F__)
#pragma message("LVQ is using AVX512F SIMD instruction set.")

int QuantizerLVQ::encode_impl(unsigned char *des, float *src, size_t bit, float low, float up){
    float max_code = (1ull << bit) - 1;
    float lvq_scale = (up - low) / max_code;

    int shift = bit;
    int mask = (1ull << bit) - 1;
    __m512 lvq_scale_reg = _mm512_set1_ps(1. / lvq_scale);
    __m512 low_reg = _mm512_set1_ps(low);

    size_t simd_step = sizeof(__m512i) / 4;         // width  of simd register: K
    size_t simd_sz   = sizeof(int32_t) * 8 / bit;   // length of simd register: H
    // one simd register can hold H*K bit_x values (bottle -> top, left -> right)
    size_t simd_values = simd_sz * simd_step;
    // each lane holds a 32-bit value
    // -------------SIMD---------------
    // lane 0:  lane 1:   ...  lane K-1:
    // S(H-1)K  S(H-1)K+1 ...  SHK-1
    // ...      ...       ...  ...
    // S2K      S2K+1     ...  S3K-1
    // SK       SK+1      ...  S2K-1
    // S0       S1        ...  SK-1

    size_t dimId = 0;
    for(; dimId + simd_values <= _feature_dim; dimId += simd_values){
        __m512i encoded_data = _mm512_setzero_si512();
        for(ssize_t i = simd_values - simd_step; i >= 0; i -= simd_step){
            __m512 data = _mm512_loadu_ps(&src[dimId + i]);
            data = _mm512_add_ps(_mm512_mul_ps(lvq_scale_reg, _mm512_sub_ps(data, low_reg)), _mm512_set1_ps(0.5));  // val = (val - low) / lvq_scale + 0.5;
            data = _mm512_min_ps(data, _mm512_set1_ps(max_code));  // val = std::min(val, max_code);
            __m512i udata = _mm512_cvtps_epi32(_mm512_roundscale_ps(data, _MM_FROUND_TO_ZERO));
            __m512 residual = _mm512_loadu_ps(&src[dimId + i]);
            encoded_data = _mm512_or_si512(_mm512_slli_epi32(encoded_data, shift), udata);

            residual = _mm512_sub_ps(residual, _mm512_add_ps(low_reg, _mm512_div_ps(_mm512_cvtepi32_ps(udata), lvq_scale_reg)));  // val -= low + lvq_scale * val;
            _mm512_storeu_ps(&src[dimId + i], residual);  // update residual 
        }
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(&des[dimId * bit / 8]), encoded_data);
    }

    // handle remaining
    #pragma GCC unroll 4
    for(; dimId < _feature_dim; dimId++){
        uint32_t val = static_cast<uint32_t>((src[dimId] - low) / lvq_scale + 0.5);
        val = std::min(val, static_cast<uint32_t>(mask));
        src[dimId] -= lvq_scale * val + low;
        des[dimId * bit / 8] |= static_cast<unsigned char>(val << (dimId * bit % 8));
    }

    return 0;
}

int QuantizerLVQ::decode_impl(float *des, const unsigned char *src, size_t bit, float low, float up){
    float max_code = (1ull << bit) - 1;
    float lvq_scale = (up - low) / max_code;

    int shift = bit;
    int mask = (1ull << bit) - 1;
    __m512i mask_reg = _mm512_set1_epi32(mask);
    __m512 lvq_scale_reg = _mm512_set1_ps(lvq_scale);
    __m512 low_reg = _mm512_set1_ps(low);

    size_t simd_step = sizeof(__m512i) / 4;         // width  of simd register: K
    size_t simd_sz   = sizeof(int32_t) * 8 / bit;   // length of simd register: H
    // one simd register can hold H*K bit_x values (bottle -> top, left -> right)
    size_t simd_values = simd_sz * simd_step;
    // each lane holds a 32-bit value
    // -------------SIMD---------------
    // lane 0:  lane 1:   ...  lane K-1:
    // S(H-1)K  S(H-1)K+1 ...  SHK-1
    // ...      ...       ...  ...
    // S2K      S2K+1     ...  S3K-1
    // SK       SK+1      ...  S2K-1
    // S0       S1        ...  SK-1
    uint32_t dimId = 0;
    for(; dimId + simd_values <= _feature_dim; dimId += simd_values){
        __m512i data = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&src[dimId * bit / 8]));
        for(size_t i = 0; i < simd_values; i += simd_step){
            // mask out except every high <bit>
            __m512i low_data = _mm512_and_si512(data, mask_reg);
            // load des 
            __m512 des_data = _mm512_loadu_ps(&des[dimId + i]);
            // decode
            __m512 decoded_data = _mm512_mul_ps(lvq_scale_reg, _mm512_cvtepi32_ps(low_data));
            decoded_data = _mm512_add_ps(decoded_data, low_reg);
            // store
            des_data = _mm512_add_ps(des_data, decoded_data);
            _mm512_storeu_ps(&des[dimId + i], des_data);
            // shift out low <bit>
            data = _mm512_srli_epi32(data, shift);
        }
    }

    // handle remaining
    #pragma GCC unroll 4
    for(; dimId < _feature_dim; dimId++){
        uint32_t val = (src[dimId * bit / 8] >> (dimId * bit % 8)) & mask;
        des[dimId] += lvq_scale * val + low;
    }
    return 0;
}

#elif defined(__AVX2__)
#pragma message("LVQ is using AVX2 SIMD instruction set.")

int QuantizerLVQ::encode_impl(unsigned char *des, float *src, size_t bit, float low, float up){
    float max_code = (1ull << bit) - 1;
    float lvq_scale = (up - low) / max_code;

    int shift = bit;
    int mask = (1ull << bit) - 1;
    __m256 lvq_scale_reg = _mm256_set1_ps(1. / lvq_scale);
    __m256 low_reg = _mm256_set1_ps(low);

    size_t simd_step = sizeof(__m256i) / 4;         // width  of simd register: K
    size_t simd_sz   = sizeof(int32_t) * 8 / bit;   // length of simd register: H
    // one simd register can hold H*K bit_x values (bottle -> top, left -> right)
    size_t simd_values = simd_sz * simd_step;
    // each lane holds a 32-bit value
    // -------------SIMD---------------
    // lane 0:  lane 1:   ...  lane K-1:
    // S(H-1)K  S(H-1)K+1 ...  SHK-1
    // ...      ...       ...  ...
    // S2K      S2K+1     ...  S3K-1
    // SK       SK+1      ...  S2K-1
    // S0       S1        ...  SK-1

    size_t dimId = 0;
    for(; dimId + simd_values <= _feature_dim; dimId += simd_values){
        __m256i encoded_data = _mm256_setzero_si256();
        for(ssize_t i = simd_values - simd_step; i >= 0; i -= simd_step){
            __m256 data = _mm256_loadu_ps(&src[dimId + i]);
            data = _mm256_add_ps(_mm256_mul_ps(lvq_scale_reg, _mm256_sub_ps(data, low_reg)), _mm256_set1_ps(0.5));  // val = (val - low) / lvq_scale + 0.5;
            data = _mm256_min_ps(data, _mm256_set1_ps(max_code));  // val = std::min(val, max_code);
            __m256i udata = _mm256_cvtps_epi32(_mm256_round_ps(data, _MM_FROUND_TO_ZERO));
            __m256 residual = _mm256_loadu_ps(&src[dimId + i]);
            encoded_data = _mm256_or_si256(_mm256_slli_epi32(encoded_data, shift), udata);

            residual = _mm256_sub_ps(residual, _mm256_add_ps(low_reg, _mm256_div_ps(_mm256_cvtepi32_ps(udata), lvq_scale_reg)));  // val -= low + lvq_scale * val;
            _mm256_storeu_ps(&src[dimId + i], residual);  // update residual 
        }
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&des[dimId * bit / 8]), encoded_data);
    }

    // handle remaining
    #pragma GCC unroll 4
    for(; dimId < _feature_dim; dimId++){
        uint32_t val = static_cast<uint32_t>((src[dimId] - low) / lvq_scale + 0.5);
        val = std::min(val, static_cast<uint32_t>(mask));
        src[dimId] -= lvq_scale * val + low;
        des[dimId * bit / 8] |= static_cast<unsigned char>(val << (dimId * bit % 8));
    }

    return 0;
}

int QuantizerLVQ::decode_impl(float *des, const unsigned char *src, size_t bit, float low, float up){
    float max_code = (1ull << bit) - 1;
    float lvq_scale = (up - low) / max_code;

    int shift = bit;
    int mask = (1ull << bit) - 1;
    __m256i mask_reg = _mm256_set1_epi32(mask);
    __m256 lvq_scale_reg = _mm256_set1_ps(lvq_scale);
    __m256 low_reg = _mm256_set1_ps(low);

    size_t simd_step = sizeof(__m256i) / 4;         // width  of simd register: K
    size_t simd_sz   = sizeof(int32_t) * 8 / bit;   // length of simd register: H
    // one simd register can hold H*K bit_x values (bottle -> top, left -> right)
    size_t simd_values = simd_sz * simd_step;
    // each lane holds a 32-bit value
    // -------------SIMD---------------
    // lane 0:  lane 1:   ...  lane K-1:
    // S(H-1)K  S(H-1)K+1 ...  SHK-1
    // ...      ...       ...  ...
    // S2K      S2K+1     ...  S3K-1
    // SK       SK+1      ...  S2K-1
    // S0       S1        ...  SK-1
    uint32_t dimId = 0;
    for(; dimId + simd_values <= _feature_dim; dimId += simd_values){
        __m256i data = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&src[dimId * bit / 8]));
        for(size_t i = 0; i < simd_values; i += simd_step){
            // mask out except every high <bit>
            __m256i low_data = _mm256_and_si256(data, mask_reg);
            // load des 
            __m256 des_data = _mm256_loadu_ps(&des[dimId + i]);
            // decode
            __m256 decoded_data = _mm256_mul_ps(lvq_scale_reg, _mm256_cvtepi32_ps(low_data));
            decoded_data = _mm256_add_ps(decoded_data, low_reg);
            // store
            des_data = _mm256_add_ps(des_data, decoded_data);
            _mm256_storeu_ps(&des[dimId + i], des_data);
            // shift out low <bit>
            data = _mm256_srli_epi32(data, shift);
        }
    }

    // handle remaining
    #pragma GCC unroll 4
    for(; dimId < _feature_dim; dimId++){
        uint32_t val = (src[dimId * bit / 8] >> (dimId * bit % 8)) & mask;
        des[dimId] += lvq_scale * val + low;
    }
    return 0;
}

#elif defined(__SSE__)
#pragma message("LVQ is using SSE SIMD instruction set.")

int QuantizerLVQ::encode_impl(unsigned char *des, float *src, size_t bit, float low, float up){
    float max_code = (1ull << bit) - 1;
    float lvq_scale = (up - low) / max_code;

    int shift = bit;
    int mask = (1ull << bit) - 1;
    __m128 lvq_scale_reg = _mm_set1_ps(1. / lvq_scale);
    __m128 low_reg = _mm_set1_ps(low);

    size_t simd_step = sizeof(__m128i) / 4;         // width  of simd register: K
    size_t simd_sz   = sizeof(int32_t) * 8 / bit;   // length of simd register: H
    // one simd register can hold H*K bit_x values (bottle -> top, left -> right)
    size_t simd_values = simd_sz * simd_step;
    // each lane holds a 32-bit value
    // -------------SIMD---------------
    // lane 0:  lane 1:   ...  lane K-1:
    // S(H-1)K  S(H-1)K+1 ...  SHK-1
    // ...      ...       ...  ...
    // S2K      S2K+1     ...  S3K-1
    // SK       SK+1      ...  S2K-1
    // S0       S1        ...  SK-1

    size_t dimId = 0;
    for(; dimId + simd_values <= _feature_dim; dimId += simd_values){
        __m128i encoded_data = _mm_setzero_si128();
        for(ssize_t i = simd_values - simd_step; i >= 0; i -= simd_step){
            __m128 data = _mm_loadu_ps(&src[dimId + i]);
            data = _mm_add_ps(_mm_mul_ps(lvq_scale_reg, _mm_sub_ps(data, low_reg)), _mm_set1_ps(0.5));  // val = (val - low) / lvq_scale + 0.5;
            data = _mm_min_ps(data, _mm_set1_ps(max_code));  // val = std::min(val, max_code);
            __m128i udata = _mm_cvtps_epi32(_mm_round_ps(data, _MM_FROUND_TO_ZERO));
            __m128 residual = _mm_loadu_ps(&src[dimId + i]);
            encoded_data = _mm_or_si128(_mm_slli_epi32(encoded_data, shift), udata);

            residual = _mm_sub_ps(residual, _mm_add_ps(low_reg, _mm_div_ps(_mm_cvtepi32_ps(udata), lvq_scale_reg)));  // val -= low + lvq_scale * val;
            _mm_storeu_ps(&src[dimId + i], residual);  // update residual 
        }
        _mm_storeu_si128(reinterpret_cast<__m128i*>(&des[dimId * bit / 8]), encoded_data);
    }

    // handle remaining
    #pragma GCC unroll 4
    for(; dimId < _feature_dim; dimId++){
        uint32_t val = static_cast<uint32_t>((src[dimId] - low) / lvq_scale + 0.5);
        val = std::min(val, static_cast<uint32_t>(mask));
        src[dimId] -= lvq_scale * val + low;
        des[dimId * bit / 8] |= static_cast<unsigned char>(val << (dimId * bit % 8));
    }

    return 0;
}

int QuantizerLVQ::decode_impl(float *des, const unsigned char *src, size_t bit, float low, float up){
    float max_code = (1ull << bit) - 1;
    float lvq_scale = (up - low) / max_code;

    int shift = bit;
    int mask = (1ull << bit) - 1;
    __m128i mask_reg = _mm_set1_epi32(mask);
    __m128 lvq_scale_reg = _mm_set1_ps(lvq_scale);
    __m128 low_reg = _mm_set1_ps(low);

    size_t simd_step = sizeof(__m128i) / 4;         // width  of simd register: K
    size_t simd_sz   = sizeof(int32_t) * 8 / bit;   // length of simd register: H
    // one simd register can hold H*K bit_x values (bottle -> top, left -> right)
    size_t simd_values = simd_sz * simd_step;
    // each lane holds a 32-bit value
    // -------------SIMD---------------
    // lane 0:  lane 1:   ...  lane K-1:
    // S(H-1)K  S(H-1)K+1 ...  SHK-1
    // ...      ...       ...  ...
    // S2K      S2K+1     ...  S3K-1
    // SK       SK+1      ...  S2K-1
    // S0       S1        ...  SK-1
    uint32_t dimId = 0;
    for(; dimId + simd_values <= _feature_dim; dimId += simd_values){
        __m128i data = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&src[dimId * bit / 8]));
        for(size_t i = 0; i < simd_values; i += simd_step){
            // mask out except every high <bit>
            __m128i low_data = _mm_and_si128(data, mask_reg);
            // load des 
            __m128 des_data = _mm_loadu_ps(&des[dimId + i]);
            // decode
            __m128 decoded_data = _mm_mul_ps(lvq_scale_reg, _mm_cvtepi32_ps(low_data));
            decoded_data = _mm_add_ps(decoded_data, low_reg);
            // store
            des_data = _mm_add_ps(des_data, decoded_data);
            _mm_storeu_ps(&des[dimId + i], des_data);
            // shift out low <bit>
            data = _mm_srli_epi32(data, shift);
        }
    }

    // handle remaining
    #pragma GCC unroll 4
    for(; dimId < _feature_dim; dimId++){
        uint32_t val = (src[dimId * bit / 8] >> (dimId * bit % 8)) & mask;
        des[dimId] += lvq_scale * val + low;
    }
    return 0;
}

#else
#pragma message("LVQ is using None/Native SIMD instruction set.")

int QuantizerLVQ::encode_impl(unsigned char *des, float *src, size_t bit, float low, float up){
    float max_code = (1ull << bit) - 1;
    float lvq_scale = (up - low) / max_code;

    // encode src to des
    for(uint32_t dimId = 0; dimId < _feature_dim; dimId++){
        float val = (src[dimId] - low) / lvq_scale + .5f;
        val = val > max_code ? max_code : val;
        val = val < 0 ? 0 : val;
        unsigned char val_u8 = static_cast<unsigned char>(val);
        // overwrite src with residual
        src[dimId] -= lvq_scale * val_u8 + low;
        des[dimId * bit / 8] |= static_cast<unsigned char>(val_u8 << (dimId * bit % 8));
    }
    return 0;
}

int QuantizerLVQ::decode_impl(float *des, const unsigned char *src, size_t bit, float low, float up){
    float max_code = (1ull << bit) - 1;
    float lvq_scale = (up - low) / max_code;

    for(uint32_t dimId = 0; dimId < _feature_dim; dimId++){
        float val = static_cast<float>((src[dimId * bit / 8] >> (dimId * bit % 8)) & ((1ull << bit) - 1));
        des[dimId] += val * lvq_scale + low;
    }
    return 0;
}

#endif


} // namespace puck
