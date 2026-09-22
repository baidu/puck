#include "tinker/internal/distances.h"

namespace puck {

#if defined(__AVX512F__)
#pragma message("AVX512F is enabled")

char *TMP_ARRAY(){
    constexpr size_t SZ = sizeof(__m512i);
    static thread_local char result[SZ];
    return result;
}

float distanceL2(const float* a, const float* b, size_t n) {
    __m512 sum = _mm512_setzero_ps();

    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 va = _mm512_loadu_ps(&a[i]);
        __m512 vb = _mm512_loadu_ps(&b[i]);
        __m512 diff = _mm512_sub_ps(va, vb);
        __m512 sq = _mm512_mul_ps(diff, diff);
        sum = _mm512_add_ps(sum, sq);
    }

    float* result = reinterpret_cast<float*>(TMP_ARRAY());
    _mm512_storeu_ps(result, sum);

    float ret = result[0] + result[1] + result[2] + result[3] +
                result[4] + result[5] + result[6] + result[7] +
                result[8] + result[9] + result[10] + result[11] +
                result[12] + result[13] + result[14] + result[15];

    #pragma GCC unroll 4
    for (; i < n; ++i) {
        float diff = a[i] - b[i];
        ret += diff * diff;
    }

    return ret;
}

#elif defined(__AVX2__)
#pragma message("AVX2 is enabled")

char *TMP_ARRAY(){
    constexpr size_t SZ = sizeof(__m256i);
    static thread_local char result[SZ];
    return result;
}

// L2 distance
float distanceL2(const float* a, const float* b, size_t n) {
    __m256 sum = _mm256_setzero_ps();

    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);

        __m256 diff = _mm256_sub_ps(va, vb);
        __m256 sq = _mm256_mul_ps(diff, diff);

        sum = _mm256_add_ps(sum, sq);
    }

    float* result = reinterpret_cast<float*>(TMP_ARRAY());
    _mm256_storeu_ps(result, sum);

    float ret = result[0] + result[1] + result[2] + result[3] +
                result[4] + result[5] + result[6] + result[7];

    #pragma GCC unroll 4
    for (; i < n; ++i) {
        float diff = a[i] - b[i];
        ret += diff * diff;
    }

    return ret;
}

#elif defined(__SSE__)
#pragma message("SSE is enabled")

char *TMP_ARRAY(){
    constexpr size_t SZ = sizeof(__m128i);
    static thread_local char result[SZ];
    return result;
}

// L2 distance
float distanceL2(const float* a, const float* b, size_t n) {
    __m128 sum = _mm_setzero_ps();

    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(&a[i]);
        __m128 vb = _mm_loadu_ps(&b[i]);

        __m128 diff = _mm_sub_ps(va, vb);
        __m128 sq = _mm_mul_ps(diff, diff);

        sum = _mm_add_ps(sum, sq);
    }

    float* result = reinterpret_cast<float*>(TMP_ARRAY());
    _mm_storeu_ps(result, sum);
    float ret = result[0] + result[1] + result[2] + result[3];

    #pragma GCC unroll 4
    for (; i < n; ++i) {
        float diff = a[i] - b[i];
        ret += diff * diff;
    }

    return ret;
}

#else
#pragma message("No SIMD is enabled")

float distanceL2(const float* a, const float* b, size_t n) {
    float ret = 0;
    #pragma GCC unroll 4
    for (size_t i = 0; i < n; ++i) {
        float diff = a[i] - b[i];
        ret += diff * diff;
    }
    return ret;
}

#endif

} // namespace puck
