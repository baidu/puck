/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-
#include <cmath>
#include <cassert>
#ifdef __SSE__
#include <immintrin.h>
#endif

#ifdef __aarch64__
#include  <arm_neon.h>
#endif
namespace faiss {

/// Squared L2 distance between two vectors
float fvec_L2sqr(
    const float* x,
    const float* y,
    size_t d);

void fvec_L2sqr_ny_ref(float* dis,
                       const float* x,
                       const float* y,
                       size_t d, size_t ny) {
    for (size_t i = 0; i < ny; i++) {
        dis[i] = fvec_L2sqr(x, y, d);
        y += d;
    }
}

/*********************************************************
 * SSE and AVX implementations
*********************************************************/
#ifdef __SSE__

// reads 0 <= d < 4 floats as __m128
static inline __m128 masked_read(int d, const float* x) {
    assert(0 <= d && d < 4);
    __attribute__((__aligned__(16))) float buf[4] = {0, 0, 0, 0};

    switch (d) {
    case 3:
        buf[0] = x[0];
        buf[1] = x[1];
        buf[2] = x[2];
        break;

    case 2:
        buf[0] = x[0];
        buf[1] = x[1];
        break;

    case 1:
        buf[0] = x[0];
        break;
    }

    return _mm_load_ps(buf);
}

namespace {
float sqr(float x) {
    return x * x;
}


void fvec_L2sqr_ny_D1(float* dis, const float* x,
                      const float* y, size_t ny) {
    float x0s = x[0];
    __m128 x0 = _mm_set_ps(x0s, x0s, x0s, x0s);

    size_t i = 0;

    for (i = 0; i + 3 < ny; i += 4) {
        __m128 tmp, accu;
        tmp = x0 - _mm_loadu_ps(y);
        y += 4;
        accu = tmp * tmp;
        dis[i] = _mm_cvtss_f32(accu);
        tmp = _mm_shuffle_ps(accu, accu, 1);
        dis[i + 1] = _mm_cvtss_f32(tmp);
        tmp = _mm_shuffle_ps(accu, accu, 2);
        dis[i + 2] = _mm_cvtss_f32(tmp);
        tmp = _mm_shuffle_ps(accu, accu, 3);
        dis[i + 3] = _mm_cvtss_f32(tmp);
    }

    while (i < ny) { // handle non-multiple-of-4 case
        dis[i++] = sqr(x0s - *y++);
    }
}

void fvec_L2sqr_ny_D2(float* dis, const float* x,
                      const float* y, size_t ny) {
    __m128 x0 = _mm_set_ps(x[1], x[0], x[1], x[0]);

    size_t i = 0;

    for (i = 0; i + 1 < ny; i += 2) {
        __m128 tmp, accu;
        tmp = x0 - _mm_loadu_ps(y);
        y += 4;
        accu = tmp * tmp;
        accu = _mm_hadd_ps(accu, accu);
        dis[i] = _mm_cvtss_f32(accu);
        accu = _mm_shuffle_ps(accu, accu, 3);
        dis[i + 1] = _mm_cvtss_f32(accu);
    }

    if (i < ny) { // handle odd case
        dis[i] = sqr(x[0] - y[0]) + sqr(x[1] - y[1]);
    }
}

void fvec_L2sqr_ny_D4(float* dis, const float* x,
                      const float* y, size_t ny) {
    __m128 x0 = _mm_loadu_ps(x);

    for (size_t i = 0; i < ny; i++) {
        __m128 tmp, accu;
        tmp = x0 - _mm_loadu_ps(y);
        y += 4;
        accu = tmp * tmp;
        accu = _mm_hadd_ps(accu, accu);
        accu = _mm_hadd_ps(accu, accu);
        dis[i] = _mm_cvtss_f32(accu);
    }
}

void fvec_L2sqr_ny_D8(float* dis, const float* x,
                      const float* y, size_t ny) {
    __m128 x0 = _mm_loadu_ps(x);
    __m128 x1 = _mm_loadu_ps(x + 4);

    for (size_t i = 0; i < ny; i++) {
        __m128 tmp, accu;
        tmp = x0 - _mm_loadu_ps(y);
        y += 4;
        accu = tmp * tmp;
        tmp = x1 - _mm_loadu_ps(y);
        y += 4;
        accu += tmp * tmp;
        accu = _mm_hadd_ps(accu, accu);
        accu = _mm_hadd_ps(accu, accu);
        dis[i] = _mm_cvtss_f32(accu);
    }
}

void fvec_L2sqr_ny_D12(float* dis, const float* x,
                       const float* y, size_t ny) {
    __m128 x0 = _mm_loadu_ps(x);
    __m128 x1 = _mm_loadu_ps(x + 4);
    __m128 x2 = _mm_loadu_ps(x + 8);

    for (size_t i = 0; i < ny; i++) {
        __m128 tmp, accu;
        tmp = x0 - _mm_loadu_ps(y);
        y += 4;
        accu = tmp * tmp;
        tmp = x1 - _mm_loadu_ps(y);
        y += 4;
        accu += tmp * tmp;
        tmp = x2 - _mm_loadu_ps(y);
        y += 4;
        accu += tmp * tmp;
        accu = _mm_hadd_ps(accu, accu);
        accu = _mm_hadd_ps(accu, accu);
        dis[i] = _mm_cvtss_f32(accu);
    }
}

} // anonymous namespace

void fvec_L2sqr_ny(float* dis, const float* x,
                   const float* y, size_t d, size_t ny) {
    // optimized for a few special cases
    switch (d) {
    case 1:
        fvec_L2sqr_ny_D1(dis, x, y, ny);
        return;

    case 2:
        fvec_L2sqr_ny_D2(dis, x, y, ny);
        return;

    case 4:
        fvec_L2sqr_ny_D4(dis, x, y, ny);
        return;

    case 8:
        fvec_L2sqr_ny_D8(dis, x, y, ny);
        return;

    case 12:
        fvec_L2sqr_ny_D12(dis, x, y, ny);
        return;

    default:
        fvec_L2sqr_ny_ref(dis, x, y, d, ny);
        return;
    }
}

/* SSE-implementation of L2 distance */
float fvec_L2sqr(const float* x,
                 const float* y,
                 size_t d) {
    __m128 msum1 = _mm_setzero_ps();

    while (d >= 4) {
        __m128 mx = _mm_loadu_ps(x);
        x += 4;
        __m128 my = _mm_loadu_ps(y);
        y += 4;
        const __m128 a_m_b1 = mx - my;
        msum1 += a_m_b1 * a_m_b1;
        d -= 4;
    }

    if (d > 0) {
        // add the last 1, 2 or 3 values
        __m128 mx = masked_read(d, x);
        __m128 my = masked_read(d, y);
        __m128 a_m_b1 = mx - my;
        msum1 += a_m_b1 * a_m_b1;
    }

    msum1 = _mm_hadd_ps(msum1, msum1);
    msum1 = _mm_hadd_ps(msum1, msum1);
    return  _mm_cvtss_f32(msum1);
}

#else
// scalar implementation

float fvec_L2sqr(const float* x,
                 const float* y,
                 size_t d) {
    return fvec_L2sqr_ref(x, y, d);
}

void fvec_L2sqr_ny(float* dis, const float* x,
                   const float* y, size_t d, size_t ny) {
    fvec_L2sqr_ny_ref(dis, x, y, d, ny);
}

#endif

} // namespace faiss
