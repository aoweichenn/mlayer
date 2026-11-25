// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_VECMATH_HPP
#define MLAYER_VECMATH_HPP

#if defined(__AVX2__)
#include <immintrin.h>

namespace mlayer {

// exp(x) for |x| <= 88 with a degree-5 polynomial after range reduction.
// Same approximation the Swish layer uses, shared with the convolution
// activation epilogue.
inline __m256 vec_exp_ps(__m256 x) {
    x = _mm256_min_ps(x, _mm256_set1_ps(88.0F));
    x = _mm256_max_ps(x, _mm256_set1_ps(-88.0F));

    const __m256 log2e = _mm256_set1_ps(1.4426950408889634F);
    const __m256 ln2 = _mm256_set1_ps(0.6931471805599453F);
    const __m256 fx = _mm256_mul_ps(x, log2e);
    const __m256 n = _mm256_round_ps(fx, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m256 r = _mm256_sub_ps(x, _mm256_mul_ps(n, ln2));

    __m256 p = _mm256_set1_ps(1.0F / 120.0F);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0F / 24.0F));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0F / 6.0F));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.5F));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0F));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0F));

    const __m256i ni = _mm256_cvtps_epi32(n);
    const __m256i bits = _mm256_slli_epi32(_mm256_add_epi32(ni, _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(bits));
}

// SiLU (swish) used by the fused convolution epilogue (mnet-only id 7).
inline __m256 vec_silu_ps(__m256 x) {
    const __m256 one = _mm256_set1_ps(1.0F);
    const __m256 denom = _mm256_add_ps(one, vec_exp_ps(_mm256_sub_ps(_mm256_setzero_ps(), x)));
    return _mm256_div_ps(x, denom);
}

}  // namespace mlayer

#endif  // __AVX2__

#endif  // MLAYER_VECMATH_HPP
