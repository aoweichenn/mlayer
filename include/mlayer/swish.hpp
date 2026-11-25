// Copyright 2020 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_SWISH_HPP
#define MLAYER_SWISH_HPP

#include <cmath>
#include <cstdint>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace mlayer {

#if defined(__AVX2__)
inline __m256 swish_exp_ps(__m256 x) {
    x = _mm256_min_ps(x, _mm256_set1_ps(88.0F));
    x = _mm256_max_ps(x, _mm256_set1_ps(-88.0F));

    const __m256 log2e = _mm256_set1_ps(1.4426950408889634F);
    const __m256 ln2 = _mm256_set1_ps(0.6931471805599453F);
    const __m256 fx = _mm256_mul_ps(x, log2e);
    const __m256 n = _mm256_round_ps(fx, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m256 r = _mm256_sub_ps(x, _mm256_mul_ps(n, ln2));

    // exp(r) for |r| <= ln2/2
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
#endif

template<typename LayerT, typename MatT, typename OptionT>
class Swish : public LayerT {
public:
    Swish() {
        this->one_blob_only = true;
        this->support_inplace = true;
    }

    std::int32_t forward_inplace(MatT& bottom_top_blob, const OptionT& opt) const override {
        const std::int32_t w = bottom_top_blob.w;
        const std::int32_t h = bottom_top_blob.h;
        const std::int32_t d = bottom_top_blob.d;
        const std::int32_t channels = bottom_top_blob.c;
        const std::int32_t size = w * h * d;

#pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t q = 0; q < channels; ++q) {
            auto* ptr = static_cast<float*>(bottom_top_blob.channel(q));
#if defined(__AVX2__)
            const __m256 one = _mm256_set1_ps(1.0F);
            std::int32_t i = 0;
            for (; i + 8 <= size; i += 8) {
                const __m256 x = _mm256_loadu_ps(ptr + i);
                const __m256 denom = _mm256_add_ps(one, swish_exp_ps(_mm256_sub_ps(_mm256_setzero_ps(), x)));
                _mm256_storeu_ps(ptr + i, _mm256_div_ps(x, denom));
            }
            for (; i < size; ++i) {
                const float x = ptr[i];
                ptr[i] = x / (1.0F + std::exp(-x));
            }
#else
            for (std::int32_t i = 0; i < size; ++i) {
                const float x = ptr[i];
                ptr[i] = x / (1.0F + std::exp(-x));
            }
#endif
        }

        return 0;
    }

    // mnet-only fused activation id used by the graph fusion pass; ncnn keeps
    // SiLU as a standalone layer (fused_activation.h has no swish case).
    std::int32_t get_fused_activation(std::int32_t& type, MatT& params) const override {
        type = 7;
        params = MatT();
        return 0;
    }
};

}  // namespace mlayer

#endif  // MLAYER_SWISH_HPP
