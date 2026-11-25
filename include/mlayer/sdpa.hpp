// Copyright 2025 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_SDPA_HPP
#define MLAYER_SDPA_HPP

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include <omp.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#include <mlayer/int8_quantize.hpp>

namespace mlayer {

namespace detail {

// Whole-2D-tensor dynamic quantization (ncnn SDPA int8 key/value path).
template<typename MatT>
inline void sdpa_quantize_2d(const MatT& blob, MatT& blob_int8, float& scale) {
    float absmax = 0.0F;
    for (std::int32_t i = 0; i < blob_int8.h; ++i) {
        const float* ptr = blob.row(i);
        for (std::int32_t j = 0; j < blob_int8.w; ++j) {
            absmax = std::max(absmax, std::fabs(ptr[j]));
        }
    }
    scale = absmax == 0.0F ? 1.0F : 127.0F / absmax;
    for (std::int32_t i = 0; i < blob_int8.h; ++i) {
        const float* ptr = blob.row(i);
        auto* outptr = blob_int8.template row<std::int8_t>(i);
        for (std::int32_t j = 0; j < blob_int8.w; ++j) {
            outptr[j] = quantize_int8_scalar(ptr[j], scale);
        }
    }
}

// Per-row dynamic quantization (ncnn SDPA int8 query / attention path).
template<typename MatT>
inline void sdpa_quantize_2d_per_h(const MatT& blob, MatT& blob_int8, MatT& scales) {
    for (std::int32_t i = 0; i < blob_int8.h; ++i) {
        const float* ptr = blob.row(i);
        float absmax = 0.0F;
        for (std::int32_t j = 0; j < blob_int8.w; ++j) {
            absmax = std::max(absmax, std::fabs(ptr[j]));
        }
        scales[i] = absmax == 0.0F ? 1.0F : 127.0F / absmax;
    }
    for (std::int32_t i = 0; i < blob_int8.h; ++i) {
        const float* ptr = blob.row(i);
        auto* outptr = blob_int8.template row<std::int8_t>(i);
        const float scale = scales[i];
        for (std::int32_t j = 0; j < blob_int8.w; ++j) {
            outptr[j] = quantize_int8_scalar(ptr[j], scale);
        }
    }
}

}  // namespace detail

template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT>
class SDPA : public LayerT {
public:
    SDPA() {
        this->one_blob_only = false;
        this->support_inplace = false;
        this->attn_mask = 0;
        this->scale = 0.0F;
        this->kv_cache = 0;
        this->int8_scale_term = 0;
    }

    std::int32_t load_param(const ParamDictT& pd) override {
        this->attn_mask = pd.get(5, 0);
        this->scale = pd.get(6, 0.0F);
        this->kv_cache = pd.get(7, 0);
        this->int8_scale_term = pd.get(18, 0);
        return 0;
    }

    std::int32_t forward(const std::vector<MatT>& bottom_blobs, std::vector<MatT>& top_blobs,
                         const OptionT& opt) const override {
        if (this->int8_scale_term != 0) {
            return this->forward_int8(bottom_blobs, top_blobs, opt);
        }

        const MatT& query = bottom_blobs[0];
        const MatT& cur_key = bottom_blobs[1];
        const MatT& cur_value = bottom_blobs[2];
        const MatT& attn_mask_blob = this->attn_mask != 0 ? bottom_blobs[3] : MatT();
        const MatT& past_key = this->kv_cache != 0 ? bottom_blobs[this->attn_mask != 0 ? 4 : 3] : MatT();
        const MatT& past_value = this->kv_cache != 0 ? bottom_blobs[this->attn_mask != 0 ? 5 : 4] : MatT();

        const std::int32_t embed_dim = query.w;
        const std::int32_t src_seqlen = query.h;
        const std::int32_t num_heads = query.c;
        const std::int32_t cur_seqlen = cur_key.h;
        const std::int32_t num_group = cur_key.c;
        const std::int32_t out_embed_dim = cur_value.w;
        const std::int32_t past_seqlen = this->kv_cache != 0 ? past_key.h : 0;
        const std::int32_t dst_seqlen = past_seqlen + cur_seqlen;

        const float scale_value = this->scale == 0.0F ? 1.0F / std::sqrt(static_cast<float>(embed_dim))
                                                      : this->scale;
        const std::int32_t num_heads_per_group = num_heads / num_group;

        MatT& top_blob = top_blobs[0];
        top_blob.create(out_embed_dim, src_seqlen, num_heads, 4u, opt.blob_allocator);
        if (top_blob.empty()) {
            return -100;
        }

        MatT qk_cross(dst_seqlen, src_seqlen, opt.num_threads, 4u, opt.workspace_allocator);
        if (qk_cross.empty()) {
            return -100;
        }

        MatT key = cur_key;
        MatT value = cur_value;
        if (this->kv_cache != 0) {
            MatT& cached_key = top_blobs[1];
            MatT& cached_value = top_blobs[2];

            const std::int32_t retk = this->create_or_grow_kvcache(
                past_key, cached_key, dst_seqlen, num_group, embed_dim, cur_key.elemsize, cur_key.elempack, opt);
            if (retk != 0) {
                return retk;
            }
            const std::int32_t retv = this->create_or_grow_kvcache(
                past_value, cached_value, dst_seqlen, num_group, out_embed_dim, cur_value.elemsize,
                cur_value.elempack, opt);
            if (retv != 0) {
                return retv;
            }

#pragma omp parallel for num_threads(opt.num_threads)
            for (std::int32_t q = 0; q < num_group; ++q) {
                MatT key_head = cached_key.channel(q);
                MatT value_head = cached_value.channel(q);
                std::memcpy(key_head.row(past_seqlen), cur_key.channel(q),
                            static_cast<std::size_t>(embed_dim) * static_cast<std::size_t>(cur_seqlen) *
                                cur_key.elemsize);
                std::memcpy(value_head.row(past_seqlen), cur_value.channel(q),
                            static_cast<std::size_t>(out_embed_dim) * static_cast<std::size_t>(cur_seqlen) *
                                cur_value.elemsize);
            }

            key = cached_key;
            value = cached_value;
        }

#pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t q = 0; q < num_heads; ++q) {
            const MatT query_head = query.channel(q);
            const MatT key_head = key.channel(q / num_heads_per_group);
            const MatT value_head = value.channel(q / num_heads_per_group);
            MatT qk_cross_head = qk_cross.channel(static_cast<std::int32_t>(omp_get_thread_num()));
            MatT top_blob_head = top_blob.channel(q);

            for (std::int32_t i = 0; i < src_seqlen; ++i) {
                const float* qptr = query_head.row(i);
                float* outptr = qk_cross_head.row(i);
                for (std::int32_t j = 0; j < dst_seqlen; ++j) {
                    const float* kptr = key_head.row(j);
                    float sum = 0.0F;
#if defined(__AVX2__)
                    __m256 acc = _mm256_setzero_ps();
                    std::int32_t k = 0;
                    for (; k + 8 <= embed_dim; k += 8) {
                        acc = _mm256_fmadd_ps(_mm256_loadu_ps(qptr + k), _mm256_loadu_ps(kptr + k), acc);
                    }
                    __m128 lo = _mm256_castps256_ps128(acc);
                    __m128 hi = _mm256_extractf128_ps(acc, 1);
                    __m128 s4 = _mm_add_ps(lo, hi);
                    s4 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
                    s4 = _mm_add_ss(s4, _mm_shuffle_ps(s4, s4, 1));
                    sum = _mm_cvtss_f32(s4);
                    for (; k < embed_dim; ++k) {
                        sum += qptr[k] * kptr[k];
                    }
#else
                    for (std::int32_t k = 0; k < embed_dim; ++k) {
                        sum += qptr[k] * kptr[k];
                    }
#endif
                    outptr[j] = sum * scale_value;
                }
            }

            if (this->attn_mask != 0) {
                const MatT& maskm = attn_mask_blob.c > 1 ? attn_mask_blob.channel(q) : attn_mask_blob;
                for (std::int32_t i = 0; i < src_seqlen; ++i) {
                    const float* mptr = maskm.row(i);
                    float* outptr = qk_cross_head.row(i);
                    for (std::int32_t j = 0; j < dst_seqlen; ++j) {
                        outptr[j] += mptr[j];
                    }
                }
            }

            for (std::int32_t i = 0; i < src_seqlen; ++i) {
                float* ptr = qk_cross_head.row(i);
                float maxv = -FLT_MAX;
                for (std::int32_t j = 0; j < dst_seqlen; ++j) {
                    maxv = std::max(maxv, ptr[j]);
                }
                float sum = 0.0F;
                for (std::int32_t j = 0; j < dst_seqlen; ++j) {
                    ptr[j] = std::exp(ptr[j] - maxv);
                    sum += ptr[j];
                }
                for (std::int32_t j = 0; j < dst_seqlen; ++j) {
                    ptr[j] /= sum;
                }
            }

            const float* value_base = value_head.row(0);
            const std::size_t value_row_stride = static_cast<std::size_t>(value_head.w);
            for (std::int32_t i = 0; i < src_seqlen; ++i) {
                const float* qkptr = qk_cross_head.row(i);
                float* outptr = top_blob_head.row(i);
                std::int32_t j = 0;
#if defined(__AVX2__)
                for (; j + 8 <= out_embed_dim; j += 8) {
                    __m256 acc = _mm256_setzero_ps();
                    const float* vptr = value_base + j;
                    for (std::int32_t k = 0; k < dst_seqlen; ++k) {
                        acc = _mm256_fmadd_ps(_mm256_set1_ps(qkptr[k]), _mm256_loadu_ps(vptr), acc);
                        vptr += value_row_stride;
                    }
                    _mm256_storeu_ps(outptr + j, acc);
                }
#endif
                for (; j < out_embed_dim; ++j) {
                    float sum = 0.0F;
                    for (std::int32_t k = 0; k < dst_seqlen; ++k) {
                        sum += qkptr[k] * value_head.row(k)[j];
                    }
                    outptr[j] = sum;
                }
            }
        }

        return 0;
    }

public:
    // ncnn SDPA int8: dynamically quantize query (per row), key/value (whole
    // tensor) per head, use int32 dot products, then dequantize.
    std::int32_t forward_int8(const std::vector<MatT>& bottom_blobs, std::vector<MatT>& top_blobs,
                              const OptionT& opt) const {
        const MatT& query = bottom_blobs[0];
        const MatT& cur_key = bottom_blobs[1];
        const MatT& cur_value = bottom_blobs[2];
        const MatT& attn_mask_blob = this->attn_mask != 0 ? bottom_blobs[3] : MatT();
        const MatT& past_key = this->kv_cache != 0 ? bottom_blobs[this->attn_mask != 0 ? 4 : 3] : MatT();
        const MatT& past_value = this->kv_cache != 0 ? bottom_blobs[this->attn_mask != 0 ? 5 : 4] : MatT();

        const std::int32_t embed_dim = query.w;
        const std::int32_t src_seqlen = query.h;
        const std::int32_t num_heads = query.c;
        const std::int32_t cur_seqlen = cur_key.h;
        const std::int32_t num_group = cur_key.c;
        const std::int32_t out_embed_dim = cur_value.w;
        const std::int32_t past_seqlen = this->kv_cache != 0 ? past_key.h : 0;
        const std::int32_t dst_seqlen = past_seqlen + cur_seqlen;

        const float scale_value = this->scale == 0.0F ? 1.0F / std::sqrt(static_cast<float>(embed_dim))
                                                      : this->scale;
        const std::int32_t num_heads_per_group = num_heads / num_group;

        MatT& top_blob = top_blobs[0];
        top_blob.create(out_embed_dim, src_seqlen, num_heads, 4u, opt.blob_allocator);
        MatT query_int8(embed_dim, src_seqlen, opt.num_threads, 1u, opt.workspace_allocator);
        MatT key_int8(embed_dim, dst_seqlen, opt.num_threads, 1u, opt.workspace_allocator);
        MatT value_int8(out_embed_dim, dst_seqlen, opt.num_threads, 1u, opt.workspace_allocator);
        MatT qk_cross(dst_seqlen, src_seqlen, opt.num_threads, 4u, opt.workspace_allocator);
        MatT qk_cross_int8(dst_seqlen, src_seqlen, opt.num_threads, 1u, opt.workspace_allocator);
        MatT qk_scales(src_seqlen, 1, opt.num_threads, 4u, opt.workspace_allocator);
        if (top_blob.empty() || query_int8.empty() || key_int8.empty() || value_int8.empty() ||
            qk_cross.empty() || qk_cross_int8.empty() || qk_scales.empty()) {
            return -100;
        }

        MatT key = cur_key;
        MatT value = cur_value;
        if (this->kv_cache != 0) {
            MatT& cached_key = top_blobs[1];
            MatT& cached_value = top_blobs[2];
            const std::int32_t retk = this->create_or_grow_kvcache(
                past_key, cached_key, dst_seqlen, num_group, embed_dim, cur_key.elemsize, cur_key.elempack, opt);
            if (retk != 0) {
                return retk;
            }
            const std::int32_t retv = this->create_or_grow_kvcache(
                past_value, cached_value, dst_seqlen, num_group, out_embed_dim, cur_value.elemsize,
                cur_value.elempack, opt);
            if (retv != 0) {
                return retv;
            }
#pragma omp parallel for num_threads(opt.num_threads)
            for (std::int32_t q = 0; q < num_group; ++q) {
                MatT key_head = cached_key.channel(q);
                MatT value_head = cached_value.channel(q);
                std::memcpy(key_head.row(past_seqlen), cur_key.channel(q),
                            static_cast<std::size_t>(embed_dim) * static_cast<std::size_t>(cur_seqlen) *
                                cur_key.elemsize);
                std::memcpy(value_head.row(past_seqlen), cur_value.channel(q),
                            static_cast<std::size_t>(out_embed_dim) * static_cast<std::size_t>(cur_seqlen) *
                                cur_value.elemsize);
            }
            key = cached_key;
            value = cached_value;
        }

#pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t q = 0; q < num_heads; ++q) {
            const std::int32_t thread_index = static_cast<std::int32_t>(omp_get_thread_num());
            const MatT query_head = query.channel(q);
            const MatT key_head = key.channel(q / num_heads_per_group);
            const MatT value_head = value.channel(q / num_heads_per_group);
            MatT qk_cross_head = qk_cross.channel(thread_index);
            MatT top_blob_head = top_blob.channel(q);

            {
                MatT query_head_int8 = query_int8.channel(thread_index);
                MatT query_head_scales = qk_scales.channel(thread_index);
                detail::sdpa_quantize_2d_per_h(query_head, query_head_int8, query_head_scales);

                MatT key_head_int8 = key_int8.channel(thread_index);
                float key_head_scale = 1.0F;
                detail::sdpa_quantize_2d(key_head, key_head_int8, key_head_scale);

                for (std::int32_t i = 0; i < src_seqlen; ++i) {
                    const auto* qptr = query_head_int8.template row<std::int8_t>(i);
                    float* outptr = qk_cross_head.row(i);
                    const float qk_descale = 1.0F / (query_head_scales[i] * key_head_scale);
                    for (std::int32_t j = 0; j < dst_seqlen; ++j) {
                        const auto* kptr = key_head_int8.template row<std::int8_t>(j);
                        int sum = 0;
                        for (std::int32_t k = 0; k < embed_dim; ++k) {
                            sum += static_cast<int>(qptr[k]) * static_cast<int>(kptr[k]);
                        }
                        outptr[j] = static_cast<float>(sum) * qk_descale * scale_value;
                    }
                }
            }

            if (this->attn_mask != 0) {
                const MatT& maskm = attn_mask_blob.c > 1 ? attn_mask_blob.channel(q) : attn_mask_blob;
                for (std::int32_t i = 0; i < src_seqlen; ++i) {
                    const float* mptr = maskm.row(i);
                    float* outptr = qk_cross_head.row(i);
                    for (std::int32_t j = 0; j < dst_seqlen; ++j) {
                        outptr[j] += mptr[j];
                    }
                }
            }

            for (std::int32_t i = 0; i < src_seqlen; ++i) {
                float* ptr = qk_cross_head.row(i);
                float maxv = -FLT_MAX;
                for (std::int32_t j = 0; j < dst_seqlen; ++j) {
                    maxv = std::max(maxv, ptr[j]);
                }
                float sum = 0.0F;
                for (std::int32_t j = 0; j < dst_seqlen; ++j) {
                    ptr[j] = std::exp(ptr[j] - maxv);
                    sum += ptr[j];
                }
                for (std::int32_t j = 0; j < dst_seqlen; ++j) {
                    ptr[j] /= sum;
                }
            }

            {
                MatT qk_head_int8 = qk_cross_int8.channel(thread_index);
                MatT qk_head_scales = qk_scales.channel(thread_index);
                detail::sdpa_quantize_2d_per_h(qk_cross_head, qk_head_int8, qk_head_scales);

                MatT value_head_int8 = value_int8.channel(thread_index);
                float value_head_scale = 1.0F;
                detail::sdpa_quantize_2d(value_head, value_head_int8, value_head_scale);

                for (std::int32_t i = 0; i < src_seqlen; ++i) {
                    const auto* qkptr = qk_head_int8.template row<std::int8_t>(i);
                    float* outptr = top_blob_head.row(i);
                    const float qkv_descale = 1.0F / (qk_head_scales[i] * value_head_scale);
                    for (std::int32_t j = 0; j < out_embed_dim; ++j) {
                        int sum = 0;
                        for (std::int32_t k = 0; k < dst_seqlen; ++k) {
                            sum += static_cast<int>(qkptr[k]) *
                                   static_cast<int>(value_head_int8.template row<std::int8_t>(k)[j]);
                        }
                        outptr[j] = static_cast<float>(sum) * qkv_descale;
                    }
                }
            }
        }
        return 0;
    }

public:
    std::int32_t attn_mask;
    float scale;
    std::int32_t kv_cache;
    std::int32_t int8_scale_term;

private:
    static std::int32_t kvcache_capacity(std::int32_t current_capacity, std::int32_t new_seqlen,
                                         std::int32_t max_seqlen_hint) {
        if (current_capacity == 0 && max_seqlen_hint >= new_seqlen && max_seqlen_hint > 0) {
            return max_seqlen_hint;
        }
        const std::int32_t capacity = current_capacity > new_seqlen ? current_capacity : new_seqlen;
        std::int32_t reserve = 0;
        if (current_capacity == 0) {
            reserve = capacity < 16 ? 16 - capacity : capacity;
            if (reserve > 256) {
                reserve = 256;
            }
        } else {
            reserve = capacity / 2;
            if (reserve < 16) {
                reserve = 16;
            }
        }
        return capacity <= INT_MAX - reserve ? capacity + reserve : capacity;
    }

    std::int32_t create_or_grow_kvcache(const MatT& cache, MatT& new_cache, std::int32_t new_seqlen,
                                        std::int32_t num_kv_head, std::int32_t head_dim, std::size_t elemsize,
                                        std::int32_t elempack, const OptionT& opt) const {
        auto* allocator = opt.kvcache_allocator;
        const bool reuse = !cache.empty() && cache.allocator == allocator;
        const std::int32_t current_capacity =
            reuse ? static_cast<std::int32_t>(cache.cstep / static_cast<std::size_t>(cache.w)) : 0;
        if (reuse && new_seqlen <= current_capacity) {
            new_cache = cache;
            new_cache.h = new_seqlen;
            return 0;
        }

        const std::int32_t capacity = kvcache_capacity(current_capacity, new_seqlen, opt.kvcache_max_seqlen_hint);
        MatT m;
        m.create(head_dim, capacity, num_kv_head, elemsize, elempack, allocator);
        if (m.empty()) {
            return -100;
        }
        m.h = new_seqlen;

        if (!cache.empty()) {
            const std::size_t valid_head_size =
                static_cast<std::size_t>(cache.w) * static_cast<std::size_t>(cache.h) * cache.elemsize;
            for (std::int32_t q = 0; q < cache.c; ++q) {
                const auto* src = static_cast<const std::uint8_t*>(cache.data) +
                                  cache.cstep * static_cast<std::size_t>(q) * cache.elemsize;
                auto* dst = static_cast<std::uint8_t*>(m.data) +
                            m.cstep * static_cast<std::size_t>(q) * m.elemsize;
                std::memcpy(dst, src, valid_head_size);
            }
        }

        new_cache = m;
        return 0;
    }
};

}  // namespace mlayer

#endif  // MLAYER_SDPA_HPP
