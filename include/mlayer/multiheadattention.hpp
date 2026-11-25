// Copyright 2021 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_MULTIHEADATTENTION_HPP
#define MLAYER_MULTIHEADATTENTION_HPP

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace mlayer {

// Full multi-head attention with optional attention mask. The KV-cache and
// quantized paths are not ported yet; they return -1 instead of producing
// silently wrong results.
template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, typename ModelBinT,
         bool Validate = false>
class MultiHeadAttention : public LayerT {
public:
    MultiHeadAttention() {
        this->one_blob_only = false;
        this->support_inplace = false;
    }

    std::int32_t load_param(const ParamDictT& pd) override {
        this->embed_dim = pd.get(0, 0);
        this->num_heads = pd.get(1, 1);
        this->weight_data_size = pd.get(2, 0);
        this->kdim = pd.get(3, this->embed_dim);
        this->vdim = pd.get(4, this->embed_dim);
        this->attn_mask = pd.get(5, 0);
        this->kv_cache = pd.get(7, 0);
        this->quantize_term = pd.get(18, 0);
        this->scale = pd.get(6, 1.0F / std::sqrt(static_cast<float>(this->embed_dim / this->num_heads)));

        if constexpr (Validate) {
            if (this->num_heads <= 0 || this->embed_dim <= 0 || this->embed_dim % this->num_heads != 0 ||
                this->weight_data_size <= 0 || this->weight_data_size % this->embed_dim != 0 || this->kdim <= 0 ||
                this->vdim <= 0) {
                return -1;
            }
        }
        return 0;
    }

    std::int32_t load_model(const ModelBinT& mb) override {
        const std::int32_t qdim = this->weight_data_size / this->embed_dim;

        this->q_weight_data = mb.load(this->embed_dim * qdim, 0);
        this->q_bias_data = mb.load(this->embed_dim, 1);
        this->k_weight_data = mb.load(this->embed_dim * this->kdim, 0);
        this->k_bias_data = mb.load(this->embed_dim, 1);
        this->v_weight_data = mb.load(this->embed_dim * this->vdim, 0);
        this->v_bias_data = mb.load(this->embed_dim, 1);
        this->out_weight_data = mb.load(qdim * this->embed_dim, 0);
        this->out_bias_data = mb.load(qdim, 1);

        if (this->q_weight_data.empty() || this->q_bias_data.empty() || this->k_weight_data.empty() ||
            this->k_bias_data.empty() || this->v_weight_data.empty() || this->v_bias_data.empty() ||
            this->out_weight_data.empty() || this->out_bias_data.empty()) {
            return -100;
        }
        return 0;
    }

    std::int32_t forward(const std::vector<MatT>& bottom_blobs, std::vector<MatT>& top_blobs,
                         const OptionT& opt) const override {
        if (this->quantize_term != 0 || top_blobs.empty()) {
            return -1;
        }
        if (this->kv_cache != 0 && top_blobs.size() < 3) {
            return -1;
        }

        std::int32_t q_blob_i = 0;
        std::int32_t k_blob_i = 0;
        std::int32_t v_blob_i = 0;
        std::int32_t attn_mask_i = 0;
        std::int32_t past_k_i = 0;
        std::int32_t past_v_i = 0;
        if (this->kv_cache != 0) {
            if (this->attn_mask != 0) {
                if (bottom_blobs.size() == 4) {
                    q_blob_i = k_blob_i = v_blob_i = 0;
                    attn_mask_i = 1;
                    past_k_i = 2;
                    past_v_i = 3;
                } else if (bottom_blobs.size() == 5) {
                    q_blob_i = 0;
                    k_blob_i = v_blob_i = 1;
                    attn_mask_i = 2;
                    past_k_i = 3;
                    past_v_i = 4;
                } else if (bottom_blobs.size() == 6) {
                    q_blob_i = 0;
                    k_blob_i = 1;
                    v_blob_i = 2;
                    attn_mask_i = 3;
                    past_k_i = 4;
                    past_v_i = 5;
                } else {
                    return -1;
                }
            } else {
                if (bottom_blobs.size() == 3) {
                    q_blob_i = k_blob_i = v_blob_i = 0;
                    past_k_i = 1;
                    past_v_i = 2;
                } else if (bottom_blobs.size() == 4) {
                    q_blob_i = 0;
                    k_blob_i = v_blob_i = 1;
                    past_k_i = 2;
                    past_v_i = 3;
                } else if (bottom_blobs.size() == 5) {
                    q_blob_i = 0;
                    k_blob_i = 1;
                    v_blob_i = 2;
                    past_k_i = 3;
                    past_v_i = 4;
                } else {
                    return -1;
                }
            }
        } else if (this->attn_mask != 0) {
            if (bottom_blobs.size() == 2) {
                q_blob_i = k_blob_i = v_blob_i = 0;
                attn_mask_i = 1;
            } else if (bottom_blobs.size() == 3) {
                q_blob_i = 0;
                k_blob_i = v_blob_i = 1;
                attn_mask_i = 2;
            } else if (bottom_blobs.size() == 4) {
                q_blob_i = 0;
                k_blob_i = 1;
                v_blob_i = 2;
                attn_mask_i = 3;
            } else {
                return -1;
            }
        } else {
            if (bottom_blobs.size() == 1) {
                q_blob_i = k_blob_i = v_blob_i = 0;
            } else if (bottom_blobs.size() == 2) {
                q_blob_i = 0;
                k_blob_i = v_blob_i = 1;
            } else if (bottom_blobs.size() == 3) {
                q_blob_i = 0;
                k_blob_i = 1;
                v_blob_i = 2;
            } else {
                return -1;
            }
        }

        const MatT& q_blob = bottom_blobs[static_cast<std::size_t>(q_blob_i)];
        const MatT& k_blob = bottom_blobs[static_cast<std::size_t>(k_blob_i)];
        const MatT& v_blob = bottom_blobs[static_cast<std::size_t>(v_blob_i)];
        const MatT& past_k = this->kv_cache != 0 ? bottom_blobs[static_cast<std::size_t>(past_k_i)] : MatT();
        const MatT& past_v = this->kv_cache != 0 ? bottom_blobs[static_cast<std::size_t>(past_v_i)] : MatT();

        const std::int32_t src_seqlen = q_blob.h;
        const std::int32_t cur_seqlen = k_blob.h;
        const std::int32_t past_seqlen = !past_k.empty() ? past_k.h : 0;
        const std::int32_t dst_seqlen = past_seqlen > 0
                                            ? (q_blob_i == k_blob_i ? past_seqlen + cur_seqlen : past_seqlen)
                                            : cur_seqlen;
        const bool append_kv = past_seqlen == 0 || q_blob_i == k_blob_i;
        const std::int32_t embed_dim_per_head = this->embed_dim / this->num_heads;
        const std::int32_t qdim = this->weight_data_size / this->embed_dim;

        MatT q_affine;
        q_affine.create(src_seqlen, this->embed_dim, 4u, opt.workspace_allocator);
        if (q_affine.empty()) {
            return -100;
        }
        #pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t i = 0; i < src_seqlen; ++i) {
            const float* kptr = static_cast<const float*>(this->q_weight_data);
            for (std::int32_t j = 0; j < this->embed_dim; ++j) {
                const float* ptr = q_blob.row(i);
                float sum = this->q_bias_data[j];
                for (std::int32_t k = 0; k < qdim; ++k) {
                    sum += *ptr++ * *kptr++;
                }
                q_affine.row(j)[i] = sum * this->scale;
            }
        }

        MatT k_affine;
        MatT v_affine;
        if (append_kv) {
            k_affine.create(cur_seqlen, this->embed_dim, 4u, opt.workspace_allocator);
            v_affine.create(cur_seqlen, this->embed_dim, 4u, opt.workspace_allocator);
            if (k_affine.empty() || v_affine.empty()) {
                return -100;
            }
            #pragma omp parallel for num_threads(opt.num_threads)
            for (std::int32_t i = 0; i < cur_seqlen; ++i) {
                const float* kptr = static_cast<const float*>(this->k_weight_data);
                for (std::int32_t j = 0; j < this->embed_dim; ++j) {
                    const float* ptr = k_blob.row(i);
                    float sum = this->k_bias_data[j];
                    for (std::int32_t k = 0; k < this->kdim; ++k) {
                        sum += *ptr++ * *kptr++;
                    }
                    k_affine.row(j)[i] = sum;
                }
            }
            #pragma omp parallel for num_threads(opt.num_threads)
            for (std::int32_t i = 0; i < cur_seqlen; ++i) {
                const float* kptr = static_cast<const float*>(this->v_weight_data);
                for (std::int32_t j = 0; j < this->embed_dim; ++j) {
                    const float* ptr = v_blob.row(i);
                    float sum = this->v_bias_data[j];
                    for (std::int32_t k = 0; k < this->vdim; ++k) {
                        sum += *ptr++ * *kptr++;
                    }
                    v_affine.row(j)[i] = sum;
                }
            }
        }

        MatT empty_cache;
        MatT& cached_k = this->kv_cache != 0 ? top_blobs[1] : empty_cache;
        MatT& cached_v = this->kv_cache != 0 ? top_blobs[2] : empty_cache;
        if (this->kv_cache != 0) {
            cached_k.create(embed_dim_per_head, dst_seqlen, this->num_heads, 4u, opt.blob_allocator);
            cached_v.create(embed_dim_per_head, dst_seqlen, this->num_heads, 4u, opt.blob_allocator);
            if (cached_k.empty() || cached_v.empty()) {
                return -100;
            }
            for (std::int32_t q = 0; q < this->num_heads; ++q) {
                if (past_seqlen > 0) {
                    for (std::int32_t y = 0; y < past_seqlen; ++y) {
                        std::memcpy(cached_k.channel(q).row(y), past_k.channel(q).row(y),
                                    static_cast<std::size_t>(embed_dim_per_head) * sizeof(float));
                        std::memcpy(cached_v.channel(q).row(y), past_v.channel(q).row(y),
                                    static_cast<std::size_t>(embed_dim_per_head) * sizeof(float));
                    }
                }
                if (append_kv) {
                    for (std::int32_t s = 0; s < cur_seqlen; ++s) {
                        float* kptr = cached_k.channel(q).row(past_seqlen + s);
                        float* vptr = cached_v.channel(q).row(past_seqlen + s);
                        for (std::int32_t d = 0; d < embed_dim_per_head; ++d) {
                            kptr[d] = k_affine.row(q * embed_dim_per_head + d)[s];
                            vptr[d] = v_affine.row(q * embed_dim_per_head + d)[s];
                        }
                    }
                }
            }
        }

        MatT qk_cross;
        qk_cross.create(dst_seqlen, src_seqlen, this->num_heads, 4u, opt.workspace_allocator);
        if (qk_cross.empty()) {
            return -100;
        }
        #pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t q = 0; q < this->num_heads; ++q) {
            const MatT q_affine_head = q_affine.row_range(q * embed_dim_per_head, embed_dim_per_head);
            MatT qk_cross_head = qk_cross.channel(q);
            for (std::int32_t i = 0; i < src_seqlen; ++i) {
                float* outptr = qk_cross_head.row(i);
                for (std::int32_t j = 0; j < dst_seqlen; ++j) {
                    float sum = 0.0F;
                    if (this->kv_cache != 0) {
                        const MatT k_head = cached_k.channel(q);
                        for (std::int32_t l = 0; l < embed_dim_per_head; ++l) {
                            sum += q_affine_head.row(l)[i] * k_head.row(j)[l];
                        }
                    } else {
                        const MatT k_affine_head =
                            k_affine.row_range(q * embed_dim_per_head, embed_dim_per_head);
                        for (std::int32_t l = 0; l < embed_dim_per_head; ++l) {
                            sum += q_affine_head.row(l)[i] * k_affine_head.row(l)[j];
                        }
                    }
                    outptr[j] = sum;
                }
            }
        }

        if (this->attn_mask != 0) {
            const MatT& mask_blob = bottom_blobs[static_cast<std::size_t>(attn_mask_i)];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (std::int32_t q = 0; q < this->num_heads; ++q) {
                const std::int32_t mask_channel = mask_blob.c == 1 ? 0 : q;
                const MatT maskm = mask_blob.dims == 3 ? mask_blob.channel(mask_channel) : mask_blob;
                MatT qk_cross_head = qk_cross.channel(q);
                for (std::int32_t i = 0; i < src_seqlen; ++i) {
                    const float* mptr = maskm.row(i);
                    float* outptr = qk_cross_head.row(i);
                    for (std::int32_t j = 0; j < dst_seqlen; ++j) {
                        outptr[j] += mptr[j];
                    }
                }
            }
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t q = 0; q < this->num_heads; ++q) {
            MatT qk_cross_head = qk_cross.channel(q);
            for (std::int32_t i = 0; i < src_seqlen; ++i) {
                float* ptr = qk_cross_head.row(i);
                float max = -FLT_MAX;
                for (std::int32_t j = 0; j < dst_seqlen; ++j) {
                    max = std::max(max, ptr[j]);
                }
                float sum = 0.0F;
                for (std::int32_t j = 0; j < dst_seqlen; ++j) {
                    ptr[j] = std::exp(ptr[j] - max);
                    sum += ptr[j];
                }
                for (std::int32_t j = 0; j < dst_seqlen; ++j) {
                    ptr[j] /= sum;
                }
            }
        }

        MatT qkv_cross;
        qkv_cross.create(src_seqlen, this->embed_dim, 4u, opt.workspace_allocator);
        if (qkv_cross.empty()) {
            return -100;
        }
        #pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t q = 0; q < this->num_heads; ++q) {
            const MatT qk_cross_head = qk_cross.channel(q);
            MatT qkv_cross_head = qkv_cross.row_range(q * embed_dim_per_head, embed_dim_per_head);
            const MatT v_affine_head =
                this->kv_cache != 0 ? cached_v.channel(q)
                                    : v_affine.row_range(q * embed_dim_per_head, embed_dim_per_head);
            for (std::int32_t i = 0; i < src_seqlen; ++i) {
                for (std::int32_t j = 0; j < embed_dim_per_head; ++j) {
                    const float* qkptr = qk_cross_head.row(i);
                    float sum = 0.0F;
                    if (this->kv_cache != 0) {
                        for (std::int32_t k = 0; k < dst_seqlen; ++k) {
                            sum += *qkptr++ * v_affine_head.row(k)[j];
                        }
                    } else {
                        for (std::int32_t k = 0; k < dst_seqlen; ++k) {
                            sum += *qkptr++ * v_affine_head.row(j)[k];
                        }
                    }
                    qkv_cross_head.row(j)[i] = sum;
                }
            }
        }

        MatT& top_blob = top_blobs[0];
        top_blob.create(qdim, src_seqlen, 4u, opt.blob_allocator);
        if (top_blob.empty()) {
            return -100;
        }
        #pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t i = 0; i < src_seqlen; ++i) {
            float* outptr = top_blob.row(i);
            for (std::int32_t j = 0; j < qdim; ++j) {
                const float* kptr = static_cast<const float*>(this->out_weight_data) + j * this->embed_dim;
                float sum = this->out_bias_data[j];
                for (std::int32_t k = 0; k < this->embed_dim; ++k) {
                    sum += qkv_cross.row(k)[i] * kptr[k];
                }
                outptr[j] = sum;
            }
        }

        return 0;
    }

public:
    std::int32_t embed_dim = 0;
    std::int32_t num_heads = 1;
    std::int32_t weight_data_size = 0;
    std::int32_t kdim = 0;
    std::int32_t vdim = 0;
    std::int32_t attn_mask = 0;
    float scale = 0.0F;
    std::int32_t kv_cache = 0;
    std::int32_t quantize_term = 0;

    MatT q_weight_data;
    MatT q_bias_data;
    MatT k_weight_data;
    MatT k_bias_data;
    MatT v_weight_data;
    MatT v_bias_data;
    MatT out_weight_data;
    MatT out_bias_data;
};

}  // namespace mlayer

#endif  // MLAYER_MULTIHEADATTENTION_HPP
