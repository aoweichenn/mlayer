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

// Full multi-head attention with optional attention mask and KV cache.
// Weight-block-quantized weights (quantize_term >= 400) and the legacy int8
// path (quantize_term 1..3, weights stored as int8) follow ncnn's arithmetic:
// per-block weight scales with per-block activation quantization, and for the
// legacy path per-output-channel weight scales with the int8 attention math.
namespace detail {

// Bit-packed weight-block-quant format (ncnn quantize_term >= 400): each
// output row is packed little-endian in `bits` per reduction index.
inline std::int32_t mha_unpack_weight(const unsigned char* ptr, std::int32_t k, std::int32_t bits) {
    const std::int32_t bit_offset = k * bits;
    const std::int32_t byte_offset = bit_offset / 8;
    const std::int32_t bit_shift = bit_offset % 8;
    const unsigned int v = static_cast<unsigned int>(ptr[byte_offset]) |
                           (static_cast<unsigned int>(ptr[byte_offset + 1]) << 8);
    const std::int32_t mask = (1 << bits) - 1;
    const std::int32_t raw = static_cast<std::int32_t>((v >> bit_shift) & static_cast<unsigned int>(mask));
    const std::int32_t sign_bit = 1 << (bits - 1);
    return (raw ^ sign_bit) - sign_bit;
}

inline signed char mha_float2int8(float v) {
    int int32 = static_cast<int>(std::round(v));
    if (int32 > 127) {
        return 127;
    }
    if (int32 < -127) {
        return -127;
    }
    return static_cast<signed char>(int32);
}

inline std::int32_t mha_packed_k_bytes(std::int32_t k, std::int32_t bits) {
    return (k * bits + 7) / 8;
}

}  // namespace detail

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
        if (this->quantize_term != 0) {
            if (this->parse_quantize_term() != 0) {
                return -1;
            }
        }
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

        if (this->weight_block_quantize) {
            const std::int32_t bits = this->weight_bits;
            this->q_weight_data = mb.load(detail::mha_packed_k_bytes(qdim, bits), this->embed_dim, bits);
            this->q_bias_data = mb.load(this->embed_dim, 1);
            this->k_weight_data = mb.load(detail::mha_packed_k_bytes(this->kdim, bits), this->embed_dim, bits);
            this->k_bias_data = mb.load(this->embed_dim, 1);
            this->v_weight_data = mb.load(detail::mha_packed_k_bytes(this->vdim, bits), this->embed_dim, bits);
            this->v_bias_data = mb.load(this->embed_dim, 1);
            this->out_weight_data = mb.load(detail::mha_packed_k_bytes(this->embed_dim, bits), qdim, bits);
            this->out_bias_data = mb.load(qdim, 1);

            this->q_weight_data_quantize_scales =
                mb.load((qdim + this->block_size - 1) / this->block_size, this->embed_dim, 1);
            this->k_weight_data_quantize_scales =
                mb.load((this->kdim + this->block_size - 1) / this->block_size, this->embed_dim, 1);
            this->v_weight_data_quantize_scales =
                mb.load((this->vdim + this->block_size - 1) / this->block_size, this->embed_dim, 1);
            this->out_weight_data_quantize_scales =
                mb.load((this->embed_dim + this->block_size - 1) / this->block_size, qdim, 1);

            if (this->q_weight_data.empty() || this->q_bias_data.empty() || this->k_weight_data.empty() ||
                this->k_bias_data.empty() || this->v_weight_data.empty() || this->v_bias_data.empty() ||
                this->out_weight_data.empty() || this->out_bias_data.empty() ||
                this->q_weight_data_quantize_scales.empty() || this->k_weight_data_quantize_scales.empty() ||
                this->v_weight_data_quantize_scales.empty() || this->out_weight_data_quantize_scales.empty()) {
                return -100;
            }

            if (this->has_input_scale) {
                this->q_weight_data_input_scales = mb.load(qdim, 1);
                this->k_weight_data_input_scales = mb.load(this->kdim, 1);
                this->v_weight_data_input_scales = mb.load(this->vdim, 1);
                this->out_weight_data_input_scales = mb.load(this->embed_dim, 1);
                if (this->q_weight_data_input_scales.empty() || this->k_weight_data_input_scales.empty() ||
                    this->v_weight_data_input_scales.empty() || this->out_weight_data_input_scales.empty()) {
                    return -100;
                }
            }
            return 0;
        }

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

        if (this->legacy_int8) {
            this->q_weight_data_int8_scales = mb.load(this->embed_dim, 1);
            this->k_weight_data_int8_scales = mb.load(this->embed_dim, 1);
            this->v_weight_data_int8_scales = mb.load(this->embed_dim, 1);
            const MatT out_scale = mb.load(1, 1);
            if (this->q_weight_data_int8_scales.empty() || this->k_weight_data_int8_scales.empty() ||
                this->v_weight_data_int8_scales.empty() || out_scale.empty()) {
                return -100;
            }
            this->out_weight_data_int8_scale = out_scale[0];
        }
        return 0;
    }

    // Parses ncnn's quantize_term: bits*100 + input_scale*10 + block_code for
    // weight-block quantization (>= 400), otherwise the legacy int8 path.
    std::int32_t parse_quantize_term() {
        const std::int32_t term = this->quantize_term;
        if (term == 4 || term == 5 || term == 6) {
            return -1;
        }
        if (term >= 400) {
            const std::int32_t bits = term / 100;
            const std::int32_t format = (term % 100) / 10;
            const std::int32_t block_code = term % 10;
            if ((bits != 4 && bits != 6 && bits != 8) || format > 1 || block_code > 2) {
                return -1;
            }
            this->weight_block_quantize = true;
            this->weight_bits = bits;
            this->has_input_scale = format == 1;
            this->block_size = block_code == 0 ? 32 : (block_code == 1 ? 64 : 128);
            return 0;
        }
        this->legacy_int8 = true;
        return 0;
    }

    bool use_quantized_weights() const {
        return this->weight_block_quantize ||
               (this->legacy_int8 && !this->q_weight_data.empty() && this->q_weight_data.elemsize == 1u);
    }

    float project_input(const MatT& A, bool transA, std::int32_t i, std::int32_t k) const {
        return transA ? A.row(k)[i] : A.row(i)[k];
    }

    // dst = A * W^T + bias with quantized weights. transA reads A as (K, M)
    // instead of (M, K); output_transpose writes dst.row(j)[i] instead of
    // row(i)[j]. Mirrors ncnn's MultiHeadAttention quantized projections.
    void project_quantized(const MatT& A, bool transA, std::int32_t M, std::int32_t K, std::int32_t N, float mul,
                           const MatT& weight, const MatT& weight_scales, const MatT& input_scales,
                           const MatT& bias, MatT& top, bool output_transpose, const OptionT& opt) const {
        const bool has_input_scale = !input_scales.empty();
        const std::int32_t block_size = this->block_size > 0 ? this->block_size : 1;

        if (this->weight_block_quantize && this->weight_bits == 8) {
            const std::int32_t block_count = (K + block_size - 1) / block_size;
            #pragma omp parallel num_threads(opt.num_threads)
            {
                std::vector<signed char> a_int8(static_cast<std::size_t>(K));
                std::vector<float> a_descale(static_cast<std::size_t>(block_count));
                #pragma omp for
                for (std::int32_t i = 0; i < M; ++i) {
                    for (std::int32_t g = 0; g < block_count; ++g) {
                        const std::int32_t k0 = g * block_size;
                        const std::int32_t k1 = std::min(k0 + block_size, K);
                        float absmax = 0.0F;
                        for (std::int32_t k = k0; k < k1; ++k) {
                            float v = this->project_input(A, transA, i, k);
                            if (has_input_scale) {
                                v *= input_scales[k];
                            }
                            absmax = std::max(absmax, std::fabs(v));
                        }
                        if (absmax == 0.0F) {
                            a_descale[static_cast<std::size_t>(g)] = 0.0F;
                            for (std::int32_t k = k0; k < k1; ++k) {
                                a_int8[static_cast<std::size_t>(k)] = 0;
                            }
                            continue;
                        }
                        const float quant_scale = 127.0F / absmax;
                        a_descale[static_cast<std::size_t>(g)] = absmax / 127.0F;
                        for (std::int32_t k = k0; k < k1; ++k) {
                            float v = this->project_input(A, transA, i, k);
                            if (has_input_scale) {
                                v *= input_scales[k];
                            }
                            a_int8[static_cast<std::size_t>(k)] = detail::mha_float2int8(v * quant_scale);
                        }
                    }

                    for (std::int32_t j = 0; j < N; ++j) {
                        const signed char* wptr = weight.template row<const signed char>(j);
                        const float* sptr = weight_scales.row(j);
                        float sum = bias[j];
                        for (std::int32_t g = 0; g < block_count; ++g) {
                            const std::int32_t k0 = g * block_size;
                            const std::int32_t k1 = std::min(k0 + block_size, K);
                            std::int32_t acc = 0;
                            for (std::int32_t k = k0; k < k1; ++k) {
                                acc += static_cast<std::int32_t>(a_int8[static_cast<std::size_t>(k)]) * wptr[k];
                            }
                            sum += static_cast<float>(acc) *
                                   (a_descale[static_cast<std::size_t>(g)] / sptr[g]);
                        }
                        if (output_transpose) {
                            top.row(j)[i] = sum * mul;
                        } else {
                            top.row(i)[j] = sum * mul;
                        }
                    }
                }
            }
            return;
        }

        if (this->weight_block_quantize) {
            const std::int32_t block_count = (K + block_size - 1) / block_size;
            const std::int32_t bits = this->weight_bits;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (std::int32_t i = 0; i < M; ++i) {
                for (std::int32_t j = 0; j < N; ++j) {
                    const unsigned char* wptr = weight.template row<const unsigned char>(j);
                    const float* sptr = weight_scales.row(j);
                    float sum = bias[j];
                    for (std::int32_t g = 0; g < block_count; ++g) {
                        const std::int32_t k0 = g * block_size;
                        const std::int32_t k1 = std::min(k0 + block_size, K);
                        const float descale = 1.0F / sptr[g];
                        for (std::int32_t k = k0; k < k1; ++k) {
                            const float q = static_cast<float>(detail::mha_unpack_weight(wptr, k, bits));
                            float v = this->project_input(A, transA, i, k);
                            if (has_input_scale) {
                                v *= input_scales[k];
                            }
                            sum += v * (q * descale);
                        }
                    }
                    if (output_transpose) {
                        top.row(j)[i] = sum * mul;
                    } else {
                        top.row(i)[j] = sum * mul;
                    }
                }
            }
            return;
        }

        // Legacy int8 activation quantization follows ncnn: q/k/v projections
        // use one scale for the whole blob (dynamic_quantize_2d), the output
        // projection one scale per sample (dynamic_quantize_2d_per_w).
        std::vector<float> a_scales(static_cast<std::size_t>(transA ? M : 1), 1.0F);
        if (transA) {
            for (std::int32_t i = 0; i < M; ++i) {
                float absmax = 0.0F;
                for (std::int32_t k = 0; k < K; ++k) {
                    absmax = std::max(absmax, std::fabs(this->project_input(A, true, i, k)));
                }
                a_scales[static_cast<std::size_t>(i)] = absmax == 0.0F ? 1.0F : 127.0F / absmax;
            }
        } else {
            float absmax = 0.0F;
            for (std::int32_t i = 0; i < M; ++i) {
                for (std::int32_t k = 0; k < K; ++k) {
                    absmax = std::max(absmax, std::fabs(this->project_input(A, false, i, k)));
                }
            }
            a_scales[0] = absmax == 0.0F ? 1.0F : 127.0F / absmax;
        }

        std::vector<signed char> a_int8(static_cast<std::size_t>(M) * static_cast<std::size_t>(K));
        #pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t i = 0; i < M; ++i) {
            const float scale = transA ? a_scales[static_cast<std::size_t>(i)] : a_scales[0];
            for (std::int32_t k = 0; k < K; ++k) {
                a_int8[static_cast<std::size_t>(i) * static_cast<std::size_t>(K) + static_cast<std::size_t>(k)] =
                    detail::mha_float2int8(this->project_input(A, transA, i, k) * scale);
            }
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t i = 0; i < M; ++i) {
            const signed char* aptr = &a_int8[static_cast<std::size_t>(i) * static_cast<std::size_t>(K)];
            for (std::int32_t j = 0; j < N; ++j) {
                const signed char* wptr = static_cast<const signed char*>(weight) + static_cast<std::size_t>(j) * static_cast<std::size_t>(K);
                std::int32_t acc = 0;
                for (std::int32_t k = 0; k < K; ++k) {
                    acc += static_cast<std::int32_t>(aptr[k]) * wptr[k];
                }
                const float weight_scale =
                    weight_scales.empty() ? this->out_weight_data_int8_scale : weight_scales[j];
                const float activation_scale = transA ? a_scales[static_cast<std::size_t>(i)] : a_scales[0];
                const float descale = 1.0F / (weight_scale * activation_scale);
                const float sum = static_cast<float>(acc) * descale + bias[j];
                if (output_transpose) {
                    top.row(j)[i] = sum * mul;
                } else {
                    top.row(i)[j] = sum * mul;
                }
            }
        }
    }

    std::int32_t forward(const std::vector<MatT>& bottom_blobs, std::vector<MatT>& top_blobs,
                         const OptionT& opt) const override {
        if (top_blobs.empty()) {
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

        const bool quantized = this->use_quantized_weights();
        const MatT& q_scale_data =
            this->weight_block_quantize ? this->q_weight_data_quantize_scales : this->q_weight_data_int8_scales;
        const MatT& k_scale_data =
            this->weight_block_quantize ? this->k_weight_data_quantize_scales : this->k_weight_data_int8_scales;
        const MatT& v_scale_data =
            this->weight_block_quantize ? this->v_weight_data_quantize_scales : this->v_weight_data_int8_scales;
        const MatT& out_scale_data =
            this->weight_block_quantize ? this->out_weight_data_quantize_scales : MatT();

        MatT q_affine;
        q_affine.create(src_seqlen, this->embed_dim, 4u, opt.workspace_allocator);
        if (q_affine.empty()) {
            return -100;
        }
        if (quantized) {
            this->project_quantized(q_blob, false, src_seqlen, qdim, this->embed_dim, this->scale,
                                    this->q_weight_data, q_scale_data, this->q_weight_data_input_scales,
                                    this->q_bias_data, q_affine, true, opt);
        } else {
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
        }

        MatT k_affine;
        MatT v_affine;
        if (append_kv) {
            k_affine.create(cur_seqlen, this->embed_dim, 4u, opt.workspace_allocator);
            v_affine.create(cur_seqlen, this->embed_dim, 4u, opt.workspace_allocator);
            if (k_affine.empty() || v_affine.empty()) {
                return -100;
            }
            if (quantized) {
                this->project_quantized(k_blob, false, cur_seqlen, this->kdim, this->embed_dim, 1.0F,
                                        this->k_weight_data, k_scale_data, this->k_weight_data_input_scales,
                                        this->k_bias_data, k_affine, true, opt);
            } else {
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
            }
            if (quantized) {
                this->project_quantized(v_blob, false, cur_seqlen, this->vdim, this->embed_dim, 1.0F,
                                        this->v_weight_data, v_scale_data, this->v_weight_data_input_scales,
                                        this->v_bias_data, v_affine, true, opt);
            } else {
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
            const MatT k_head = this->kv_cache != 0
                                    ? cached_k.channel(q)
                                    : k_affine.row_range(q * embed_dim_per_head, embed_dim_per_head);
            MatT qk_cross_head = qk_cross.channel(q);
            if (this->legacy_int8) {
                // ncnn quantizes q per head column and k per head blob, then
                // does the dot product in int8.
                std::vector<float> q_scales(static_cast<std::size_t>(src_seqlen), 1.0F);
                std::vector<signed char> q_int(static_cast<std::size_t>(embed_dim_per_head) *
                                               static_cast<std::size_t>(src_seqlen));
                for (std::int32_t i = 0; i < src_seqlen; ++i) {
                    float absmax = 0.0F;
                    for (std::int32_t l = 0; l < embed_dim_per_head; ++l) {
                        absmax = std::max(absmax, std::fabs(q_affine_head.row(l)[i]));
                    }
                    q_scales[static_cast<std::size_t>(i)] = absmax == 0.0F ? 1.0F : 127.0F / absmax;
                    for (std::int32_t l = 0; l < embed_dim_per_head; ++l) {
                        q_int[static_cast<std::size_t>(l) * static_cast<std::size_t>(src_seqlen) +
                              static_cast<std::size_t>(i)] = detail::mha_float2int8(
                            q_affine_head.row(l)[i] * q_scales[static_cast<std::size_t>(i)]);
                    }
                }
                float k_absmax = 0.0F;
                if (this->kv_cache != 0) {
                    for (std::int32_t j = 0; j < dst_seqlen; ++j) {
                        for (std::int32_t l = 0; l < embed_dim_per_head; ++l) {
                            k_absmax = std::max(k_absmax, std::fabs(k_head.row(j)[l]));
                        }
                    }
                } else {
                    for (std::int32_t l = 0; l < embed_dim_per_head; ++l) {
                        for (std::int32_t j = 0; j < dst_seqlen; ++j) {
                            k_absmax = std::max(k_absmax, std::fabs(k_head.row(l)[j]));
                        }
                    }
                }
                const float k_scale = k_absmax == 0.0F ? 1.0F : 127.0F / k_absmax;
                std::vector<signed char> k_int(static_cast<std::size_t>(k_head.w) *
                                               static_cast<std::size_t>(k_head.h));
                if (this->kv_cache != 0) {
                    for (std::int32_t j = 0; j < dst_seqlen; ++j) {
                        for (std::int32_t l = 0; l < embed_dim_per_head; ++l) {
                            k_int[static_cast<std::size_t>(j) * static_cast<std::size_t>(k_head.w) +
                                  static_cast<std::size_t>(l)] =
                                detail::mha_float2int8(k_head.row(j)[l] * k_scale);
                        }
                    }
                } else {
                    for (std::int32_t l = 0; l < embed_dim_per_head; ++l) {
                        for (std::int32_t j = 0; j < dst_seqlen; ++j) {
                            k_int[static_cast<std::size_t>(l) * static_cast<std::size_t>(k_head.w) +
                                  static_cast<std::size_t>(j)] =
                                detail::mha_float2int8(k_head.row(l)[j] * k_scale);
                        }
                    }
                }
                for (std::int32_t i = 0; i < src_seqlen; ++i) {
                    float* outptr = qk_cross_head.row(i);
                    const float descale = 1.0F / (q_scales[static_cast<std::size_t>(i)] * k_scale);
                    for (std::int32_t j = 0; j < dst_seqlen; ++j) {
                        std::int32_t acc = 0;
                        for (std::int32_t l = 0; l < embed_dim_per_head; ++l) {
                            const std::int32_t ki = this->kv_cache != 0
                                                        ? j * k_head.w + l
                                                        : l * k_head.w + j;
                            acc += static_cast<std::int32_t>(
                                       q_int[static_cast<std::size_t>(l) * static_cast<std::size_t>(src_seqlen) +
                                             static_cast<std::size_t>(i)]) *
                                   static_cast<std::int32_t>(k_int[static_cast<std::size_t>(ki)]);
                        }
                        outptr[j] = static_cast<float>(acc) * descale;
                    }
                }
                continue;
            }
            for (std::int32_t i = 0; i < src_seqlen; ++i) {
                float* outptr = qk_cross_head.row(i);
                for (std::int32_t j = 0; j < dst_seqlen; ++j) {
                    float sum = 0.0F;
                    if (this->kv_cache != 0) {
                        for (std::int32_t l = 0; l < embed_dim_per_head; ++l) {
                            sum += q_affine_head.row(l)[i] * k_head.row(j)[l];
                        }
                    } else {
                        for (std::int32_t l = 0; l < embed_dim_per_head; ++l) {
                            sum += q_affine_head.row(l)[i] * k_head.row(l)[j];
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
            if (this->legacy_int8) {
                // ncnn quantizes the attention weights per row and v per head
                // blob before the int8 dot product.
                std::vector<float> qk_scales(static_cast<std::size_t>(src_seqlen), 1.0F);
                std::vector<signed char> qk_int(static_cast<std::size_t>(src_seqlen) *
                                                static_cast<std::size_t>(dst_seqlen));
                for (std::int32_t i = 0; i < src_seqlen; ++i) {
                    float absmax = 0.0F;
                    for (std::int32_t k = 0; k < dst_seqlen; ++k) {
                        absmax = std::max(absmax, std::fabs(qk_cross_head.row(i)[k]));
                    }
                    qk_scales[static_cast<std::size_t>(i)] = absmax == 0.0F ? 1.0F : 127.0F / absmax;
                    for (std::int32_t k = 0; k < dst_seqlen; ++k) {
                        qk_int[static_cast<std::size_t>(i) * static_cast<std::size_t>(dst_seqlen) +
                               static_cast<std::size_t>(k)] =
                            detail::mha_float2int8(qk_cross_head.row(i)[k] * qk_scales[static_cast<std::size_t>(i)]);
                    }
                }
                float v_absmax = 0.0F;
                if (this->kv_cache != 0) {
                    for (std::int32_t k = 0; k < dst_seqlen; ++k) {
                        for (std::int32_t j = 0; j < embed_dim_per_head; ++j) {
                            v_absmax = std::max(v_absmax, std::fabs(v_affine_head.row(k)[j]));
                        }
                    }
                } else {
                    for (std::int32_t j = 0; j < embed_dim_per_head; ++j) {
                        for (std::int32_t k = 0; k < dst_seqlen; ++k) {
                            v_absmax = std::max(v_absmax, std::fabs(v_affine_head.row(j)[k]));
                        }
                    }
                }
                const float v_scale = v_absmax == 0.0F ? 1.0F : 127.0F / v_absmax;
                std::vector<signed char> v_int(static_cast<std::size_t>(v_affine_head.w) *
                                               static_cast<std::size_t>(v_affine_head.h));
                if (this->kv_cache != 0) {
                    for (std::int32_t k = 0; k < dst_seqlen; ++k) {
                        for (std::int32_t j = 0; j < embed_dim_per_head; ++j) {
                            v_int[static_cast<std::size_t>(k) * static_cast<std::size_t>(v_affine_head.w) +
                                  static_cast<std::size_t>(j)] =
                                detail::mha_float2int8(v_affine_head.row(k)[j] * v_scale);
                        }
                    }
                } else {
                    for (std::int32_t j = 0; j < embed_dim_per_head; ++j) {
                        for (std::int32_t k = 0; k < dst_seqlen; ++k) {
                            v_int[static_cast<std::size_t>(j) * static_cast<std::size_t>(v_affine_head.w) +
                                  static_cast<std::size_t>(k)] =
                                detail::mha_float2int8(v_affine_head.row(j)[k] * v_scale);
                        }
                    }
                }
                for (std::int32_t i = 0; i < src_seqlen; ++i) {
                    const signed char* qkptr =
                        &qk_int[static_cast<std::size_t>(i) * static_cast<std::size_t>(dst_seqlen)];
                    const float descale = 1.0F / (qk_scales[static_cast<std::size_t>(i)] * v_scale);
                    for (std::int32_t j = 0; j < embed_dim_per_head; ++j) {
                        std::int32_t acc = 0;
                        for (std::int32_t k = 0; k < dst_seqlen; ++k) {
                            const std::int32_t vi = this->kv_cache != 0 ? k * v_affine_head.w + j
                                                                       : j * v_affine_head.w + k;
                            acc += static_cast<std::int32_t>(qkptr[k]) *
                                   static_cast<std::int32_t>(v_int[static_cast<std::size_t>(vi)]);
                        }
                        qkv_cross_head.row(j)[i] = static_cast<float>(acc) * descale;
                    }
                }
                continue;
            }
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
        if (quantized) {
            this->project_quantized(qkv_cross, true, src_seqlen, this->embed_dim, qdim, 1.0F,
                                    this->out_weight_data, out_scale_data, this->out_weight_data_input_scales,
                                    this->out_bias_data, top_blob, false, opt);
        } else {
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
    bool weight_block_quantize = false;
    bool legacy_int8 = false;
    std::int32_t weight_bits = 0;
    std::int32_t block_size = 0;
    bool has_input_scale = false;
    float out_weight_data_int8_scale = 1.0F;
    MatT q_weight_data_int8_scales;
    MatT k_weight_data_int8_scales;
    MatT v_weight_data_int8_scales;
    MatT q_weight_data_quantize_scales;
    MatT k_weight_data_quantize_scales;
    MatT v_weight_data_quantize_scales;
    MatT out_weight_data_quantize_scales;
    MatT q_weight_data_input_scales;
    MatT k_weight_data_input_scales;
    MatT v_weight_data_input_scales;
    MatT out_weight_data_input_scales;

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
