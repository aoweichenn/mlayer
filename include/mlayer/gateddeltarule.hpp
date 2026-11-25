// Copyright 2026 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_GATEDDELTARULE_HPP
#define MLAYER_GATEDDELTARULE_HPP

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace mlayer {

// Recurrent gated delta rule used by Qwen3.5-style linear attention blocks.
template<typename LayerT, typename MatT, typename OptionT>
class GatedDeltaRule : public LayerT {
public:
    GatedDeltaRule() {
        this->one_blob_only = false;
        this->support_inplace = false;
    }

    std::int32_t forward(const std::vector<MatT>& bottom_blobs, std::vector<MatT>& top_blobs,
                         const OptionT& opt) const override {
        if (bottom_blobs.size() < 7 || top_blobs.empty()) {
            return -100;
        }

        const MatT& A_log = bottom_blobs[0];
        const MatT& dt_bias = bottom_blobs[1];
        const MatT& b = bottom_blobs[2];
        const MatT& a = bottom_blobs[3];
        const MatT& query = bottom_blobs[4];
        const MatT& key = bottom_blobs[5];
        const MatT& value = bottom_blobs[6];
        const MatT& initial_state = bottom_blobs.size() > 7 ? bottom_blobs[7] : MatT();

        const std::int32_t query_heads = query.h;
        const std::int32_t num_heads = value.h;
        const std::int32_t seq_len = query.c;
        const std::int32_t k_head_dim = query.w;
        const std::int32_t v_head_dim = value.w;

        if (query_heads <= 0 || num_heads <= 0 || num_heads % query_heads != 0 || seq_len <= 0 ||
            k_head_dim <= 0 || v_head_dim <= 0 || query.dims != 3 || key.dims != 3 || value.dims != 3 ||
            b.dims != 2 || a.dims != 2 || A_log.dims != 1 || dt_bias.dims != 1 ||
            key.w != k_head_dim || key.h != query_heads || key.c != seq_len || value.c != seq_len ||
            b.w != num_heads || b.h != seq_len || a.w != num_heads || a.h != seq_len ||
            A_log.w < num_heads || dt_bias.w < num_heads) {
            return -100;
        }
        if (!initial_state.empty() &&
            (initial_state.dims != 3 || initial_state.w != v_head_dim || initial_state.h != k_head_dim ||
             initial_state.c != num_heads)) {
            return -100;
        }

        MatT& top_blob = top_blobs[0];
        top_blob.create(v_head_dim, num_heads, seq_len, 4u, opt.blob_allocator);
        if (top_blob.empty()) {
            return -100;
        }

        MatT state_out_workspace;
        MatT& state_out = top_blobs.size() > 1 ? top_blobs[1] : state_out_workspace;
        if (top_blobs.size() > 1) {
            state_out.create(v_head_dim, k_head_dim, num_heads, 4u, opt.blob_allocator);
        } else {
            state_out.create(v_head_dim, k_head_dim, num_heads, 4u, opt.workspace_allocator);
        }
        if (state_out.empty()) {
            return -100;
        }

        const auto* query_data = static_cast<const float*>(query.data);
        const auto* key_data = static_cast<const float*>(key.data);
        const auto* value_data = static_cast<const float*>(value.data);

        MatT query_t(k_head_dim, seq_len, num_heads, 4u, opt.workspace_allocator);
        MatT key_t(k_head_dim, seq_len, num_heads, 4u, opt.workspace_allocator);
        MatT value_t(v_head_dim, seq_len, num_heads, 4u, opt.workspace_allocator);
        if (query_t.empty() || key_t.empty() || value_t.empty()) {
            return -100;
        }

        auto* query_t_data = static_cast<float*>(query_t.data);
        auto* key_t_data = static_cast<float*>(key_t.data);
        auto* value_t_data = static_cast<float*>(value_t.data);

        for (std::int32_t t = 0; t < seq_len; ++t) {
            for (std::int32_t h = 0; h < num_heads; ++h) {
                const std::int32_t repeat = num_heads / query_heads;
                const std::int32_t source_head = query_heads == num_heads ? h : h / repeat;
                for (std::int32_t d = 0; d < k_head_dim; ++d) {
                    const std::int32_t src_idx = (t * query_heads + source_head) * k_head_dim + d;
                    const std::int32_t dst_idx = (h * seq_len + t) * k_head_dim + d;
                    query_t_data[dst_idx] = query_data[src_idx];
                    key_t_data[dst_idx] = key_data[src_idx];
                }
                for (std::int32_t d = 0; d < v_head_dim; ++d) {
                    const std::int32_t src_idx = (t * num_heads + h) * v_head_dim + d;
                    const std::int32_t dst_idx = (h * seq_len + t) * v_head_dim + d;
                    value_t_data[dst_idx] = value_data[src_idx];
                }
            }
        }

        const auto* b_data = static_cast<const float*>(b.data);
        const auto* a_data = static_cast<const float*>(a.data);
        const auto* A_log_data = static_cast<const float*>(A_log.data);
        const auto* dt_bias_data = static_cast<const float*>(dt_bias.data);

        MatT beta(seq_len, num_heads, 4u, opt.workspace_allocator);
        MatT g(seq_len, num_heads, 4u, opt.workspace_allocator);
        if (beta.empty() || g.empty()) {
            return -100;
        }

        auto* beta_data = static_cast<float*>(beta.data);
        auto* g_data = static_cast<float*>(g.data);

        for (std::int32_t h = 0; h < num_heads; ++h) {
            for (std::int32_t t = 0; t < seq_len; ++t) {
                beta_data[h * seq_len + t] = sigmoid(b_data[t * num_heads + h]);
            }
        }

        for (std::int32_t h = 0; h < num_heads; ++h) {
            const float exp_A = std::exp(A_log_data[h]);
            const float dt_bias_val = dt_bias_data[h];
            for (std::int32_t t = 0; t < seq_len; ++t) {
                const float sp_val = softplus(a_data[t * num_heads + h] + dt_bias_val);
                g_data[h * seq_len + t] = -exp_A * sp_val;
            }
        }

        MatT core_attn_out(v_head_dim, seq_len, num_heads, 4u, opt.workspace_allocator);
        if (core_attn_out.empty()) {
            return -100;
        }

        auto* state_data = static_cast<float*>(state_out.data);
        const std::size_t state_bytes =
            static_cast<std::size_t>(num_heads) * static_cast<std::size_t>(k_head_dim) *
            static_cast<std::size_t>(v_head_dim) * sizeof(float);
        if (!initial_state.empty()) {
            std::memcpy(state_data, initial_state.data, state_bytes);
        } else {
            std::memset(state_data, 0, state_bytes);
        }

        const std::int32_t ret = recurrent_gated_delta_rule(
            query_t_data, key_t_data, value_t_data, g_data, beta_data,
            static_cast<float*>(core_attn_out.data), state_data, num_heads, seq_len, k_head_dim,
            v_head_dim, opt);
        if (ret != 0) {
            return ret;
        }

        const auto* core_attn_out_data = static_cast<const float*>(core_attn_out.data);
        auto* top_data = static_cast<float*>(top_blob.data);
        for (std::int32_t h = 0; h < num_heads; ++h) {
            for (std::int32_t t = 0; t < seq_len; ++t) {
                for (std::int32_t d = 0; d < v_head_dim; ++d) {
                    const std::int32_t src_idx = (h * seq_len + t) * v_head_dim + d;
                    const std::int32_t dst_idx = (t * num_heads + h) * v_head_dim + d;
                    top_data[dst_idx] = core_attn_out_data[src_idx];
                }
            }
        }

        return 0;
    }

private:
    static float sigmoid(float x) { return 1.0F / (1.0F + std::exp(-x)); }

    static float softplus(float x) {
        if (x > 20.0F) {
            return x;
        }
        if (x < -20.0F) {
            return std::exp(x);
        }
        return std::log(1.0F + std::exp(x));
    }

    static void l2norm(const float* x, float* out, std::int32_t n, std::int32_t dim, float eps) {
        for (std::int32_t i = 0; i < n; ++i) {
            const float* row_in = x + i * dim;
            float* row_out = out + i * dim;
            float sum = 0.0F;
            for (std::int32_t j = 0; j < dim; ++j) {
                sum += row_in[j] * row_in[j];
            }
            const float inv_norm = 1.0F / std::sqrt(sum + eps);
            for (std::int32_t j = 0; j < dim; ++j) {
                row_out[j] = row_in[j] * inv_norm;
            }
        }
    }

    static std::int32_t recurrent_gated_delta_rule(const float* query, const float* key,
                                                   const float* value, const float* g, const float* beta,
                                                   float* core_attn_out, float* last_recurrent_state,
                                                   std::int32_t num_heads, std::int32_t seq_len,
                                                   std::int32_t k_head_dim, std::int32_t v_head_dim,
                                                   const OptionT& opt) {
        MatT query_norm;
        MatT key_norm;
        query_norm.create(k_head_dim, seq_len, num_heads, 4u, opt.workspace_allocator);
        key_norm.create(k_head_dim, seq_len, num_heads, 4u, opt.workspace_allocator);
        if (query_norm.empty() || key_norm.empty()) {
            return -100;
        }
        const std::int32_t rows = num_heads * seq_len;
        l2norm(query, static_cast<float*>(query_norm.data), rows, k_head_dim, 1e-6F);
        l2norm(key, static_cast<float*>(key_norm.data), rows, k_head_dim, 1e-6F);

        const auto* q_ptr = static_cast<const float*>(query_norm.data);
        const auto* k_ptr = static_cast<const float*>(key_norm.data);
        const float scale = 1.0F / std::sqrt(static_cast<float>(k_head_dim));

        std::memset(core_attn_out, 0,
                    static_cast<std::size_t>(num_heads) * static_cast<std::size_t>(seq_len) *
                        static_cast<std::size_t>(v_head_dim) * sizeof(float));

        MatT kv_mem(v_head_dim, 4u, opt.workspace_allocator);
        MatT delta(v_head_dim, 4u, opt.workspace_allocator);
        if (kv_mem.empty() || delta.empty()) {
            return -100;
        }
        auto* kv_mem_ptr = static_cast<float*>(kv_mem.data);
        auto* delta_ptr = static_cast<float*>(delta.data);

        for (std::int32_t t = 0; t < seq_len; ++t) {
            for (std::int32_t h = 0; h < num_heads; ++h) {
                const float* q_t = q_ptr + (h * seq_len + t) * k_head_dim;
                const float* k_t = k_ptr + (h * seq_len + t) * k_head_dim;
                const float* v_t = value + (h * seq_len + t) * v_head_dim;
                const float g_t = g[h * seq_len + t];
                const float beta_t = beta[h * seq_len + t];

                float* state = last_recurrent_state + h * k_head_dim * v_head_dim;
                float* out_t = core_attn_out + (h * seq_len + t) * v_head_dim;

                const float g_t_exp = std::exp(g_t);
                for (std::int32_t i = 0; i < k_head_dim * v_head_dim; ++i) {
                    state[i] *= g_t_exp;
                }

                std::memset(kv_mem_ptr, 0, static_cast<std::size_t>(v_head_dim) * sizeof(float));
                for (std::int32_t dv = 0; dv < v_head_dim; ++dv) {
                    for (std::int32_t dk = 0; dk < k_head_dim; ++dk) {
                        kv_mem_ptr[dv] += state[dk * v_head_dim + dv] * k_t[dk];
                    }
                }

                for (std::int32_t dv = 0; dv < v_head_dim; ++dv) {
                    delta_ptr[dv] = (v_t[dv] - kv_mem_ptr[dv]) * beta_t;
                }

                for (std::int32_t dk = 0; dk < k_head_dim; ++dk) {
                    for (std::int32_t dv = 0; dv < v_head_dim; ++dv) {
                        state[dk * v_head_dim + dv] += k_t[dk] * delta_ptr[dv];
                    }
                }

                for (std::int32_t dv = 0; dv < v_head_dim; ++dv) {
                    float sum = 0.0F;
                    for (std::int32_t dk = 0; dk < k_head_dim; ++dk) {
                        sum += state[dk * v_head_dim + dv] * q_t[dk] * scale;
                    }
                    out_t[dv] = sum;
                }
            }
        }

        return 0;
    }
};

}  // namespace mlayer

#endif  // MLAYER_GATEDDELTARULE_HPP
