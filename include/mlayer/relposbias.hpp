// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_RELPOSBIAS_HPP
#define MLAYER_RELPOSBIAS_HPP

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace mlayer {

// Relative-position attention bias with an attention sink, as used by
// Inkling-2-Mini:
//
//   bias[h, q, k] = dot(rel_bias[:, d], R[:, h, q]) - sink[h, q]
//   d = clamp(rel_pos[q, k], 0, extent - 1), out-of-window d yields no dot term
//
// llama.cpp builds the same tensor with
//   table = mul_mat(pad(attn_rel_bias, 1), Rcur) - sinks
//   kq_b  = get_rows(table, rel_pos)
// (the zero pad row is what makes d == extent contribute nothing).
//
// Inputs:  [0] R          (n_rel, tokens, heads)
//          [1] bias table (n_rel, extent)
//          [2] sink       (heads, tokens)
//          [3] rel_pos    (n_kv, tokens), float codes of the distance index
// Output:  (n_kv, tokens, heads), directly usable as an SDPA attn_mask.
template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT>
class RelPosBias : public LayerT {
public:
    RelPosBias() {
        this->one_blob_only = false;
        this->support_inplace = false;
    }

    std::int32_t load_param(const ParamDictT& /*pd*/) override { return 0; }

    std::int32_t forward(const std::vector<MatT>& bottom_blobs, std::vector<MatT>& top_blobs,
                         const OptionT& opt) const override {
        if (bottom_blobs.size() < 4 || top_blobs.empty()) {
            return -100;
        }
        const MatT& rel = bottom_blobs[0];
        const MatT& table = bottom_blobs[1];
        const MatT& sink = bottom_blobs[2];
        const MatT& rel_pos = bottom_blobs[3];

        const std::int32_t n_rel = rel.w;
        const std::int32_t tokens = rel.h;
        const std::int32_t heads = rel.c;
        const std::int32_t extent = table.h;
        const std::int32_t n_kv = rel_pos.w;

        if (table.w != n_rel || sink.w != heads || sink.h != tokens || rel_pos.h != tokens) {
            return -100;
        }

        MatT& top_blob = top_blobs[0];
        top_blob.create(n_kv, tokens, heads, 4u, opt.blob_allocator);
        if (top_blob.empty()) {
            return -100;
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t h = 0; h < heads; ++h) {
            const MatT rel_head = rel.channel(h);
            MatT out_head = top_blob.channel(h);
            for (std::int32_t q = 0; q < tokens; ++q) {
                const float* r = rel_head.row(q);
                const float sink_h = sink.row(q)[h];
                const float* pos = rel_pos.row(q);
                float* outptr = out_head.row(q);
                for (std::int32_t k = 0; k < n_kv; ++k) {
                    float distance = pos[k];
                    if (distance < 0.0F) {
                        distance = 0.0F;
                    }
                    float sum = 0.0F;
                    if (distance < static_cast<float>(extent)) {
                        const auto d = static_cast<std::int32_t>(distance);
                        const float* bias_row = table.row(d);
                        for (std::int32_t i = 0; i < n_rel; ++i) {
                            sum += bias_row[i] * r[i];
                        }
                    }
                    outptr[k] = sum - sink_h;
                }
            }
        }
        return 0;
    }
};

}  // namespace mlayer

#endif  // MLAYER_RELPOSBIAS_HPP
