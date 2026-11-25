// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_SITU_HPP
#define MLAYER_SITU_HPP

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace mlayer {

// SiTU gated activation used by the Inkling-2-Mini / Kimi-K3 MoE experts:
//
//   act(gate) = beta * tanh(gate / beta) * sigmoid(gate)
//   out       = act(gate) * (linear_beta > 0 ? linear_beta * tanh(up / linear_beta) : up)
//
// Two inputs (gate, up) and one output, matching llama.cpp's
// `LLM_FFN_SITU` branch in build_moe_ffn.
template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT>
class Situ : public LayerT {
public:
    Situ() {
        this->one_blob_only = false;
        this->support_inplace = false;
        this->beta = 1.0F;
        this->linear_beta = 0.0F;
    }

    std::int32_t load_param(const ParamDictT& pd) override {
        this->beta = pd.get(0, 1.0F);
        this->linear_beta = pd.get(1, 0.0F);
        if (this->beta <= 0.0F) {
            return -1;
        }
        return 0;
    }

    std::int32_t forward(const std::vector<MatT>& bottom_blobs, std::vector<MatT>& top_blobs,
                         const OptionT& opt) const override {
        if (bottom_blobs.size() < 2 || top_blobs.empty()) {
            return -100;
        }
        const MatT& gate = bottom_blobs[0];
        const MatT& up = bottom_blobs[1];
        if (gate.w != up.w || gate.h != up.h || gate.c != up.c) {
            return -100;
        }

        MatT& top_blob = top_blobs[0];
        top_blob.create(gate.w, gate.h, gate.c, 4u, opt.blob_allocator);
        if (top_blob.empty()) {
            return -100;
        }

        const std::int32_t plane = gate.w * gate.h;
        const float beta = this->beta;
        const float linear_beta = this->linear_beta;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t q = 0; q < gate.c; ++q) {
            const float* g = gate.channel(q);
            const float* u = up.channel(q);
            float* o = top_blob.channel(q);
            for (std::int32_t i = 0; i < plane; ++i) {
                const std::size_t index = static_cast<std::size_t>(i);
                const float gv = g[index];
                const float uv = u[index];
                const float act = beta * std::tanh(gv / beta) / (1.0F + std::exp(-gv));
                const float up_scaled = linear_beta > 0.0F ? linear_beta * std::tanh(uv / linear_beta) : uv;
                o[index] = act * up_scaled;
            }
        }
        return 0;
    }

    float beta;
    float linear_beta;
};

}  // namespace mlayer

#endif  // MLAYER_SITU_HPP
