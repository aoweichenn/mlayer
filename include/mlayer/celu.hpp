// Copyright 2020 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_CELU_HPP
#define MLAYER_CELU_HPP

#include <cmath>
#include <cstdint>

namespace mlayer {

// Continuously differentiable ELU: x > 0 ? x : alpha * (exp(x / alpha) - 1).
template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT>
class CELU : public LayerT {
public:
    CELU() {
        this->one_blob_only = true;
        this->support_inplace = true;
        this->alpha = 1.0F;
    }

    std::int32_t load_param(const ParamDictT& pd) override {
        this->alpha = pd.get(0, 1.0F);
        return 0;
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
            for (std::int32_t i = 0; i < size; ++i) {
                const float v = ptr[i];
                ptr[i] = v > 0.0F ? v : this->alpha * (std::exp(v / this->alpha) - 1.0F);
            }
        }
        return 0;
    }

public:
    float alpha;
};

}  // namespace mlayer

#endif  // MLAYER_CELU_HPP
