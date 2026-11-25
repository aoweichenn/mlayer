// Copyright 2020 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_SHRINK_HPP
#define MLAYER_SHRINK_HPP

#include <cstdint>

namespace mlayer {

// Soft thresholding: |x| <= lambda -> 0, x > lambda -> x - bias,
// x < -lambda -> x + bias.
template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT>
class Shrink : public LayerT {
public:
    Shrink() {
        this->one_blob_only = true;
        this->support_inplace = true;
        this->bias = 0.0F;
        this->lambd = 0.5F;
    }

    std::int32_t load_param(const ParamDictT& pd) override {
        this->bias = pd.get(0, 0.0F);
        this->lambd = pd.get(1, 0.5F);
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
                if (v > this->lambd) {
                    ptr[i] = v - this->bias;
                } else if (v < -this->lambd) {
                    ptr[i] = v + this->bias;
                } else {
                    ptr[i] = 0.0F;
                }
            }
        }
        return 0;
    }

public:
    float bias;
    float lambd;
};

}  // namespace mlayer

#endif  // MLAYER_SHRINK_HPP
