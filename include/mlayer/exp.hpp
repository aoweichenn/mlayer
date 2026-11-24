// Copyright 2017 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_EXP_HPP
#define MLAYER_EXP_HPP

#include <cmath>
#include <cstdint>

namespace mlayer {

template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT>
class Exp : public LayerT {
public:
    Exp() {
        this->one_blob_only = true;
        this->support_inplace = true;
        this->base = -1.0F;
        this->scale = 1.0F;
        this->shift = 0.0F;
    }

    std::int32_t load_param(const ParamDictT& pd) override {
        this->base = pd.get(0, -1.0F);
        this->scale = pd.get(1, 1.0F);
        this->shift = pd.get(2, 0.0F);
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
            if (this->base == -1.0F) {
                for (std::int32_t i = 0; i < size; ++i) {
                    ptr[i] = std::exp(this->shift + ptr[i] * this->scale);
                }
            } else {
                for (std::int32_t i = 0; i < size; ++i) {
                    ptr[i] = std::pow(this->base, this->shift + ptr[i] * this->scale);
                }
            }
        }

        return 0;
    }

public:
    float base;
    float scale;
    float shift;
};

}  // namespace mlayer

#endif  // MLAYER_EXP_HPP
