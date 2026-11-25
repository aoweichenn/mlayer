// Copyright 2017 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_HARDSWISH_HPP
#define MLAYER_HARDSWISH_HPP

#include <cstdint>

namespace mlayer {

template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT>
class HardSwish : public LayerT {
public:
    HardSwish() {
        this->one_blob_only = true;
        this->support_inplace = true;
        this->alpha = 0.2F;
        this->beta = 0.5F;
        this->lower = -this->beta / this->alpha;
        this->upper = (1.0F / this->alpha) + this->lower;
    }

    std::int32_t load_param(const ParamDictT& pd) override {
        this->alpha = pd.get(0, 0.2F);
        this->beta = pd.get(1, 0.5F);
        this->lower = -this->beta / this->alpha;
        this->upper = (1.0F / this->alpha) + this->lower;
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
                if (ptr[i] < this->lower) {
                    ptr[i] = 0.0F;
                } else if (ptr[i] > this->upper) {
                    // keep value
                } else {
                    ptr[i] = ptr[i] * (ptr[i] * this->alpha + this->beta);
                }
            }
        }

        return 0;
    }

    std::int32_t get_fused_activation(std::int32_t& type, MatT& params) const override {
        type = 6;
        params = MatT(2);
        params[0] = alpha;
        params[1] = beta;
        return 0;
    }

public:
    float alpha;
    float beta;
    float lower;
    float upper;
};

}  // namespace mlayer

#endif  // MLAYER_HARDSWISH_HPP
