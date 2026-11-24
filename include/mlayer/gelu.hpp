// Copyright 2021 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_GELU_HPP
#define MLAYER_GELU_HPP

#include <cmath>
#include <cstdint>

namespace mlayer {

template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT>
class GELU : public LayerT {
public:
    GELU() {
        this->one_blob_only = true;
        this->support_inplace = true;
        this->fast_gelu = 0;
    }

    std::int32_t load_param(const ParamDictT& pd) override {
        this->fast_gelu = pd.get(0, 0);
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
            if (this->fast_gelu != 0) {
                for (std::int32_t i = 0; i < size; ++i) {
                    // y = 0.5x * (1 + tanh(sqrt(2/Pi) * (x + 0.044715x^3)))
                    ptr[i] = 0.5F * ptr[i] *
                             (1.0F + std::tanh(kFastGeluScale * (ptr[i] + kFastGeluCubic * ptr[i] * ptr[i] * ptr[i])));
                }
            } else {
                for (std::int32_t i = 0; i < size; ++i) {
                    // y = x * P(X <= x) where X ~ N(0, 1)
                    ptr[i] = 0.5F * ptr[i] * std::erfc(kInvSqrt2 * ptr[i]);
                }
            }
        }

        return 0;
    }

public:
    std::int32_t fast_gelu;

private:
    static constexpr float kFastGeluScale = 0.79788452F;
    static constexpr float kFastGeluCubic = 0.044715F;
    static constexpr float kInvSqrt2 = -0.70710678F;
};

}  // namespace mlayer

#endif  // MLAYER_GELU_HPP
