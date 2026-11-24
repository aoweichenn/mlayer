// Copyright 2017 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_SELU_HPP
#define MLAYER_SELU_HPP

#include <cmath>
#include <cstdint>

namespace mlayer {

template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT>
class SELU : public LayerT {
public:
    SELU() {
        this->one_blob_only = true;
        this->support_inplace = true;
        this->alpha = 1.67326324F;
        this->lambda = 1.050700987F;
    }

    std::int32_t load_param(const ParamDictT& pd) override {
        this->alpha = pd.get(0, 1.67326324F);
        this->lambda = pd.get(1, 1.050700987F);
        return 0;
    }

    std::int32_t forward_inplace(MatT& bottom_top_blob, const OptionT& opt) const override {
        const std::int32_t w = bottom_top_blob.w;
        const std::int32_t h = bottom_top_blob.h;
        const std::int32_t d = bottom_top_blob.d;
        const std::int32_t channels = bottom_top_blob.c;
        const std::int32_t size = w * h * d;
        const float alphaxlambda = this->alpha * this->lambda;

#pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t q = 0; q < channels; ++q) {
            auto* ptr = static_cast<float*>(bottom_top_blob.channel(q));
            for (std::int32_t i = 0; i < size; ++i) {
                if (ptr[i] < 0.0F) {
                    ptr[i] = (std::exp(ptr[i]) - 1.0F) * alphaxlambda;
                } else {
                    ptr[i] *= this->lambda;
                }
            }
        }

        return 0;
    }

public:
    float alpha;
    float lambda;
};

}  // namespace mlayer

#endif  // MLAYER_SELU_HPP
