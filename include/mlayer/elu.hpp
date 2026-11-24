// Copyright 2017 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_ELU_HPP
#define MLAYER_ELU_HPP

#include <cmath>
#include <cstdint>

namespace mlayer {

template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT>
class ELU : public LayerT {
public:
    ELU() {
        this->one_blob_only = true;
        this->support_inplace = true;
        this->alpha = 0.1F;
    }

    std::int32_t load_param(const ParamDictT& pd) override {
        this->alpha = pd.get(0, 0.1F);
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
                if (ptr[i] < 0.0F) {
                    ptr[i] = this->alpha * (std::exp(ptr[i]) - 1.0F);
                }
            }
        }

        return 0;
    }

public:
    float alpha;
};

}  // namespace mlayer

#endif  // MLAYER_ELU_HPP
