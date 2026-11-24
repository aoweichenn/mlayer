// Copyright 2017 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_ABSVAL_HPP
#define MLAYER_ABSVAL_HPP

#include <cstdint>

namespace mlayer {

// Element-wise absolute value. LayerT/MatT/OptionT stay consumer-defined so the
// implementation can live outside the ncnn layer tree.
template<typename LayerT, typename MatT, typename OptionT>
class AbsVal : public LayerT {
public:
    AbsVal() {
        this->one_blob_only = true;
        this->support_inplace = true;
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
                    ptr[i] = -ptr[i];
                }
            }
        }

        return 0;
    }
};

}  // namespace mlayer

#endif  // MLAYER_ABSVAL_HPP
