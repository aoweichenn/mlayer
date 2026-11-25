// Copyright 2024 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_CUMULATIVESUM_HPP
#define MLAYER_CUMULATIVESUM_HPP

#include <cstdint>

namespace mlayer {

// Inclusive prefix sum along one axis (dims 1..4).
template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT>
class CumulativeSum : public LayerT {
public:
    CumulativeSum() {
        this->one_blob_only = true;
        this->support_inplace = true;
        this->axis = 0;
    }

    std::int32_t load_param(const ParamDictT& pd) override {
        this->axis = pd.get(0, 0);
        return 0;
    }

    std::int32_t forward_inplace(MatT& bottom_top_blob, const OptionT& opt) const override {
        const std::int32_t dims = bottom_top_blob.dims;
        const std::int32_t axis = this->axis < 0 ? dims + this->axis : this->axis;
        const std::int32_t w = bottom_top_blob.w;
        const std::int32_t h = bottom_top_blob.h;
        const std::int32_t d = bottom_top_blob.d;
        const std::int32_t channels = bottom_top_blob.c;

        if (dims == 1) {
            float* ptr = bottom_top_blob;
            for (std::int32_t i = 1; i < w; ++i) {
                ptr[i] += ptr[i - 1];
            }
            return 0;
        }

        // w is always the fastest axis; walk every element and add the value at
        // distance `step` when the axis coordinate is not zero.
        std::int32_t step = 0;
        std::int32_t extent = 1;
        if (dims == 2) {
            step = axis == 0 ? w : 1;
            extent = axis == 0 ? h : w;
        } else if (dims == 3) {
            step = axis == 0 ? w * h : (axis == 1 ? w : 1);
            extent = axis == 0 ? channels : (axis == 1 ? h : w);
        } else if (dims == 4) {
            step = axis == 0 ? w * h * d : (axis == 1 ? w * h : (axis == 2 ? w : 1));
            extent = axis == 0 ? channels : (axis == 1 ? d : (axis == 2 ? h : w));
        } else {
            return -1;
        }

        const std::int32_t count = w * h * d * channels;
        float* ptr = bottom_top_blob;
        const std::int32_t outer = count / (step * extent);
        // Parallel over the planes orthogonal to `axis`; each strand walks the
        // axis serially so the prefix dependency stays intact.
#pragma omp parallel for collapse(2) num_threads(opt.num_threads)
        for (std::int32_t o = 0; o < outer; ++o) {
            for (std::int32_t j = 0; j < step; ++j) {
                const std::int32_t base = o * step * extent + j;
                for (std::int32_t e = 1; e < extent; ++e) {
                    ptr[base + e * step] += ptr[base + (e - 1) * step];
                }
            }
        }
        return 0;
    }

public:
    std::int32_t axis;
};

}  // namespace mlayer

#endif  // MLAYER_CUMULATIVESUM_HPP
