// Copyright 2020 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_STATISTICSPOOLING_HPP
#define MLAYER_STATISTICSPOOLING_HPP

#include <cmath>
#include <cstdint>

namespace mlayer {

// Global mean (and optional standard deviation) pooling over w/h/d.
template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT>
class StatisticsPooling : public LayerT {
public:
    StatisticsPooling() {
        this->one_blob_only = true;
        this->support_inplace = false;
        this->include_stddev = 0;
    }

    std::int32_t load_param(const ParamDictT& pd) override {
        this->include_stddev = pd.get(0, 0);
        return 0;
    }

    std::int32_t forward(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const override {
        const std::int32_t w = bottom_blob.w;
        const std::int32_t h = bottom_blob.h;
        const std::int32_t d = bottom_blob.d;
        const std::int32_t channels = bottom_blob.c;
        const std::int32_t size = w * h * d;
        const std::size_t elemsize = bottom_blob.elemsize;
        const std::int32_t out_channels = this->include_stddev != 0 ? channels * 2 : channels;

        top_blob.create(out_channels, elemsize, opt.blob_allocator);
        if (top_blob.empty()) {
            return -100;
        }

#pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t q = 0; q < channels; ++q) {
            const float* ptr = bottom_blob.channel(q);
            float mean = 0.0F;
            for (std::int32_t i = 0; i < size; ++i) {
                mean += ptr[i];
            }
            top_blob[q] = mean / static_cast<float>(size);
        }
        if (this->include_stddev != 0) {
#pragma omp parallel for num_threads(opt.num_threads)
            for (std::int32_t q = channels; q < out_channels; ++q) {
                const float* ptr = bottom_blob.channel(q - channels);
                const float mean = top_blob[q - channels];
                float var = 0.0F;
                for (std::int32_t i = 0; i < size; ++i) {
                    const float v = ptr[i] - mean;
                    var += v * v;
                }
                top_blob[q] = std::sqrt(var / static_cast<float>(size));
            }
        }
        return 0;
    }

public:
    std::int32_t include_stddev;
};

}  // namespace mlayer

#endif  // MLAYER_STATISTICSPOOLING_HPP
