// Copyright 2021 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_MVN_HPP
#define MLAYER_MVN_HPP

#include <cmath>
#include <cstdint>
#include <vector>

namespace mlayer {

// Mean/variance normalization, per channel or across all channels.
template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT>
class MVN : public LayerT {
public:
    MVN();

    std::int32_t load_param(const ParamDictT& pd) override;
    std::int32_t forward(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const override;

public:
    std::int32_t normalize_variance;
    std::int32_t across_channels;
    float eps;
};

#define MLAYER_MVN_TEMPLATE template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT>

MLAYER_MVN_TEMPLATE
MVN<LayerT, MatT, OptionT, ParamDictT>::MVN() {
    this->one_blob_only = true;
    this->support_inplace = false;
    this->normalize_variance = 0;
    this->across_channels = 0;
    this->eps = 0.0001F;
}

MLAYER_MVN_TEMPLATE
std::int32_t MVN<LayerT, MatT, OptionT, ParamDictT>::load_param(const ParamDictT& pd) {
    this->normalize_variance = pd.get(0, 0);
    this->across_channels = pd.get(1, 0);
    this->eps = pd.get(2, 0.0001F);
    return 0;
}

MLAYER_MVN_TEMPLATE
std::int32_t MVN<LayerT, MatT, OptionT, ParamDictT>::forward(const MatT& bottom_blob, MatT& top_blob,
                                                            const OptionT& opt) const {
    const std::int32_t w = bottom_blob.w;
    const std::int32_t h = bottom_blob.h;
    const std::int32_t d = bottom_blob.d;
    const std::int32_t channels = bottom_blob.c;
    const std::int32_t dims = bottom_blob.dims;
    const std::size_t elemsize = bottom_blob.elemsize;
    const std::int32_t size = w * h * d;

    if (dims == 4) {
        top_blob.create(w, h, d, channels, elemsize, opt.blob_allocator);
    } else {
        top_blob.create(w, h, channels, elemsize, opt.blob_allocator);
    }
    if (top_blob.empty()) {
        return -100;
    }

    std::vector<float> sum(static_cast<std::size_t>(channels));
#pragma omp parallel for num_threads(opt.num_threads)
    for (std::int32_t q = 0; q < channels; ++q) {
        const float* ptr = bottom_blob.channel(q);
        float s = 0.0F;
        for (std::int32_t i = 0; i < size; ++i) {
            s += ptr[i];
        }
        sum[static_cast<std::size_t>(q)] = s;
    }

    float mean_all = 0.0F;
    if (this->across_channels != 0) {
        for (std::int32_t q = 0; q < channels; ++q) {
            mean_all += sum[static_cast<std::size_t>(q)];
        }
        mean_all = mean_all / (static_cast<float>(channels) * static_cast<float>(size));
    }

#pragma omp parallel for num_threads(opt.num_threads)
    for (std::int32_t q = 0; q < channels; ++q) {
        const float* ptr = bottom_blob.channel(q);
        float* outptr = top_blob.channel(q);
        const float mean = this->across_channels != 0 ? mean_all : sum[static_cast<std::size_t>(q)] / static_cast<float>(size);
        for (std::int32_t i = 0; i < size; ++i) {
            outptr[i] = ptr[i] - mean;
        }
    }

    if (this->normalize_variance == 0) {
        return 0;
    }

    std::vector<float> sqsum(static_cast<std::size_t>(channels));
#pragma omp parallel for num_threads(opt.num_threads)
    for (std::int32_t q = 0; q < channels; ++q) {
        const float* ptr = top_blob.channel(q);
        float s = 0.0F;
        for (std::int32_t i = 0; i < size; ++i) {
            s += ptr[i] * ptr[i];
        }
        sqsum[static_cast<std::size_t>(q)] = s;
    }

    float var_all = 0.0F;
    if (this->across_channels != 0) {
        for (std::int32_t q = 0; q < channels; ++q) {
            var_all += sqsum[static_cast<std::size_t>(q)];
        }
        var_all = var_all / (static_cast<float>(channels) * static_cast<float>(size));
    }

#pragma omp parallel for num_threads(opt.num_threads)
    for (std::int32_t q = 0; q < channels; ++q) {
        float* outptr = top_blob.channel(q);
        const float var = this->across_channels != 0 ? var_all : sqsum[static_cast<std::size_t>(q)] / static_cast<float>(size);
        const float a = 1.0F / std::sqrt(var + this->eps);
        for (std::int32_t i = 0; i < size; ++i) {
            outptr[i] *= a;
        }
    }
    return 0;
}

}  // namespace mlayer

#endif  // MLAYER_MVN_HPP
