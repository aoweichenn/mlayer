// Copyright 2017 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_SPP_HPP
#define MLAYER_SPP_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace mlayer {

// Spatial pyramid pooling: 1x1, 2x2, 4x4 ... bins over the feature map. The
// output is (1 + 4 + ... + (2^p)^2) x 1 x channels x 2, matching ncnn.
template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, bool Validate = false>
class SPP : public LayerT {
public:
    SPP();

    std::int32_t load_param(const ParamDictT& pd) override;
    std::int32_t forward(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const override;

public:
    std::int32_t pooling_type;  // 0 = max, 1 = average
    std::int32_t pyramid_height;
};

#define MLAYER_SPP_TEMPLATE \
    template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, bool Validate>

MLAYER_SPP_TEMPLATE
SPP<LayerT, MatT, OptionT, ParamDictT, Validate>::SPP() {
    this->one_blob_only = true;
    this->support_inplace = false;
    this->pooling_type = 0;
    this->pyramid_height = 1;
}

MLAYER_SPP_TEMPLATE
std::int32_t SPP<LayerT, MatT, OptionT, ParamDictT, Validate>::load_param(const ParamDictT& pd) {
    this->pooling_type = pd.get(0, 0);
    this->pyramid_height = pd.get(1, 1);
    return 0;
}

MLAYER_SPP_TEMPLATE
std::int32_t SPP<LayerT, MatT, OptionT, ParamDictT, Validate>::forward(const MatT& bottom_blob, MatT& top_blob,
                                                                      const OptionT& opt) const {
    if (this->pyramid_height <= 0 || this->pyramid_height > 8) {
        return -1;
    }
    const std::size_t elemsize = bottom_blob.elemsize;
    const std::int32_t pyramid_num_bins = ((1 << (this->pyramid_height * 2)) - 1) / 3;
    top_blob.create(pyramid_num_bins, 1, 2, elemsize, opt.blob_allocator);
    if (top_blob.empty()) {
        return -100;
    }
    float* pyramid_ptr = static_cast<float*>(top_blob);

    for (std::int32_t p = 0; p < this->pyramid_height; ++p) {
        const std::int32_t w = bottom_blob.w;
        const std::int32_t h = bottom_blob.h;
        const std::int32_t channels = bottom_blob.c;
        const std::int32_t num_bins = 1 << p;

        const std::int32_t kernel_h = static_cast<std::int32_t>(std::ceil(static_cast<float>(h) / num_bins));
        const std::int32_t kernel_w = static_cast<std::int32_t>(std::ceil(static_cast<float>(w) / num_bins));
        const std::int32_t pad_h = (kernel_h * num_bins - h + 1) / 2;
        const std::int32_t pad_w = (kernel_w * num_bins - w + 1) / 2;
        const std::int32_t pw = w + 2 * pad_w;
        const std::int32_t ph = h + 2 * pad_h;
        const std::int32_t outw = num_bins;
        const std::int32_t maxk = kernel_h * kernel_w;

        for (std::int32_t q = 0; q < channels; ++q) {
            const float* src = bottom_blob.channel(q);
            float* outptr = pyramid_ptr + q * outw * outw;
            for (std::int32_t i = 0; i < outw; ++i) {
                for (std::int32_t j = 0; j < outw; ++j) {
                    float acc = this->pooling_type == 0 ? -3.402823466e+38F : 0.0F;
                    for (std::int32_t ky = 0; ky < kernel_h; ++ky) {
                        const std::int32_t sy = i * kernel_h + ky - pad_h;
                        if (sy < 0 || sy >= h) {
                            if (this->pooling_type == 0) {
                                // zero padding never wins a max against real values
                            }
                            continue;
                        }
                        for (std::int32_t kx = 0; kx < kernel_w; ++kx) {
                            const std::int32_t sx = j * kernel_w + kx - pad_w;
                            if (sx < 0 || sx >= w) {
                                continue;
                            }
                            const float v = src[sy * w + sx];
                            acc = this->pooling_type == 0 ? std::max(acc, v) : acc + v;
                        }
                    }
                    outptr[j] = this->pooling_type == 0 ? acc : acc / static_cast<float>(maxk);
                }
                outptr += outw;
            }
        }
        pyramid_ptr += channels * outw * outw;
        static_cast<void>(pw);
        static_cast<void>(ph);
    }

    // The second output channel mirrors the first (ncnn keeps the blob 2 deep).
    std::memcpy(static_cast<float*>(top_blob) + pyramid_num_bins, static_cast<const float*>(top_blob),
                static_cast<std::size_t>(pyramid_num_bins) * sizeof(float));
    return 0;
}

}  // namespace mlayer

#endif  // MLAYER_SPP_HPP
