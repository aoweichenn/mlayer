// Copyright 2017 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_ROIPOOLING_HPP
#define MLAYER_ROIPOOLING_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace mlayer {

// ROI max pooling for Faster-RCNN: one ROI per blob, pooled on the scaled
// feature map. Mirrors ncnn's ROI start/end rounding and empty-bin handling.
template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT>
class ROIPooling : public LayerT {
public:
    ROIPooling();

    std::int32_t load_param(const ParamDictT& pd) override;
    std::int32_t forward(const std::vector<MatT>& bottom_blobs, std::vector<MatT>& top_blobs,
                         const OptionT& opt) const override;

public:
    std::int32_t pooled_width;
    std::int32_t pooled_height;
    float spatial_scale;
};

#define MLAYER_ROIPOOLING_TEMPLATE template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT>

MLAYER_ROIPOOLING_TEMPLATE
ROIPooling<LayerT, MatT, OptionT, ParamDictT>::ROIPooling() {
    this->one_blob_only = false;
    this->support_inplace = false;
    this->pooled_width = 0;
    this->pooled_height = 0;
    this->spatial_scale = 1.0F;
}

MLAYER_ROIPOOLING_TEMPLATE
std::int32_t ROIPooling<LayerT, MatT, OptionT, ParamDictT>::load_param(const ParamDictT& pd) {
    this->pooled_width = pd.get(0, 0);
    this->pooled_height = pd.get(1, 0);
    this->spatial_scale = pd.get(2, 1.0F);
    return 0;
}

MLAYER_ROIPOOLING_TEMPLATE
std::int32_t ROIPooling<LayerT, MatT, OptionT, ParamDictT>::forward(const std::vector<MatT>& bottom_blobs,
                                                                   std::vector<MatT>& top_blobs,
                                                                   const OptionT& opt) const {
    if (bottom_blobs.size() < 2 || top_blobs.empty()) {
        return -1;
    }
    const MatT& bottom_blob = bottom_blobs[0];
    const MatT& roi_blob = bottom_blobs[1];
    const std::int32_t w = bottom_blob.w;
    const std::int32_t h = bottom_blob.h;
    const std::int32_t channels = bottom_blob.c;
    const std::size_t elemsize = bottom_blob.elemsize;

    MatT& top_blob = top_blobs[0];
    top_blob.create(this->pooled_width, this->pooled_height, channels, elemsize, opt.blob_allocator);
    if (top_blob.empty()) {
        return -100;
    }

    const float* roi_ptr = roi_blob;
    const std::int32_t roi_x1 = static_cast<std::int32_t>(std::round(roi_ptr[0] * this->spatial_scale));
    const std::int32_t roi_y1 = static_cast<std::int32_t>(std::round(roi_ptr[1] * this->spatial_scale));
    const std::int32_t roi_x2 = static_cast<std::int32_t>(std::round(roi_ptr[2] * this->spatial_scale));
    const std::int32_t roi_y2 = static_cast<std::int32_t>(std::round(roi_ptr[3] * this->spatial_scale));

    const std::int32_t roi_w = std::max(roi_x2 - roi_x1 + 1, 1);
    const std::int32_t roi_h = std::max(roi_y2 - roi_y1 + 1, 1);

    const float bin_size_w = static_cast<float>(roi_w) / static_cast<float>(this->pooled_width);
    const float bin_size_h = static_cast<float>(roi_h) / static_cast<float>(this->pooled_height);

#pragma omp parallel for num_threads(opt.num_threads)
    for (std::int32_t q = 0; q < channels; ++q) {
        const float* ptr = bottom_blob.channel(q);
        float* outptr = top_blob.channel(q);
        for (std::int32_t ph = 0; ph < this->pooled_height; ++ph) {
            for (std::int32_t pw = 0; pw < this->pooled_width; ++pw) {
                const float phf = static_cast<float>(ph);
                const float pwf = static_cast<float>(pw);
                std::int32_t hstart =
                    static_cast<std::int32_t>(static_cast<float>(roi_y1) + std::floor(phf * bin_size_h));
                std::int32_t wstart =
                    static_cast<std::int32_t>(static_cast<float>(roi_x1) + std::floor(pwf * bin_size_w));
                std::int32_t hend =
                    static_cast<std::int32_t>(static_cast<float>(roi_y1) + std::ceil((phf + 1.0F) * bin_size_h));
                std::int32_t wend =
                    static_cast<std::int32_t>(static_cast<float>(roi_x1) + std::ceil((pwf + 1.0F) * bin_size_w));

                hstart = std::min(std::max(hstart, 0), h);
                wstart = std::min(std::max(wstart, 0), w);
                hend = std::min(std::max(hend, 0), h);
                wend = std::min(std::max(wend, 0), w);

                const bool is_empty = (hend <= hstart) || (wend <= wstart);
                float max = is_empty ? 0.0F : ptr[hstart * w + wstart];
                for (std::int32_t y = hstart; y < hend; ++y) {
                    for (std::int32_t x = wstart; x < wend; ++x) {
                        max = std::max(max, ptr[y * w + x]);
                    }
                }
                outptr[pw] = max;
            }
            outptr += this->pooled_width;
        }
    }

    return 0;
}

}  // namespace mlayer

#endif  // MLAYER_ROIPOOLING_HPP
