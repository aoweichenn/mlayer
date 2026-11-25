// Copyright 2021 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_ROIALIGN_HPP
#define MLAYER_ROIALIGN_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace mlayer {
namespace detail {

inline float roialign_bilinear(const float* ptr, std::int32_t w, std::int32_t h, float x, float y) {
    std::int32_t x0 = static_cast<std::int32_t>(x);
    std::int32_t x1 = x0 + 1;
    std::int32_t y0 = static_cast<std::int32_t>(y);
    std::int32_t y1 = y0 + 1;

    float a0 = static_cast<float>(x1) - x;
    float a1 = x - static_cast<float>(x0);
    float b0 = static_cast<float>(y1) - y;
    float b1 = y - static_cast<float>(y0);

    if (x1 >= w) {
        x1 = w - 1;
        a0 = 1.0F;
        a1 = 0.0F;
    }
    if (y1 >= h) {
        y1 = h - 1;
        b0 = 1.0F;
        b1 = 0.0F;
    }

    const float r0 = ptr[y0 * w + x0] * a0 + ptr[y0 * w + x1] * a1;
    const float r1 = ptr[y1 * w + x0] * a0 + ptr[y1 * w + x1] * a1;
    return r0 * b0 + r1 * b1;
}

}  // namespace detail

// ROIAlign (Mask R-CNN): bilinear sampling of one ROI into pooled_w x pooled_h.
// version 0 = original, version 1 = detectron2.
template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, bool Validate = false>
class ROIAlign : public LayerT {
public:
    ROIAlign();

    std::int32_t load_param(const ParamDictT& pd) override;
    std::int32_t forward(const std::vector<MatT>& bottom_blobs, std::vector<MatT>& top_blobs,
                         const OptionT& opt) const override;

public:
    std::int32_t pooled_width;
    std::int32_t pooled_height;
    float spatial_scale;
    std::int32_t sampling_ratio;
    bool aligned;
    std::int32_t version;
};

#define MLAYER_ROIALIGN_TEMPLATE \
    template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, bool Validate>

MLAYER_ROIALIGN_TEMPLATE
ROIAlign<LayerT, MatT, OptionT, ParamDictT, Validate>::ROIAlign() {
    this->one_blob_only = false;
    this->support_inplace = false;
    this->pooled_width = 0;
    this->pooled_height = 0;
    this->spatial_scale = 1.0F;
    this->sampling_ratio = 0;
    this->aligned = false;
    this->version = 0;
}

MLAYER_ROIALIGN_TEMPLATE
std::int32_t ROIAlign<LayerT, MatT, OptionT, ParamDictT, Validate>::load_param(const ParamDictT& pd) {
    this->pooled_width = pd.get(0, 0);
    this->pooled_height = pd.get(1, 0);
    this->spatial_scale = pd.get(2, 1.0F);
    this->sampling_ratio = pd.get(3, 0);
    this->aligned = pd.get(4, 0) != 0;
    this->version = pd.get(5, 0);
    return 0;
}

MLAYER_ROIALIGN_TEMPLATE
std::int32_t ROIAlign<LayerT, MatT, OptionT, ParamDictT, Validate>::forward(const std::vector<MatT>& bottom_blobs,
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
    float roi_x1 = roi_ptr[0] * this->spatial_scale;
    float roi_y1 = roi_ptr[1] * this->spatial_scale;
    float roi_x2 = roi_ptr[2] * this->spatial_scale;
    float roi_y2 = roi_ptr[3] * this->spatial_scale;
    if (this->aligned) {
        roi_x1 -= 0.5F;
        roi_y1 -= 0.5F;
        roi_x2 -= 0.5F;
        roi_y2 -= 0.5F;
    }

    float roi_w = roi_x2 - roi_x1;
    float roi_h = roi_y2 - roi_y1;
    if (!this->aligned) {
        roi_w = std::max(roi_w, 1.0F);
        roi_h = std::max(roi_h, 1.0F);
    }

    const float bin_size_w = roi_w / static_cast<float>(this->pooled_width);
    const float bin_size_h = roi_h / static_cast<float>(this->pooled_height);

    if (this->version == 0) {
#pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t q = 0; q < channels; ++q) {
            const float* ptr = bottom_blob.channel(q);
            float* outptr = top_blob.channel(q);
            for (std::int32_t ph = 0; ph < this->pooled_height; ++ph) {
                for (std::int32_t pw = 0; pw < this->pooled_width; ++pw) {
                    float hstart = roi_y1 + ph * bin_size_h;
                    float wstart = roi_x1 + pw * bin_size_w;
                    float hend = roi_y1 + (ph + 1) * bin_size_h;
                    float wend = roi_x1 + (pw + 1) * bin_size_w;

                    hstart = std::min(std::max(hstart, 0.0F), static_cast<float>(h));
                    wstart = std::min(std::max(wstart, 0.0F), static_cast<float>(w));
                    hend = std::min(std::max(hend, 0.0F), static_cast<float>(h));
                    wend = std::min(std::max(wend, 0.0F), static_cast<float>(w));

                    const std::int32_t bin_grid_h =
                        this->sampling_ratio > 0 ? this->sampling_ratio : static_cast<std::int32_t>(std::ceil(hend - hstart));
                    const std::int32_t bin_grid_w =
                        this->sampling_ratio > 0 ? this->sampling_ratio : static_cast<std::int32_t>(std::ceil(wend - wstart));

                    const bool is_empty = (hend <= hstart) || (wend <= wstart);
                    const std::int32_t area = bin_grid_h * bin_grid_w;

                    float sum = 0.0F;
                    for (std::int32_t by = 0; by < bin_grid_h; ++by) {
                        const float y = hstart + (by + 0.5F) * bin_size_h / static_cast<float>(bin_grid_h);
                        for (std::int32_t bx = 0; bx < bin_grid_w; ++bx) {
                            const float x = wstart + (bx + 0.5F) * bin_size_w / static_cast<float>(bin_grid_w);
                            sum += detail::roialign_bilinear(ptr, w, h, x, y);
                        }
                    }
                    outptr[pw] = is_empty ? 0.0F : (sum / static_cast<float>(area));
                }
                outptr += this->pooled_width;
            }
        }
        return 0;
    }

    // version 1 (detectron2)
    const std::int32_t roi_bin_grid_h =
        this->sampling_ratio > 0 ? this->sampling_ratio : static_cast<std::int32_t>(std::ceil(roi_h / this->pooled_height));
    const std::int32_t roi_bin_grid_w =
        this->sampling_ratio > 0 ? this->sampling_ratio : static_cast<std::int32_t>(std::ceil(roi_w / this->pooled_width));
    const float count = static_cast<float>(std::max(roi_bin_grid_h * roi_bin_grid_w, 1));

#pragma omp parallel for num_threads(opt.num_threads)
    for (std::int32_t q = 0; q < channels; ++q) {
        const float* ptr = bottom_blob.channel(q);
        float* outptr = top_blob.channel(q);
        for (std::int32_t ph = 0; ph < this->pooled_height; ++ph) {
            for (std::int32_t pw = 0; pw < this->pooled_width; ++pw) {
                float sum = 0.0F;
                for (std::int32_t by = 0; by < roi_bin_grid_h; ++by) {
                    float y = roi_y1 + ph * bin_size_h + (by + 0.5F) * bin_size_h / static_cast<float>(roi_bin_grid_h);
                    for (std::int32_t bx = 0; bx < roi_bin_grid_w; ++bx) {
                        float x = roi_x1 + pw * bin_size_w +
                                  (bx + 0.5F) * bin_size_w / static_cast<float>(roi_bin_grid_w);
                        if (y < -1.0F || y > static_cast<float>(h) || x < -1.0F || x > static_cast<float>(w)) {
                            continue;
                        }
                        if (y <= 0.0F) {
                            y = 0.0F;
                        }
                        if (x <= 0.0F) {
                            x = 0.0F;
                        }
                        sum += detail::roialign_bilinear(ptr, w, h, x, y);
                    }
                }
                outptr[pw] = sum / count;
            }
            outptr += this->pooled_width;
        }
    }
    return 0;
}

}  // namespace mlayer

#endif  // MLAYER_ROIALIGN_HPP
