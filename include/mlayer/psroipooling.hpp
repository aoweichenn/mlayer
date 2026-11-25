// Copyright 2018 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_PSROIPOOLING_HPP
#define MLAYER_PSROIPOOLING_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace mlayer {

template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, bool Validate = false>
class PSROIPooling : public LayerT {
public:
    PSROIPooling() {
        this->one_blob_only = false;
        this->support_inplace = false;
    }

    std::int32_t load_param(const ParamDictT& pd) override {
        this->pooled_width = pd.get(0, 7);
        this->pooled_height = pd.get(1, 7);
        this->spatial_scale = pd.get(2, 0.0625F);
        this->output_dim = pd.get(3, 0);

        if constexpr (Validate) {
            if (this->pooled_width <= 0 || this->pooled_height <= 0 || this->output_dim <= 0) {
                return -1;
            }
        }
        return 0;
    }

    std::int32_t forward(const std::vector<MatT>& bottom_blobs, std::vector<MatT>& top_blobs,
                         const OptionT& opt) const override {
        if (bottom_blobs.size() < 2 || top_blobs.empty()) {
            return -1;
        }
        const MatT& bottom_blob = bottom_blobs[0];
        const MatT& roi_blob = bottom_blobs[1];

        const std::int32_t w = bottom_blob.w;
        const std::int32_t h = bottom_blob.h;
        const std::int32_t channels = bottom_blob.c;
        const std::size_t elemsize = bottom_blob.elemsize;

        if (channels != this->output_dim * this->pooled_width * this->pooled_height) {
            return -1;
        }

        MatT& top_blob = top_blobs[0];
        top_blob.create(this->pooled_width, this->pooled_height, this->output_dim, elemsize, opt.blob_allocator);
        if (top_blob.empty()) {
            return -100;
        }

        const float* roi_ptr = roi_blob;
        const float roi_x1 = std::round(roi_ptr[0]) * this->spatial_scale;
        const float roi_y1 = std::round(roi_ptr[1]) * this->spatial_scale;
        const float roi_x2 = std::round(roi_ptr[2] + 1.0F) * this->spatial_scale;
        const float roi_y2 = std::round(roi_ptr[3] + 1.0F) * this->spatial_scale;

        const float roi_w = std::max(roi_x2 - roi_x1, 0.1F);
        const float roi_h = std::max(roi_y2 - roi_y1, 0.1F);
        const float bin_size_w = roi_w / static_cast<float>(this->pooled_width);
        const float bin_size_h = roi_h / static_cast<float>(this->pooled_height);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t q = 0; q < this->output_dim; ++q) {
            float* outptr = top_blob.channel(q);
            for (std::int32_t ph = 0; ph < this->pooled_height; ++ph) {
                for (std::int32_t pw = 0; pw < this->pooled_width; ++pw) {
                    const float* ptr = bottom_blob.channel((q * this->pooled_height + ph) *
                                                               this->pooled_width +
                                                           pw);

                    std::int32_t hstart = static_cast<std::int32_t>(
                        std::floor(roi_y1 + static_cast<float>(ph) * bin_size_h));
                    std::int32_t wstart = static_cast<std::int32_t>(
                        std::floor(roi_x1 + static_cast<float>(pw) * bin_size_w));
                    std::int32_t hend = static_cast<std::int32_t>(
                        std::ceil(roi_y1 + static_cast<float>(ph + 1) * bin_size_h));
                    std::int32_t wend = static_cast<std::int32_t>(
                        std::ceil(roi_x1 + static_cast<float>(pw + 1) * bin_size_w));

                    hstart = std::min(std::max(hstart, 0), h);
                    wstart = std::min(std::max(wstart, 0), w);
                    hend = std::min(std::max(hend, 0), h);
                    wend = std::min(std::max(wend, 0), w);

                    const bool is_empty = (hend <= hstart) || (wend <= wstart);
                    const std::int32_t area = (hend - hstart) * (wend - wstart);

                    float sum = 0.0F;
                    for (std::int32_t y = hstart; y < hend; ++y) {
                        for (std::int32_t x = wstart; x < wend; ++x) {
                            sum += ptr[y * w + x];
                        }
                    }
                    outptr[pw] = is_empty ? 0.0F : sum / static_cast<float>(area);
                }
                outptr += this->pooled_width;
            }
        }
        return 0;
    }

public:
    std::int32_t pooled_width = 7;
    std::int32_t pooled_height = 7;
    float spatial_scale = 0.0625F;
    std::int32_t output_dim = 0;
};

}  // namespace mlayer

#endif  // MLAYER_PSROIPOOLING_HPP
