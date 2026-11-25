// Copyright 2017 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_INTERP_HPP
#define MLAYER_INTERP_HPP

#include <cmath>
#include <cstdint>

namespace mlayer {

// Minimal CPU interp: nearest (type 1) and bilinear (type 2). Other resize
// modes are rejected so callers can add them explicitly later.
template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT>
class Interp : public LayerT {
public:
    Interp() {
        this->one_blob_only = true;
        this->support_inplace = false;
        this->resize_type = 1;
        this->height_scale = 1.0F;
        this->width_scale = 1.0F;
        this->output_height = 0;
        this->output_width = 0;
    }

    std::int32_t load_param(const ParamDictT& pd) override {
        this->resize_type = pd.get(0, 1);
        this->height_scale = pd.get(1, 1.0F);
        this->width_scale = pd.get(2, 1.0F);
        this->output_height = pd.get(3, 0);
        this->output_width = pd.get(4, 0);
        return 0;
    }

    std::int32_t forward(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const override {
        const std::int32_t w = bottom_blob.w;
        const std::int32_t h = bottom_blob.h;
        const std::int32_t c = bottom_blob.c;

        const std::int32_t outw = this->output_width > 0
                                      ? this->output_width
                                      : static_cast<std::int32_t>(static_cast<float>(w) * this->width_scale);
        const std::int32_t outh = this->output_height > 0
                                      ? this->output_height
                                      : static_cast<std::int32_t>(static_cast<float>(h) * this->height_scale);
        if (outw <= 0 || outh <= 0) {
            return -1;
        }

        top_blob.create(outw, outh, c, bottom_blob.elemsize, opt.blob_allocator);
        if (top_blob.empty()) {
            return -100;
        }

        if (this->resize_type == 1) {
#pragma omp parallel for num_threads(opt.num_threads)
            for (std::int32_t q = 0; q < c; ++q) {
                const float* src = bottom_blob.channel(q);
                float* dst = top_blob.channel(q);
                for (std::int32_t y = 0; y < outh; ++y) {
                    const std::int32_t sy = y * h / outh;
                    for (std::int32_t x = 0; x < outw; ++x) {
                        const std::int32_t sx = x * w / outw;
                        dst[y * outw + x] = src[sy * w + sx];
                    }
                }
            }
            return 0;
        }

        if (this->resize_type == 2) {
#pragma omp parallel for num_threads(opt.num_threads)
            for (std::int32_t q = 0; q < c; ++q) {
                const float* src = bottom_blob.channel(q);
                float* dst = top_blob.channel(q);
                const float scale_x = static_cast<float>(w) / static_cast<float>(outw);
                const float scale_y = static_cast<float>(h) / static_cast<float>(outh);
                for (std::int32_t y = 0; y < outh; ++y) {
                    const float fy = (static_cast<float>(y) + 0.5F) * scale_y - 0.5F;
                    const std::int32_t y0 = fy < 0.0F ? 0 : static_cast<std::int32_t>(fy);
                    const std::int32_t y1 = y0 + 1 < h ? y0 + 1 : h - 1;
                    const float dy = fy - static_cast<float>(y0);
                    for (std::int32_t x = 0; x < outw; ++x) {
                        const float fx = (static_cast<float>(x) + 0.5F) * scale_x - 0.5F;
                        const std::int32_t x0 = fx < 0.0F ? 0 : static_cast<std::int32_t>(fx);
                        const std::int32_t x1 = x0 + 1 < w ? x0 + 1 : w - 1;
                        const float dx = fx - static_cast<float>(x0);

                        const float v00 = src[y0 * w + x0];
                        const float v01 = src[y0 * w + x1];
                        const float v10 = src[y1 * w + x0];
                        const float v11 = src[y1 * w + x1];
                        const float v0 = v00 + (v01 - v00) * dx;
                        const float v1 = v10 + (v11 - v10) * dx;
                        dst[y * outw + x] = v0 + (v1 - v0) * dy;
                    }
                }
            }
            return 0;
        }

        return -1;
    }

public:
    std::int32_t resize_type;
    float height_scale;
    float width_scale;
    std::int32_t output_height;
    std::int32_t output_width;
};

}  // namespace mlayer

#endif  // MLAYER_INTERP_HPP
