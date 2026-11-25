// Copyright 2021 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_GLU_HPP
#define MLAYER_GLU_HPP

#include <cmath>
#include <cstdint>

namespace mlayer {

// Gated linear unit: split along `axis`, out = a * sigmoid(b).
template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT>
class GLU : public LayerT {
public:
    GLU() {
        this->one_blob_only = true;
        this->support_inplace = false;
        this->axis = 0;
    }

    std::int32_t load_param(const ParamDictT& pd) override {
        this->axis = pd.get(0, 0);
        return 0;
    }

    std::int32_t forward(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const override {
        const std::int32_t dims = bottom_blob.dims;
        const std::int32_t axis = this->axis < 0 ? dims + this->axis : this->axis;

        if (dims == 1) {
            const std::int32_t outw = bottom_blob.w / 2;
            top_blob.create(outw, 4u, opt.blob_allocator);
            if (top_blob.empty()) {
                return -100;
            }
            for (std::int32_t x = 0; x < outw; ++x) {
                const float a = bottom_blob[x];
                const float b = bottom_blob[x + outw];
                top_blob[x] = a / (1.0F + std::exp(-b));
            }
            return 0;
        }

        if (dims == 2) {
            const std::int32_t w = bottom_blob.w;
            const std::int32_t h = bottom_blob.h;
            if (axis == 1) {
                const std::int32_t outw = w / 2;
                top_blob.create(outw, h, 4u, opt.blob_allocator);
                if (top_blob.empty()) {
                    return -100;
                }
                for (std::int32_t y = 0; y < h; ++y) {
                    const float* row = bottom_blob.row(y);
                    float* outrow = top_blob.row(y);
                    for (std::int32_t x = 0; x < outw; ++x) {
                        outrow[x] = row[x] / (1.0F + std::exp(-row[x + outw]));
                    }
                }
                return 0;
            }
            // axis == 0: split rows
            const std::int32_t outh = h / 2;
            top_blob.create(w, outh, 4u, opt.blob_allocator);
            if (top_blob.empty()) {
                return -100;
            }
            for (std::int32_t y = 0; y < outh; ++y) {
                const float* row = bottom_blob.row(y);
                const float* grow = bottom_blob.row(y + outh);
                float* outrow = top_blob.row(y);
                for (std::int32_t x = 0; x < w; ++x) {
                    outrow[x] = row[x] / (1.0F + std::exp(-grow[x]));
                }
            }
            return 0;
        }

        const std::int32_t w = bottom_blob.w;
        const std::int32_t h = bottom_blob.h;
        const std::int32_t channels = bottom_blob.c;
        if (axis == 2) {
            const std::int32_t outw = w / 2;
            top_blob.create(outw, h, channels, 4u, opt.blob_allocator);
            if (top_blob.empty()) {
                return -100;
            }
            for (std::int32_t q = 0; q < channels; ++q) {
                const float* c = bottom_blob.channel(q);
                float* outc = top_blob.channel(q);
                for (std::int32_t y = 0; y < h; ++y) {
                    const float* row = c + y * w;
                    float* outrow = outc + y * outw;
                    for (std::int32_t x = 0; x < outw; ++x) {
                        outrow[x] = row[x] / (1.0F + std::exp(-row[x + outw]));
                    }
                }
            }
            return 0;
        }
        if (axis == 1) {
            const std::int32_t outh = h / 2;
            top_blob.create(w, outh, channels, 4u, opt.blob_allocator);
            if (top_blob.empty()) {
                return -100;
            }
            for (std::int32_t q = 0; q < channels; ++q) {
                const float* c = bottom_blob.channel(q);
                float* outc = top_blob.channel(q);
                for (std::int32_t y = 0; y < outh; ++y) {
                    const float* row = c + y * w;
                    const float* grow = c + (y + outh) * w;
                    float* outrow = outc + y * w;
                    for (std::int32_t x = 0; x < w; ++x) {
                        outrow[x] = row[x] / (1.0F + std::exp(-grow[x]));
                    }
                }
            }
            return 0;
        }
        // axis == 0: split channels
        const std::int32_t outc = channels / 2;
        top_blob.create(w, h, outc, 4u, opt.blob_allocator);
        if (top_blob.empty()) {
            return -100;
        }
        for (std::int32_t q = 0; q < outc; ++q) {
            const float* a = bottom_blob.channel(q);
            const float* b = bottom_blob.channel(q + outc);
            float* outc_ptr = top_blob.channel(q);
            for (std::int32_t i = 0; i < w * h; ++i) {
                outc_ptr[i] = a[i] / (1.0F + std::exp(-b[i]));
            }
        }
        return 0;
    }

public:
    std::int32_t axis;
};

}  // namespace mlayer

#endif  // MLAYER_GLU_HPP
