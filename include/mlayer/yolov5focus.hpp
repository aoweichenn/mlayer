// Copyright 2021 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_YOLOV5FOCUS_HPP
#define MLAYER_YOLOV5FOCUS_HPP

#include <cstdint>

namespace mlayer {

// Optional custom layer used by legacy YOLOv5 params (space-to-depth focus).
// It is not part of the ncnn core layer set; applications register it by name.
template<typename LayerT, typename MatT, typename OptionT>
class YoloV5Focus : public LayerT {
public:
    YoloV5Focus() {
        this->one_blob_only = true;
    }

    std::int32_t forward(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const override {
        const std::int32_t w = bottom_blob.w;
        const std::int32_t h = bottom_blob.h;
        const std::int32_t channels = bottom_blob.c;

        const std::int32_t outw = w / 2;
        const std::int32_t outh = h / 2;
        const std::int32_t outc = channels * 4;

        top_blob.create(outw, outh, outc, 4u, 1, opt.blob_allocator);
        if (top_blob.empty()) {
            return -100;
        }

#pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t p = 0; p < outc; ++p) {
            const float* ptr =
                bottom_blob.channel(p % channels).row((p / channels) % 2) + ((p / channels) / 2);
            float* outptr = top_blob.channel(p);

            for (std::int32_t i = 0; i < outh; ++i) {
                for (std::int32_t j = 0; j < outw; ++j) {
                    *outptr = *ptr;

                    outptr += 1;
                    ptr += 2;
                }

                ptr += w;
            }
        }

        return 0;
    }
};

}  // namespace mlayer

#endif  // MLAYER_YOLOV5FOCUS_HPP
