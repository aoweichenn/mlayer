// Copyright 2024 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_FLIP_HPP
#define MLAYER_FLIP_HPP

#include <cstdint>

namespace mlayer {

// Reverse the tensor along the axes listed in the `axes` array
// (dims 1: w, dims 2: h/w, dims 3: c/h/w, dims 4: c/d/h/w).
template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT>
class Flip : public LayerT {
public:
    Flip() {
        this->one_blob_only = true;
        this->support_inplace = false;
    }

    std::int32_t load_param(const ParamDictT& pd) override {
        this->axes = pd.get(0, MatT());
        return 0;
    }

    std::int32_t forward(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const override {
        if (this->axes.empty()) {
            top_blob = bottom_blob;
            return 0;
        }

        const std::int32_t dims = bottom_blob.dims;
        const std::int32_t w = bottom_blob.w;
        const std::int32_t h = bottom_blob.h;
        const std::int32_t d = bottom_blob.d;
        const std::int32_t channels = bottom_blob.c;
        const std::size_t elemsize = bottom_blob.elemsize;

        bool flip_w = false;
        bool flip_h = false;
        bool flip_d = false;
        bool flip_c = false;
        const auto* axes_ptr = static_cast<const std::int32_t*>(this->axes.data);
        for (std::int32_t i = 0; i < this->axes.w; ++i) {
            std::int32_t axis = axes_ptr[i];
            if (axis < 0) {
                axis += dims;
            }
            if (dims == 1) {
                flip_w = true;
            } else if (dims == 2) {
                flip_h = flip_h || axis == 0;
                flip_w = flip_w || axis == 1;
            } else if (dims == 3) {
                flip_c = flip_c || axis == 0;
                flip_h = flip_h || axis == 1;
                flip_w = flip_w || axis == 2;
            } else if (dims == 4) {
                flip_c = flip_c || axis == 0;
                flip_d = flip_d || axis == 1;
                flip_h = flip_h || axis == 2;
                flip_w = flip_w || axis == 3;
            }
        }
        if (dims == 1) {
            flip_w = true;
        }

        if (dims == 1) {
            top_blob.create(w, elemsize, opt.blob_allocator);
        } else if (dims == 2) {
            top_blob.create(w, h, elemsize, opt.blob_allocator);
        } else if (dims == 3) {
            top_blob.create(w, h, channels, elemsize, opt.blob_allocator);
        } else {
            top_blob.create(w, h, d, channels, elemsize, opt.blob_allocator);
        }
        if (top_blob.empty()) {
            return -100;
        }

#pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t q = 0; q < channels; ++q) {
            const std::int32_t qs = flip_c ? channels - 1 - q : q;
            const MatT src_c = bottom_blob.channel(qs);
            MatT dst_c = top_blob.channel(q);
            for (std::int32_t z = 0; z < (dims == 4 ? d : 1); ++z) {
                const std::int32_t zs = flip_d ? d - 1 - z : z;
                const MatT src_d = dims == 4 ? src_c.depth(zs) : src_c;
                MatT dst_d = dims == 4 ? dst_c.depth(z) : dst_c;
                for (std::int32_t y = 0; y < h; ++y) {
                    const std::int32_t ys = flip_h ? h - 1 - y : y;
                    const float* srow = src_d.row(ys);
                    float* drow = dst_d.row(y);
                    for (std::int32_t x = 0; x < w; ++x) {
                        drow[x] = srow[flip_w ? w - 1 - x : x];
                    }
                }
            }
        }
        return 0;
    }

public:
    MatT axes;
};

}  // namespace mlayer

#endif  // MLAYER_FLIP_HPP
