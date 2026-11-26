// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_GATHER_HPP
#define MLAYER_GATHER_HPP

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace mlayer {

// Index gather along one axis (ONNX `Gather`).
//
//   axis 0: out[i, ...] = data[index[i], ...]
//   axis 1: out[..., i] = data[..., index[i]]   (2-D data only)
//
// Indices come as a float blob (mnet blobs are float); they are rounded.
template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT>
class Gather : public LayerT {
public:
    Gather() {
        this->one_blob_only = false;
        this->support_inplace = false;
        this->axis = 0;
    }

    std::int32_t load_param(const ParamDictT& pd) override {
        this->axis = pd.get(0, 0);
        return 0;
    }

    std::int32_t forward(const std::vector<MatT>& bottom_blobs, std::vector<MatT>& top_blobs,
                         const OptionT& opt) const override {
        if (bottom_blobs.size() < 2 || top_blobs.empty()) {
            return -100;
        }
        const MatT& data = bottom_blobs[0];
        const MatT& indices = bottom_blobs[1];
        const std::int32_t count = indices.w * indices.h;

        MatT& top_blob = top_blobs[0];

        // Axes beyond the effective rank are implicit singletons: the emitter
        // maps ONNX [N,C,L] to (w,h,c) and frequently squeezes the leading
        // dims away, so `Gather(axis=1, index=0)` on a 1-D blob must behave
        // as the identity rather than fail.
        std::int32_t effective_rank = data.dims;
        if (data.dims == 3) {
            effective_rank = (data.h == 1 && data.c == 1) ? 1 : (data.c == 1 ? 2 : 3);
        }
        if (this->axis >= effective_rank) {
            for (std::int32_t i = 0; i < count; ++i) {
                if (std::lround(indices[i]) != 0) {
                    return -100;
                }
            }
            top_blob = data;
            return 0;
        }

        const bool one_d = data.dims == 1 || (data.dims == 3 && data.h == 1 && data.c == 1);
        const bool two_d = data.dims == 2 || (data.dims == 3 && data.c == 1);
        if (this->axis == 0) {
            // Prefer the row (h) interpretation for a squeezed 3-D singleton
            // blob: (w,1,1) is often an ONNX [1, w] row, not a 1-D vector.
            if (two_d) {
                top_blob.create(data.w, count, 4u, opt.blob_allocator);
                if (top_blob.empty()) {
                    return -100;
                }
                for (std::int32_t i = 0; i < count; ++i) {
                    const auto index = static_cast<std::int32_t>(std::lround(indices[i]));
                    if (index < 0 || index >= data.h) {
                        return -100;
                    }
                    std::memcpy(top_blob.row(i), data.row(index), static_cast<std::size_t>(data.w) * sizeof(float));
                }
                return 0;
            }
            if (one_d) {
                top_blob.create(count, 4u, opt.blob_allocator);
                if (top_blob.empty()) {
                    return -100;
                }
                for (std::int32_t i = 0; i < count; ++i) {
                    const auto index = static_cast<std::int32_t>(std::lround(indices[i]));
                    if (index < 0 || index >= data.w) {
                        return -100;
                    }
                    top_blob[i] = data[index];
                }
                return 0;
            }
            if (data.dims == 3) {
                top_blob.create(data.w, data.h, count, 4u, opt.blob_allocator);
                if (top_blob.empty()) {
                    return -100;
                }
                const std::size_t plane = static_cast<std::size_t>(data.w) * static_cast<std::size_t>(data.h);
                for (std::int32_t i = 0; i < count; ++i) {
                    const auto index = static_cast<std::int32_t>(std::lround(indices[i]));
                    if (index < 0 || index >= data.c) {
                        return -100;
                    }
                    std::memcpy(top_blob.channel(i), data.channel(index), plane * sizeof(float));
                }
                return 0;
            }
            return -100;
        }

        // Width gather (ONNX axis 3 of [N,C,H,W]): pick columns of every row.
        if (this->axis == 3) {
            const std::int32_t rows = data.h;
            const std::int32_t planes = data.c;
            top_blob.create(count, rows, planes, 4u, opt.blob_allocator);
            if (top_blob.empty()) {
                return -100;
            }
            for (std::int32_t q = 0; q < planes; ++q) {
                for (std::int32_t y = 0; y < rows; ++y) {
                    const float* src = data.channel(q).row(y);
                    float* dst = top_blob.channel(q).row(y);
                    for (std::int32_t i = 0; i < count; ++i) {
                        const auto index = static_cast<std::int32_t>(std::lround(indices[i]));
                        if (index < 0 || index >= data.w) {
                            return -100;
                        }
                        dst[i] = src[index];
                    }
                }
            }
            return 0;
        }
        if (this->axis == 1 && data.dims == 3 && data.c > 1) {
            top_blob.create(data.w, count, data.c, 4u, opt.blob_allocator);
            if (top_blob.empty()) {
                return -100;
            }
            for (std::int32_t q = 0; q < data.c; ++q) {
                for (std::int32_t i = 0; i < count; ++i) {
                    const auto index = static_cast<std::int32_t>(std::lround(indices[i]));
                    if (index < 0 || index >= data.h) {
                        return -100;
                    }
                    std::memcpy(top_blob.channel(q).row(i), data.channel(q).row(index),
                                static_cast<std::size_t>(data.w) * sizeof(float));
                }
            }
            return 0;
        }
        if (this->axis == 1 && two_d) {
            top_blob.create(count, data.h, 4u, opt.blob_allocator);
            if (top_blob.empty()) {
                return -100;
            }
            for (std::int32_t y = 0; y < data.h; ++y) {
                const float* src = data.row(y);
                float* dst = top_blob.row(y);
                for (std::int32_t i = 0; i < count; ++i) {
                    const auto index = static_cast<std::int32_t>(std::lround(indices[i]));
                    if (index < 0 || index >= data.w) {
                        return -100;
                    }
                    dst[i] = src[index];
                }
            }
            return 0;
        }
        return -100;
    }

    std::int32_t axis;
};

}  // namespace mlayer

#endif  // MLAYER_GATHER_HPP
