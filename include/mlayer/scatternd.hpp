// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_SCATTERND_HPP
#define MLAYER_SCATTERND_HPP

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace mlayer {

// ONNX `ScatterND` restricted to the pattern Kokoro's iSTFT overlap-add uses:
// `data` is 1-D, `indices` is [n, 1] (or [n]) of flat positions and `updates`
// has n values. out = data; out[indices[j]] = updates[j].
//
// The copy keeps `data` intact (out-of-place), matching ONNX semantics.
template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT>
class ScatterND : public LayerT {
public:
    ScatterND() {
        this->one_blob_only = false;
        this->support_inplace = false;
    }

    std::int32_t load_param(const ParamDictT& /*pd*/) override { return 0; }

    std::int32_t forward(const std::vector<MatT>& bottom_blobs, std::vector<MatT>& top_blobs,
                         const OptionT& opt) const override {
        if (bottom_blobs.size() < 3 || top_blobs.empty()) {
            return -100;
        }
        const MatT& data = bottom_blobs[0];
        const MatT& indices = bottom_blobs[1];
        const MatT& updates = bottom_blobs[2];

        // Accepted shapes:
        //   data (N)        + indices (n)      -> flat positions
        //   data (W, H)     + indices (2, n)   -> (row, col) pairs, row-major
        const bool flat = data.dims == 1 || (data.dims == 3 && data.h == 1 && data.c == 1);
        const bool ranked = (data.dims == 2 || (data.dims == 3 && data.c == 1)) && indices.w == 2;
        if (!flat && !ranked) {
            return -100;
        }
        const std::int32_t count = ranked ? indices.h * indices.c : indices.w * indices.h;
        if (updates.w != count || updates.h * updates.c != 1) {
            return -100;
        }

        MatT& top_blob = top_blobs[0];
        const std::int32_t total = flat ? data.w : data.w * data.h;
        top_blob.create(data.w, data.h, 4u, opt.blob_allocator);
        if (top_blob.empty()) {
            return -100;
        }
        std::memcpy(top_blob.data, data.data, static_cast<std::size_t>(total) * sizeof(float));

        for (std::int32_t j = 0; j < count; ++j) {
            std::int32_t index = 0;
            if (flat) {
                index = static_cast<std::int32_t>(std::lround(indices[j]));
            } else {
                const std::int32_t row = static_cast<std::int32_t>(std::lround(indices.row(j)[0]));
                const std::int32_t col = static_cast<std::int32_t>(std::lround(indices.row(j)[1]));
                if (row < 0 || row >= data.h) {
                    return -100;
                }
                index = row * data.w + col;
            }
            if (index < 0 || index >= total) {
                return -100;
            }
            top_blob[index] = updates[j];
        }
        return 0;
    }
};

}  // namespace mlayer

#endif  // MLAYER_SCATTERND_HPP
