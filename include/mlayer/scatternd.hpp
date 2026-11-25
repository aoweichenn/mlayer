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

        if (data.dims != 1) {
            return -100;
        }
        const std::int32_t count = indices.w * indices.h;
        if (updates.w != count || updates.h * updates.c != 1) {
            return -100;
        }

        MatT& top_blob = top_blobs[0];
        top_blob.create(data.w, 4u, opt.blob_allocator);
        if (top_blob.empty()) {
            return -100;
        }
        std::memcpy(top_blob.data, data.data, static_cast<std::size_t>(data.w) * sizeof(float));

        for (std::int32_t j = 0; j < count; ++j) {
            const auto index = static_cast<std::int32_t>(std::lround(indices[j]));
            if (index < 0 || index >= data.w) {
                return -100;
            }
            top_blob[index] = updates[j];
        }
        return 0;
    }
};

}  // namespace mlayer

#endif  // MLAYER_SCATTERND_HPP
