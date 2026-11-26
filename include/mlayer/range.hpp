// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_RANGE_HPP
#define MLAYER_RANGE_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace mlayer {

// ONNX Range: start, limit, delta -> 1-D sequence.
template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT>
class Range : public LayerT {
public:
    Range() {
        this->one_blob_only = false;
        this->support_inplace = false;
    }

    std::int32_t load_param(const ParamDictT& /*pd*/) override { return 0; }

    std::int32_t forward(const std::vector<MatT>& bottom_blobs, std::vector<MatT>& top_blobs,
                         const OptionT& opt) const override {
        if (bottom_blobs.size() < 3 || top_blobs.empty()) {
            return -100;
        }
        const float start = bottom_blobs[0][0];
        const float limit = bottom_blobs[1][0];
        const float delta = bottom_blobs[2][0];
        if (delta == 0.0F) {
            return -1;
        }
        const float span = (limit - start) / delta;
        const auto count = std::max(0, static_cast<std::int32_t>(std::ceil(span)));
        MatT& top_blob = top_blobs[0];
        top_blob.create(count, 4u, opt.blob_allocator);
        if (top_blob.empty() && count > 0) {
            return -100;
        }
        for (std::int32_t i = 0; i < count; ++i) {
            top_blob[i] = start + delta * static_cast<float>(i);
        }
        return 0;
    }
};

}  // namespace mlayer

#endif  // MLAYER_RANGE_HPP
