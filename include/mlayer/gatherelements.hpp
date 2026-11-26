// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_GATHERELEMENTS_HPP
#define MLAYER_GATHERELEMENTS_HPP

#include <cmath>
#include <cstdint>
#include <vector>

namespace mlayer {

// ONNX GatherElements on the last axis for a 2-D (w,h) blob. VITS uses this to
// select one element from each row with an (n,1) index tensor.
template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT>
class GatherElements : public LayerT {
public:
    GatherElements() {
        this->one_blob_only = false;
        this->support_inplace = false;
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
        if (this->axis != 0) {
            return -1;
        }
        MatT& top_blob = top_blobs[0];
        top_blob.create(indices.w, indices.h, 4u, opt.blob_allocator);
        if (top_blob.empty()) {
            return -100;
        }
        for (std::int32_t row = 0; row < indices.h; ++row) {
            const float* index_row = indices.row(row);
            float* output_row = top_blob.row(row);
            for (std::int32_t column = 0; column < indices.w; ++column) {
                const auto index = static_cast<std::int32_t>(std::lround(index_row[column]));
                if (index < 0 || index >= data.w) {
                    return -100;
                }
                output_row[column] = data.row(row)[index];
            }
        }
        return 0;
    }

public:
    // 0 = last (w) axis in mnet's (w,h) layout.
    std::int32_t axis;
};

}  // namespace mlayer

#endif  // MLAYER_GATHERELEMENTS_HPP
