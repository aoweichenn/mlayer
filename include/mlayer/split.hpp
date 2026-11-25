// Copyright 2017 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_SPLIT_HPP
#define MLAYER_SPLIT_HPP

#include <cstdint>
#include <vector>

namespace mlayer {

template<typename LayerT, typename MatT, typename OptionT>
class Split : public LayerT {
public:
    Split() {
        this->one_blob_only = false;
        this->support_inplace = false;
    }

    std::int32_t forward(const std::vector<MatT>& bottom_blobs, std::vector<MatT>& top_blobs,
                         const OptionT&) const override {
        const MatT& bottom_blob = bottom_blobs[0];
        for (MatT& top_blob : top_blobs) {
            top_blob = bottom_blob;
        }
        return 0;
    }
};

}  // namespace mlayer

#endif  // MLAYER_SPLIT_HPP
