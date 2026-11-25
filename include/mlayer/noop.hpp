// Copyright 2019 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_NOOP_HPP
#define MLAYER_NOOP_HPP

#include <cstdint>
#include <vector>

namespace mlayer {

template<typename LayerT, typename MatT, typename OptionT>
class Noop : public LayerT {
public:
    Noop() {
        this->one_blob_only = false;
        this->support_inplace = false;
    }

    std::int32_t forward(const std::vector<MatT>& bottom_blobs, std::vector<MatT>& top_blobs,
                         const OptionT&) const override {
        top_blobs = bottom_blobs;
        return 0;
    }
};

}  // namespace mlayer

#endif  // MLAYER_NOOP_HPP
