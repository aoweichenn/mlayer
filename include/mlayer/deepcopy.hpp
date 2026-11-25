// Copyright 2020 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_DEEPCOPY_HPP
#define MLAYER_DEEPCOPY_HPP

#include <cstdint>

namespace mlayer {

// Materialise an independent copy of the input blob.
template<typename LayerT, typename MatT, typename OptionT>
class DeepCopy : public LayerT {
public:
    DeepCopy() {
        this->one_blob_only = true;
        this->support_inplace = false;
    }

    std::int32_t forward(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const override {
        top_blob = bottom_blob.clone(opt.blob_allocator);
        return top_blob.empty() ? -100 : 0;
    }
};

}  // namespace mlayer

#endif  // MLAYER_DEEPCOPY_HPP
