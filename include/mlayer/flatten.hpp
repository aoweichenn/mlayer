// Copyright 2017 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_FLATTEN_HPP
#define MLAYER_FLATTEN_HPP

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace mlayer {

template<typename LayerT, typename MatT, typename OptionT>
class Flatten : public LayerT {
public:
    Flatten() {
        this->one_blob_only = true;
        this->support_inplace = false;
    }

    std::int32_t forward(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const override {
        const std::int32_t w = bottom_blob.w;
        const std::int32_t h = bottom_blob.h;
        const std::int32_t d = bottom_blob.d;
        const std::int32_t channels = bottom_blob.c;
        const std::size_t elemsize = bottom_blob.elemsize;
        const std::int32_t size = w * h * d;

        top_blob.create(size * channels, elemsize, opt.blob_allocator);
        if (top_blob.empty()) {
            return -100;
        }

#pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t q = 0; q < channels; ++q) {
            const auto* ptr = static_cast<const std::uint8_t*>(bottom_blob.channel(q));
            auto* outptr = static_cast<std::uint8_t*>(top_blob) +
                           static_cast<std::size_t>(size) * elemsize * static_cast<std::size_t>(q);
            std::memcpy(outptr, ptr, static_cast<std::size_t>(size) * elemsize);
        }

        return 0;
    }
};

}  // namespace mlayer

#endif  // MLAYER_FLATTEN_HPP
