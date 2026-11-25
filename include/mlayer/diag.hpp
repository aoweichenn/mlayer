// Copyright 2024 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_DIAG_HPP
#define MLAYER_DIAG_HPP

#include <algorithm>
#include <cstdint>

namespace mlayer {

// Vector -> diagonal matrix (dims 1) and matrix -> diagonal (dims 2).
template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT>
class Diag : public LayerT {
public:
    Diag() {
        this->one_blob_only = true;
        this->support_inplace = false;
        this->diagonal = 0;
    }

    std::int32_t load_param(const ParamDictT& pd) override {
        this->diagonal = pd.get(0, 0);
        return 0;
    }

    std::int32_t forward(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const override {
        const std::int32_t dims = bottom_blob.dims;
        const std::size_t elemsize = bottom_blob.elemsize;

        if (dims == 1) {
            const std::int32_t w = bottom_blob.w;
            const std::int32_t top_w = w + (this->diagonal >= 0 ? this->diagonal : -this->diagonal);
            top_blob.create(top_w, top_w, elemsize, opt.blob_allocator);
            if (top_blob.empty()) {
                return -100;
            }
            top_blob.fill(0.0F);
            const std::int32_t bias_r = -std::min(this->diagonal, 0);
            const std::int32_t bias_c = std::max(this->diagonal, 0);
            for (std::int32_t i = 0; i < w; ++i) {
                top_blob.row(i + bias_r)[i + bias_c] = bottom_blob[i];
            }
            return 0;
        }

        if (dims == 2) {
            const std::int32_t w = bottom_blob.w;
            const std::int32_t h = bottom_blob.h;
            const std::int32_t minimum = std::min(w - h, 0);
            const std::int32_t maximum = std::max(w - h, 0);
            std::int32_t len = 0;
            if (this->diagonal <= maximum && this->diagonal >= minimum) {
                len = std::min(w, h);
            } else if (this->diagonal > -h && this->diagonal < minimum) {
                len = this->diagonal + h;
            } else if (this->diagonal > maximum && this->diagonal < w) {
                len = -this->diagonal + w;
            }
            top_blob.create(len, elemsize, opt.blob_allocator);
            if (top_blob.empty()) {
                return len == 0 ? 0 : -100;
            }
            const std::int32_t bias_r = -std::min(this->diagonal, 0);
            const std::int32_t bias_c = std::max(this->diagonal, 0);
            for (std::int32_t i = 0; i < len; ++i) {
                top_blob[i] = bottom_blob.row(i + bias_r)[i + bias_c];
            }
            return 0;
        }

        // dims 3/4 diagonals are not used by any model this runtime loads.
        return -1;
    }

public:
    std::int32_t diagonal;
};

}  // namespace mlayer

#endif  // MLAYER_DIAG_HPP
