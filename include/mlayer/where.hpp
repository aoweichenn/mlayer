// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_WHERE_HPP
#define MLAYER_WHERE_HPP

#include <algorithm>
#include <cstdint>
#include <vector>

namespace mlayer {

// ONNX `Where(cond, a, b)`: elementwise select with NumPy broadcasting.
// `cond != 0` picks `a`, otherwise `b`. Output is float, like the rest of the
// runtime's blobs, so masks travel as 0/1 floats.
template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, bool Validate>
class Where : public LayerT {
public:
    Where() {
        this->one_blob_only = false;
        this->support_inplace = false;
    }

    std::int32_t load_param(const ParamDictT&) override {
        return 0;
    }

    std::int32_t forward(const std::vector<MatT>& bottom_blobs, std::vector<MatT>& top_blobs,
                         const OptionT& opt) const override {
        if (bottom_blobs.size() < 3 || top_blobs.empty()) {
            return -100;
        }
        const MatT& cond = bottom_blobs[0];
        const MatT& a = bottom_blobs[1];
        const MatT& b = bottom_blobs[2];
        if (cond.empty() || a.empty() || b.empty()) {
            return -100;
        }
        MatT& top = top_blobs[0];

        const std::int32_t outw = std::max({cond.w, a.w, b.w});
        const std::int32_t outh = std::max({cond.h, a.h, b.h});
        const std::int32_t outc = std::max({cond.c, a.c, b.c});
        const bool three_d = cond.dims >= 3 || a.dims >= 3 || b.dims >= 3;
        if (three_d) {
            top.create(outw, outh, outc, 4u, opt.blob_allocator);
        } else {
            top.create(outw, outh, 4u, opt.blob_allocator);
        }
        if (top.empty()) {
            return -100;
        }

#pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t q = 0; q < outc; ++q) {
            float* outptr = top.channel(q);
            const MatT& cc = cond.channel(std::min(q, cond.c - 1));
            const MatT& ac = a.channel(std::min(q, a.c - 1));
            const MatT& bc = b.channel(std::min(q, b.c - 1));
            for (std::int32_t y = 0; y < outh; ++y) {
                const float* pc = cc.row(std::min(y, cc.h - 1));
                const float* pa = ac.row(std::min(y, ac.h - 1));
                const float* pb = bc.row(std::min(y, bc.h - 1));
                for (std::int32_t x = 0; x < outw; ++x) {
                    const float vc = pc[std::min(x, cc.w - 1)];
                    const float va = pa[std::min(x, ac.w - 1)];
                    const float vb = pb[std::min(x, bc.w - 1)];
                    *outptr++ = vc != 0.0F ? va : vb;
                }
            }
        }
        return 0;
    }
};

}  // namespace mlayer

#endif  // MLAYER_WHERE_HPP
