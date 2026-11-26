// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_COMPARE_HPP
#define MLAYER_COMPARE_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace mlayer {

// Elementwise comparison with NumPy-style broadcasting, emitting a 0/1 float
// mask (ONNX Equal/Less/Greater/...). Index clamping matches the BinaryOp
// broadcast rules already used by the runtime.
template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, bool Validate>
class Compare : public LayerT {
public:
    enum OpType {
        OpEqual = 0,
        OpNotEqual = 1,
        OpLess = 2,
        OpLessEqual = 3,
        OpGreater = 4,
        OpGreaterEqual = 5,
    };

    Compare() {
        this->one_blob_only = false;
        this->support_inplace = false;
        this->op_type = OpEqual;
    }

    std::int32_t load_param(const ParamDictT& pd) override {
        this->op_type = pd.get(0, OpEqual);
        if constexpr (Validate) {
            if (this->op_type < OpEqual || this->op_type > OpGreaterEqual) {
                return -1;
            }
        }
        return 0;
    }

    std::int32_t forward(const std::vector<MatT>& bottom_blobs, std::vector<MatT>& top_blobs,
                         const OptionT& opt) const override {
        if (bottom_blobs.size() < 2 || top_blobs.empty()) {
            return -100;
        }
        const MatT& a = bottom_blobs[0];
        const MatT& b = bottom_blobs[1];
        MatT& top = top_blobs[0];

        const std::int32_t outw = std::max(a.w, b.w);
        const std::int32_t outh = std::max(a.h, b.h);
        const std::int32_t outd = std::max(a.d, b.d);
        const std::int32_t outc = std::max(a.c, b.c);

        if (a.dims == 1 && b.dims == 1) {
            top.create(outw, 4u, opt.blob_allocator);
        } else if (a.dims <= 2 && b.dims <= 2) {
            top.create(outw, outh, 4u, opt.blob_allocator);
        } else {
            top.create(outw, outh, outc, 4u, opt.blob_allocator);
        }
        if (top.empty()) {
            return -100;
        }

        const std::int32_t size = outw * outh * outd;
#pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t q = 0; q < outc; ++q) {
            float* outptr = top.channel(q);
            const MatT& ac = a.channel(std::min(q, a.c - 1));
            const MatT& bc = b.channel(std::min(q, b.c - 1));
            for (std::int32_t z = 0; z < (top.dims >= 4 ? outd : 1); ++z) {
                for (std::int32_t y = 0; y < outh; ++y) {
                    const float* pa = ac.depth(std::min(z, ac.d - 1)).row(std::min(y, ac.h - 1));
                    const float* pb = bc.depth(std::min(z, bc.d - 1)).row(std::min(y, bc.h - 1));
                    for (std::int32_t x = 0; x < outw; ++x) {
                        const float va = pa[std::min(x, ac.w - 1)];
                        const float vb = pb[std::min(x, bc.w - 1)];
                        const bool r = this->compare(va, vb);
                        *outptr++ = r ? 1.0F : 0.0F;
                    }
                }
            }
        }
        return 0;
    }

    std::int32_t op_type;

private:
    [[nodiscard]] bool compare(float a, float b) const {
        switch (this->op_type) {
            case OpEqual:
                return a == b;
            case OpNotEqual:
                return a != b;
            case OpLess:
                return a < b;
            case OpLessEqual:
                return a <= b;
            case OpGreater:
                return a > b;
            case OpGreaterEqual:
                return a >= b;
            default:
                return false;
        }
    }
};

}  // namespace mlayer

#endif  // MLAYER_COMPARE_HPP
