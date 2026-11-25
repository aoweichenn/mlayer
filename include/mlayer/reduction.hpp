// Copyright 2017 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_REDUCTION_HPP
#define MLAYER_REDUCTION_HPP

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace mlayer {

template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, bool Validate = false>
class Reduction : public LayerT {
public:
    enum ReductionOp {
        ReductionOp_SUM = 0,
        ReductionOp_ASUM = 1,
        ReductionOp_SUMSQ = 2,
        ReductionOp_MEAN = 3,
        ReductionOp_MAX = 4,
        ReductionOp_MIN = 5,
        ReductionOp_PROD = 6,
        ReductionOp_L1 = 7,
        ReductionOp_L2 = 8,
        ReductionOp_LogSum = 9,
        ReductionOp_LogSumExp = 10
    };

    Reduction() {
        this->one_blob_only = true;
        this->support_inplace = false;
    }

    std::int32_t load_param(const ParamDictT& pd) override {
        this->operation = pd.get(0, 0);
        this->reduce_all = pd.get(1, 1);
        this->coeff = pd.get(2, 1.0F);
        this->axes = pd.get(3, MatT());
        this->keepdims = pd.get(4, 0);
        const std::int32_t fixbug0 = pd.get(5, 0);

        if constexpr (Validate) {
            if (this->operation < ReductionOp_SUM || this->operation > ReductionOp_LogSumExp) {
                return -1;
            }
            if (this->axes.w > 4) {
                return -1;
            }
        }
        // ncnn rejects params generated before the axes/keepdims fix.
        if (fixbug0 == 0 && !this->axes.empty()) {
            return -1;
        }
        return 0;
    }

    std::int32_t forward(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const override {
        std::int32_t outdims = 0;
        std::int32_t outw = 1;
        std::int32_t outh = 1;
        std::int32_t outd = 1;
        std::int32_t outc = 1;
        bool reduce[4] = {false, false, false, false};
        this->resolve_shape(bottom_blob, reduce, outdims, outw, outh, outd, outc);

        if (outdims == 0) {
            top_blob.create(1, bottom_blob.elemsize, opt.blob_allocator);
        } else if (outdims == 1) {
            top_blob.create(outw, bottom_blob.elemsize, opt.blob_allocator);
        } else if (outdims == 2) {
            top_blob.create(outw, outh, bottom_blob.elemsize, opt.blob_allocator);
        } else if (outdims == 3) {
            top_blob.create(outw, outh, outc, bottom_blob.elemsize, opt.blob_allocator);
        } else {
            top_blob.create(outw, outh, outd, outc, bottom_blob.elemsize, opt.blob_allocator);
        }
        if (top_blob.empty()) {
            return -100;
        }

        const std::int32_t dims = bottom_blob.dims;
        // Compact axis list in fastest-first order. For 3D blobs the third
        // axis is c (there is no d), matching ncnn's reduce flags.
        const std::int32_t axis2_size = dims == 4 ? bottom_blob.d : (dims == 3 ? bottom_blob.c : 1);
        const bool axis2_reduce = dims == 4 ? reduce[2] : (dims == 3 ? reduce[3] : false);
        const std::int32_t sz[4] = {bottom_blob.w, dims >= 2 ? bottom_blob.h : 1, axis2_size,
                                    dims == 4 ? bottom_blob.c : 1};
        const bool red[4] = {reduce[0], reduce[1], axis2_reduce, dims == 4 ? reduce[3] : false};

        std::int32_t out_stride[4] = {0, 0, 0, 0};
        {
            std::int32_t stride = 1;
            for (std::int32_t axis = 0; axis < 4; ++axis) {
                if (this->keepdims) {
                    out_stride[axis] = stride;
                    stride *= red[axis] ? 1 : sz[axis];
                } else if (!red[axis]) {
                    out_stride[axis] = stride;
                    stride *= sz[axis];
                }
            }
        }

        std::int32_t reduce_count = 1;
        for (std::int32_t axis = 0; axis < 4; ++axis) {
            if (red[axis]) {
                reduce_count *= sz[axis];
            }
        }

        const auto* in_ptr = static_cast<const float*>(bottom_blob);
        float* out_ptr = top_blob;
        const std::int32_t out_total = static_cast<std::int32_t>(top_blob.total());
        const std::size_t cstep = bottom_blob.cstep;
        const std::int32_t plane = bottom_blob.w * bottom_blob.h;

        const std::int32_t n3 = dims == 4 ? bottom_blob.c : 1;
        const std::int32_t n2 = axis2_size;
        const std::int32_t n1 = sz[1];
        const std::int32_t n0 = sz[0];

        auto out_index = [&](std::int32_t a3, std::int32_t a2, std::int32_t a1, std::int32_t a0) {
            return (red[3] ? 0 : a3) * out_stride[3] + (red[2] ? 0 : a2) * out_stride[2] +
                   (red[1] ? 0 : a1) * out_stride[1] + (red[0] ? 0 : a0) * out_stride[0];
        };

        if (this->operation == ReductionOp_LogSumExp) {
            std::vector<float> max_value(static_cast<std::size_t>(out_total), -FLT_MAX);
            for (std::int32_t a3 = 0; a3 < n3; ++a3) {
                for (std::int32_t a2 = 0; a2 < n2; ++a2) {
                    for (std::int32_t a1 = 0; a1 < n1; ++a1) {
                        for (std::int32_t a0 = 0; a0 < n0; ++a0) {
                            const float* cbase = in_ptr;
                            if (dims == 4) {
                                cbase += static_cast<std::size_t>(a3) * cstep + static_cast<std::size_t>(a2) * plane;
                            } else if (dims == 3) {
                                cbase += static_cast<std::size_t>(a2) * cstep;
                            }
                            const float value = cbase[a1 * n0 + a0];
                            const std::int32_t oi = out_index(a3, a2, a1, a0);
                            max_value[static_cast<std::size_t>(oi)] =
                                std::max(max_value[static_cast<std::size_t>(oi)], value);
                        }
                    }
                }
            }
            for (std::int32_t i = 0; i < out_total; ++i) {
                out_ptr[i] = 0.0F;
            }
            for (std::int32_t a3 = 0; a3 < n3; ++a3) {
                for (std::int32_t a2 = 0; a2 < n2; ++a2) {
                    for (std::int32_t a1 = 0; a1 < n1; ++a1) {
                        for (std::int32_t a0 = 0; a0 < n0; ++a0) {
                            const float* cbase = in_ptr;
                            if (dims == 4) {
                                cbase += static_cast<std::size_t>(a3) * cstep + static_cast<std::size_t>(a2) * plane;
                            } else if (dims == 3) {
                                cbase += static_cast<std::size_t>(a2) * cstep;
                            }
                            const float value = cbase[a1 * n0 + a0];
                            const std::int32_t oi = out_index(a3, a2, a1, a0);
                            out_ptr[oi] += std::exp(value - max_value[static_cast<std::size_t>(oi)]);
                        }
                    }
                }
            }
            for (std::int32_t i = 0; i < out_total; ++i) {
                out_ptr[i] = std::log(out_ptr[i]) + max_value[static_cast<std::size_t>(i)];
            }
        } else {
            if (this->operation == ReductionOp_PROD) {
                for (std::int32_t i = 0; i < out_total; ++i) {
                    out_ptr[i] = 1.0F;
                }
            } else if (this->operation == ReductionOp_MAX) {
                for (std::int32_t i = 0; i < out_total; ++i) {
                    out_ptr[i] = -FLT_MAX;
                }
            } else if (this->operation == ReductionOp_MIN) {
                for (std::int32_t i = 0; i < out_total; ++i) {
                    out_ptr[i] = FLT_MAX;
                }
            } else {
                for (std::int32_t i = 0; i < out_total; ++i) {
                    out_ptr[i] = 0.0F;
                }
            }
            for (std::int32_t a3 = 0; a3 < n3; ++a3) {
                for (std::int32_t a2 = 0; a2 < n2; ++a2) {
                    for (std::int32_t a1 = 0; a1 < n1; ++a1) {
                        for (std::int32_t a0 = 0; a0 < n0; ++a0) {
                            const float* cbase = in_ptr;
                            if (dims == 4) {
                                cbase += static_cast<std::size_t>(a3) * cstep + static_cast<std::size_t>(a2) * plane;
                            } else if (dims == 3) {
                                cbase += static_cast<std::size_t>(a2) * cstep;
                            }
                            const std::int32_t oi = out_index(a3, a2, a1, a0);
                            const float v = cbase[a1 * n0 + a0];
                            float& acc = out_ptr[oi];
                            switch (this->operation) {
                                case ReductionOp_SUM:
                                case ReductionOp_MEAN:
                                    acc += v;
                                    break;
                                case ReductionOp_ASUM:
                                case ReductionOp_L1:
                                    acc += std::fabs(v);
                                    break;
                                case ReductionOp_SUMSQ:
                                case ReductionOp_L2:
                                    acc += v * v;
                                    break;
                                case ReductionOp_MAX:
                                    acc = std::max(acc, v);
                                    break;
                                case ReductionOp_MIN:
                                    acc = std::min(acc, v);
                                    break;
                                case ReductionOp_PROD:
                                    acc *= v;
                                    break;
                                case ReductionOp_LogSum:
                                    acc += v;
                                    break;
                                default:
                                    break;
                            }
                        }
                    }
                }
            }
        }

        if (this->operation == ReductionOp_LogSum) {
            for (std::int32_t i = 0; i < out_total; ++i) {
                out_ptr[i] = std::log(out_ptr[i]);
            }
        }
        if (this->operation == ReductionOp_L2) {
            for (std::int32_t i = 0; i < out_total; ++i) {
                out_ptr[i] = std::sqrt(out_ptr[i] < FLT_MIN ? 0.0F : out_ptr[i]);
            }
        }

        float scale = this->coeff;
        if (this->operation == ReductionOp_MEAN) {
            scale /= static_cast<float>(reduce_count);
        }
        if (scale != 1.0F) {
            for (std::int32_t i = 0; i < out_total; ++i) {
                out_ptr[i] *= scale;
            }
        }
        return 0;
    }

public:
    std::int32_t operation = ReductionOp_SUM;
    std::int32_t reduce_all = 1;
    float coeff = 1.0F;
    MatT axes;
    std::int32_t keepdims = 0;

private:
    void resolve_shape(const MatT& blob, bool reduce[4], std::int32_t& outdims, std::int32_t& outw,
                       std::int32_t& outh, std::int32_t& outd, std::int32_t& outc) const {
        const std::int32_t dims = blob.dims;

        if (this->reduce_all) {
            for (std::int32_t i = 0; i < 4; ++i) {
                reduce[i] = true;
            }
        } else {
            std::int32_t axes_flag[4] = {0, 0, 0, 0};
            const auto* axes_ptr = static_cast<const std::int32_t*>(this->axes);
            for (std::int32_t i = 0; i < this->axes.w; ++i) {
                std::int32_t axis = axes_ptr[i];
                if (axis < 0) {
                    axis += dims;
                }
                axes_flag[axis] = 1;
            }
            if (dims == 1) {
                reduce[0] = true;
            }
            if (dims == 2) {
                reduce[1] = axes_flag[0] == 1;
                reduce[0] = axes_flag[1] == 1;
            }
            if (dims == 3) {
                reduce[3] = axes_flag[0] == 1;
                reduce[1] = axes_flag[1] == 1;
                reduce[0] = axes_flag[2] == 1;
            }
            if (dims == 4) {
                reduce[3] = axes_flag[0] == 1;
                reduce[2] = axes_flag[1] == 1;
                reduce[1] = axes_flag[2] == 1;
                reduce[0] = axes_flag[3] == 1;
            }
        }

        if (this->keepdims) {
            outdims = dims;
            outw = reduce[0] ? 1 : blob.w;
            outh = reduce[1] ? 1 : blob.h;
            outd = reduce[2] ? 1 : blob.d;
            outc = reduce[3] ? 1 : blob.c;
            return;
        }

        std::int32_t out_shape[4] = {0, 0, 0, 0};
        std::int32_t count = 0;
        if (!reduce[0]) {
            out_shape[count++] = blob.w;
        }
        if (dims >= 2 && !reduce[1]) {
            out_shape[count++] = blob.h;
        }
        if (dims == 4 && !reduce[2]) {
            out_shape[count++] = blob.d;
        }
        if (dims >= 3 && !reduce[3]) {
            out_shape[count++] = blob.c;
        }

        outdims = count;
        outw = 1;
        outh = 1;
        outd = 1;
        outc = 1;
        if (count == 1) {
            outw = out_shape[0];
        }
        if (count == 2) {
            outw = out_shape[0];
            outh = out_shape[1];
        }
        if (count == 3) {
            outw = out_shape[0];
            outh = out_shape[1];
            outc = out_shape[2];
        }
        if (count == 4) {
            outw = out_shape[0];
            outh = out_shape[1];
            outd = out_shape[2];
            outc = out_shape[3];
        }
    }
};

}  // namespace mlayer

#endif  // MLAYER_REDUCTION_HPP
