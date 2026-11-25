// Copyright 2017 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_UNARYOP_HPP
#define MLAYER_UNARYOP_HPP

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace mlayer {

template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, bool Validate = false>
class UnaryOp : public LayerT {
public:
    UnaryOp() {
        this->one_blob_only = true;
        this->support_inplace = true;
    }

    std::int32_t load_param(const ParamDictT& pd) override {
        this->op_type = pd.get(0, 0);
        if constexpr (Validate) {
            if (this->op_type < Operation_ABS || this->op_type > Operation_LOG1P) {
                return -1;
            }
        }
        return 0;
    }

    std::int32_t forward_inplace(MatT& blob, const OptionT& opt) const override {
        float* ptr = blob;
        const std::int32_t size = static_cast<std::int32_t>(blob.total());
        #pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t i = 0; i < size; ++i) {
            ptr[i] = apply(ptr[i]);
        }
        return 0;
    }

public:
    enum OperationType {
        Operation_ABS = 0,
        Operation_NEG = 1,
        Operation_FLOOR = 2,
        Operation_CEIL = 3,
        Operation_SQUARE = 4,
        Operation_SQRT = 5,
        Operation_RSQRT = 6,
        Operation_EXP = 7,
        Operation_LOG = 8,
        Operation_SIN = 9,
        Operation_COS = 10,
        Operation_TAN = 11,
        Operation_ASIN = 12,
        Operation_ACOS = 13,
        Operation_ATAN = 14,
        Operation_RECIPROCAL = 15,
        Operation_TANH = 16,
        Operation_LOG10 = 17,
        Operation_ROUND = 18,
        Operation_TRUNC = 19,
        Operation_SIGN = 20,
        Operation_EXPM1 = 21,
        Operation_SINH = 22,
        Operation_ASINH = 23,
        Operation_COSH = 24,
        Operation_ACOSH = 25,
        Operation_ATANH = 26,
        Operation_LOG1P = 27
    };

    std::int32_t op_type = Operation_ABS;

private:
    float apply(float x) const {
        switch (this->op_type) {
            case Operation_ABS:
                return std::fabs(x);
            case Operation_NEG:
                return -x;
            case Operation_FLOOR:
                return std::floor(x);
            case Operation_CEIL:
                return std::ceil(x);
            case Operation_SQUARE:
                return x * x;
            case Operation_SQRT:
                return std::sqrt(x);
            case Operation_RSQRT:
                return 1.0F / std::sqrt(x);
            case Operation_EXP:
                return std::exp(x);
            case Operation_LOG:
                return std::log(x);
            case Operation_SIN:
                return std::sin(x);
            case Operation_COS:
                return std::cos(x);
            case Operation_TAN:
                return std::tan(x);
            case Operation_ASIN:
                return std::asin(x);
            case Operation_ACOS:
                return std::acos(x);
            case Operation_ATAN:
                return std::atan(x);
            case Operation_RECIPROCAL:
                return 1.0F / x;
            case Operation_TANH:
                return std::tanh(x);
            case Operation_LOG10:
                return std::log10(x);
            case Operation_ROUND:
                return std::round(x);
            case Operation_TRUNC:
                return std::trunc(x);
            case Operation_SIGN:
                return x > 0.0F ? 1.0F : (x < 0.0F ? -1.0F : 0.0F);
            case Operation_EXPM1:
                return std::expm1(x);
            case Operation_SINH:
                return std::sinh(x);
            case Operation_ASINH:
                return std::asinh(x);
            case Operation_COSH:
                return std::cosh(x);
            case Operation_ACOSH:
                return std::acosh(x);
            case Operation_ATANH:
                return std::atanh(x);
            case Operation_LOG1P:
                return std::log1p(x);
            default:
                return x;
        }
    }
};

}  // namespace mlayer

#endif  // MLAYER_UNARYOP_HPP
