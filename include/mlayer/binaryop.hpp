// Copyright 2017 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_BINARYOP_HPP
#define MLAYER_BINARYOP_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mlayer {

template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, bool Validate>
class BinaryOp : public LayerT {
public:
    BinaryOp();

    std::int32_t load_param(const ParamDictT& pd) override;
    std::int32_t forward(const std::vector<MatT>& bottom_blobs, std::vector<MatT>& top_blobs, const OptionT& opt) const override;
    std::int32_t forward_inplace(MatT& bottom_top_blob, const OptionT& opt) const override;

    enum OperationType {
        Operation_ADD = 0,
        Operation_SUB = 1,
        Operation_MUL = 2,
        Operation_DIV = 3,
        Operation_MAX = 4,
        Operation_MIN = 5,
        Operation_POW = 6,
        Operation_RSUB = 7,
        Operation_RDIV = 8,
        Operation_RPOW = 9,
        Operation_ATAN2 = 10,
        Operation_RATAN2 = 11,
        Operation_FMOD = 12,
        Operation_RFMOD = 13,
        Operation_LOGADDEXP = 14,
        Operation_FLOOR_DIVIDE = 15,
        Operation_RFLOOR_DIVIDE = 16,
        Operation_REMAINDER = 17,
        Operation_RREMAINDER = 18
    };

public:
    std::int32_t op_type;
    float with_scalar;
    float b;
};

#define MLAYER_BINARYOP_TEMPLATE \
    template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, bool Validate>

MLAYER_BINARYOP_TEMPLATE
BinaryOp<LayerT, MatT, OptionT, ParamDictT, Validate>::BinaryOp()
{
    this->one_blob_only = false;
    this->support_inplace = false;
}

MLAYER_BINARYOP_TEMPLATE
std::int32_t BinaryOp<LayerT, MatT, OptionT, ParamDictT, Validate>::load_param(const ParamDictT& pd)
{
    op_type = pd.get(0, 0);
    with_scalar = pd.get(1, 0);
    b = pd.get(2, 0.f);

if constexpr (Validate) {
    if (op_type < Operation_ADD || op_type > Operation_RREMAINDER)
        return -1;
}

    if (with_scalar != 0)
    {
        this->one_blob_only = true;
        this->support_inplace = true;
    }

    return 0;
}

// broadcasting rule
// https://github.com/Tencent/ncnn/wiki/binaryop-broadcasting

template<typename Op, typename MatT, typename OptionT>
static void binary_op_broadcast(const MatT& a, const MatT& b, MatT& c, const OptionT& opt)
{
    // general broadcast
    const Op op;

    const std::int32_t dims = c.dims;
    const std::int32_t w = c.w;
    const std::int32_t h = c.h;
    const std::int32_t d = c.d;
    const std::int32_t channels = c.c;

    if (dims == 1)
    {
        const float* ptr = a;
        const float* ptr1 = b;
        float* outptr = c;

        const std::int32_t ainc = a.w > 1 ? 1 : 0;
        const std::int32_t binc = b.w > 1 ? 1 : 0;

        for (std::int32_t x = 0; x < w; x++)
        {
            outptr[x] = op(*ptr, *ptr1);
            ptr += ainc;
            ptr1 += binc;
        }
    }

    if (dims == 2)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t y = 0; y < h; y++)
        {
            const float* ptr = a.row(std::min(y, a.h - 1));
            const float* ptr1 = b.row(std::min(y, b.h - 1));
            float* outptr = c.row(y);

            const std::int32_t ainc = a.w > 1 ? 1 : 0;
            const std::int32_t binc = b.w > 1 ? 1 : 0;

            for (std::int32_t x = 0; x < w; x++)
            {
                outptr[x] = op(*ptr, *ptr1);
                ptr += ainc;
                ptr1 += binc;
            }
        }
    }

    if (dims == 3 || dims == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t q = 0; q < channels; q++)
        {
            float* outptr = c.channel(q);

            const std::int32_t ainc = a.w > 1 ? 1 : 0;
            const std::int32_t binc = b.w > 1 ? 1 : 0;

            for (std::int32_t z = 0; z < d; z++)
            {
                for (std::int32_t y = 0; y < h; y++)
                {
                    const float* ptr = a.channel(std::min(q, a.c - 1)).depth(std::min(z, a.d - 1)).row(std::min(y, a.h - 1));
                    const float* ptr1 = b.channel(std::min(q, b.c - 1)).depth(std::min(z, b.d - 1)).row(std::min(y, b.h - 1));

                    for (std::int32_t x = 0; x < w; x++)
                    {
                        outptr[x] = op(*ptr, *ptr1);
                        ptr += ainc;
                        ptr1 += binc;
                    }

                    outptr += w;
                }
            }
        }
    }
}

template<typename Op, typename MatT, typename OptionT>
static void binary_op_scalar_inplace(MatT& a, float b, const OptionT& opt)
{
    const Op op;

    const std::int32_t channels = a.c;
    const std::int32_t size = a.w * a.h * a.d;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (std::int32_t q = 0; q < channels; q++)
    {
        float* ptr = a.channel(q);

        for (std::int32_t i = 0; i < size; i++)
        {
            ptr[i] = op(ptr[i], b);
        }
    }
}

struct binary_op_add
{
    float operator()(const float& x, const float& y) const
    {
        return x + y;
    }
};

struct binary_op_sub
{
    float operator()(const float& x, const float& y) const
    {
        return x - y;
    }
};

struct binary_op_mul
{
    float operator()(const float& x, const float& y) const
    {
        return x * y;
    }
};

struct binary_op_div
{
    float operator()(const float& x, const float& y) const
    {
        return x / y;
    }
};

struct binary_op_max
{
    float operator()(const float& x, const float& y) const
    {
        return std::max(x, y);
    }
};

struct binary_op_min
{
    float operator()(const float& x, const float& y) const
    {
        return std::min(x, y);
    }
};

struct binary_op_pow
{
    float operator()(const float& x, const float& y) const
    {
        return (float)powf(x, y);
    }
};

struct binary_op_rsub
{
    float operator()(const float& x, const float& y) const
    {
        return y - x;
    }
};

struct binary_op_rdiv
{
    float operator()(const float& x, const float& y) const
    {
        return y / x;
    }
};

struct binary_op_rpow
{
    float operator()(const float& x, const float& y) const
    {
        return (float)powf(y, x);
    }
};

struct binary_op_atan2
{
    float operator()(const float& x, const float& y) const
    {
        return (float)atan2f(x, y);
    }
};

struct binary_op_ratan2
{
    float operator()(const float& x, const float& y) const
    {
        return (float)atan2f(y, x);
    }
};

struct binary_op_fmod
{
    float operator()(const float& x, const float& y) const
    {
        return (float)fmodf(x, y);
    }
};

struct binary_op_logaddexp
{
    float operator()(const float& x, const float& y) const
    {
        float max_xy = std::max(x, y);
        float min_xy = std::min(x, y);
        return (float)(max_xy + log1pf(expf(min_xy - max_xy)));
    }
};

struct binary_op_floor_divide
{
    float operator()(const float& x, const float& y) const
    {
        return (float)floorf(x / y);
    }
};

struct binary_op_remainder
{
    float operator()(const float& x, const float& y) const
    {
        return (float)remainderf(x, y);
    }
};

struct binary_op_rfmod
{
    float operator()(const float& x, const float& y) const
    {
        return (float)fmodf(y, x);
    }
};

struct binary_op_rfloor_divide
{
    float operator()(const float& x, const float& y) const
    {
        return (float)floorf(y / x);
    }
};

struct binary_op_rremainder
{
    float operator()(const float& x, const float& y) const
    {
        return (float)remainderf(y, x);
    }
};

template<typename MatT, typename OptionT>
static void binary_op_broadcast(const MatT& a, const MatT& b, MatT& c, std::int32_t op_type, const OptionT& opt)
{
    if (op_type == 0) return binary_op_broadcast<binary_op_add>(a, b, c, opt);
    if (op_type == 1) return binary_op_broadcast<binary_op_sub>(a, b, c, opt);
    if (op_type == 2) return binary_op_broadcast<binary_op_mul>(a, b, c, opt);
    if (op_type == 3) return binary_op_broadcast<binary_op_div>(a, b, c, opt);
    if (op_type == 4) return binary_op_broadcast<binary_op_max>(a, b, c, opt);
    if (op_type == 5) return binary_op_broadcast<binary_op_min>(a, b, c, opt);
    if (op_type == 6) return binary_op_broadcast<binary_op_pow>(a, b, c, opt);
    if (op_type == 7) return binary_op_broadcast<binary_op_sub>(b, a, c, opt);
    if (op_type == 8) return binary_op_broadcast<binary_op_div>(b, a, c, opt);
    if (op_type == 9) return binary_op_broadcast<binary_op_pow>(b, a, c, opt);
    if (op_type == 10) return binary_op_broadcast<binary_op_atan2>(a, b, c, opt);
    if (op_type == 11) return binary_op_broadcast<binary_op_atan2>(b, a, c, opt);
    if (op_type == 12) return binary_op_broadcast<binary_op_fmod>(a, b, c, opt);
    if (op_type == 13) return binary_op_broadcast<binary_op_fmod>(b, a, c, opt);
    if (op_type == 14) return binary_op_broadcast<binary_op_logaddexp>(a, b, c, opt);
    if (op_type == 15) return binary_op_broadcast<binary_op_floor_divide>(a, b, c, opt);
    if (op_type == 16) return binary_op_broadcast<binary_op_floor_divide>(b, a, c, opt);
    if (op_type == 17) return binary_op_broadcast<binary_op_remainder>(a, b, c, opt);
    if (op_type == 18) return binary_op_broadcast<binary_op_remainder>(b, a, c, opt);

    // should never reach here
}

template<typename MatT, typename OptionT>
static void binary_op_scalar_inplace(MatT& bottom_top_blob, float b, std::int32_t op_type, const OptionT& opt)
{
    if (op_type == 0) return binary_op_scalar_inplace<binary_op_add>(bottom_top_blob, b, opt);
    if (op_type == 1) return binary_op_scalar_inplace<binary_op_sub>(bottom_top_blob, b, opt);
    if (op_type == 2) return binary_op_scalar_inplace<binary_op_mul>(bottom_top_blob, b, opt);
    if (op_type == 3) return binary_op_scalar_inplace<binary_op_div>(bottom_top_blob, b, opt);
    if (op_type == 4) return binary_op_scalar_inplace<binary_op_max>(bottom_top_blob, b, opt);
    if (op_type == 5) return binary_op_scalar_inplace<binary_op_min>(bottom_top_blob, b, opt);
    if (op_type == 6) return binary_op_scalar_inplace<binary_op_pow>(bottom_top_blob, b, opt);
    if (op_type == 7) return binary_op_scalar_inplace<binary_op_rsub>(bottom_top_blob, b, opt);
    if (op_type == 8) return binary_op_scalar_inplace<binary_op_rdiv>(bottom_top_blob, b, opt);
    if (op_type == 9) return binary_op_scalar_inplace<binary_op_rpow>(bottom_top_blob, b, opt);
    if (op_type == 10) return binary_op_scalar_inplace<binary_op_atan2>(bottom_top_blob, b, opt);
    if (op_type == 11) return binary_op_scalar_inplace<binary_op_ratan2>(bottom_top_blob, b, opt);
    if (op_type == 12) return binary_op_scalar_inplace<binary_op_fmod>(bottom_top_blob, b, opt);
    if (op_type == 13) return binary_op_scalar_inplace<binary_op_rfmod>(bottom_top_blob, b, opt);
    if (op_type == 14) return binary_op_scalar_inplace<binary_op_logaddexp>(bottom_top_blob, b, opt);
    if (op_type == 15) return binary_op_scalar_inplace<binary_op_floor_divide>(bottom_top_blob, b, opt);
    if (op_type == 16) return binary_op_scalar_inplace<binary_op_rfloor_divide>(bottom_top_blob, b, opt);
    if (op_type == 17) return binary_op_scalar_inplace<binary_op_remainder>(bottom_top_blob, b, opt);
    if (op_type == 18) return binary_op_scalar_inplace<binary_op_rremainder>(bottom_top_blob, b, opt);

    // should never reach here
}

MLAYER_BINARYOP_TEMPLATE
std::int32_t BinaryOp<LayerT, MatT, OptionT, ParamDictT, Validate>::forward(const std::vector<MatT>& bottom_blobs, std::vector<MatT>& top_blobs, const OptionT& opt) const
{
    const MatT& A = bottom_blobs[0];
    const MatT& B = bottom_blobs[1];
        if (A.empty() || B.empty()) {
            return -100;
        }
    const std::int32_t outdims = std::max(A.dims, B.dims);

    MatT A2 = A;
    MatT B2 = B;
    if (A.dims < outdims)
    {
        // expand inner axes
        if (outdims == 2)
        {
            if (A.w == B.h)
                A2 = A.reshape(1, A.w);
            else // if (A.w == B.w)
                A2 = A.reshape(A.w, 1);
        }
        if (outdims == 3 && A.dims == 1)
        {
            if (A.w == B.c)
                A2 = A.reshape(1, 1, A.w);
            else if (A.w == B.h)
                A2 = A.reshape(1, A.w, 1);   // per-channel vector: ONNX [1, C, 1]
            else // if (A.w == B.w)
                A2 = A.reshape(A.w, 1, 1);
        }
        if (outdims == 3 && A.dims == 2)
            // A 2-D (w,h) tensor is (w,h,1) when the other side has one
            // channel; expanding to (1,w,h) instead would square the width.
            A2 = (B.c == 1) ? A.reshape(A.w, A.h, 1) : A.reshape(1, A.w, A.h);
        if (outdims == 4 && A.dims == 1)
        {
            if (A.w == B.c)
                A2 = A.reshape(1, 1, 1, A.w);
            else // if (A.w == B.w)
                A2 = A.reshape(A.w, 1, 1, 1);
        }
        if (outdims == 4 && A.dims == 2)
            A2 = A.reshape(1, 1, A.w, A.h);
        if (outdims == 4 && A.dims == 3)
            A2 = A.reshape(1, A.w, A.h, A.c);
    }
    if (B.dims < outdims)
    {
        // expand inner axes
        if (outdims == 2)
        {
            if (B.w == A.h)
                B2 = B.reshape(1, B.w);
            else // if (B.w == A.w)
                B2 = B.reshape(B.w, 1);
        }
        if (outdims == 3 && B.dims == 1)
        {
            if (B.w == A.c)
                B2 = B.reshape(1, 1, B.w);
            else if (B.w == A.h)
                B2 = B.reshape(1, B.w, 1);   // per-channel vector: ONNX [1, C, 1]
            else // if (B.w == A.w)
                B2 = B.reshape(B.w, 1, 1);
        }
        if (outdims == 3 && B.dims == 2)
            B2 = (A.c == 1) ? B.reshape(B.w, B.h, 1) : B.reshape(1, B.w, B.h);
        if (outdims == 4 && B.dims == 1)
        {
            if (B.w == A.c)
                B2 = B.reshape(1, 1, 1, B.w);
            else // if (B.w == A.w)
                B2 = B.reshape(B.w, 1, 1, 1);
        }
        if (outdims == 4 && B.dims == 2)
            B2 = B.reshape(1, 1, B.w, B.h);
        if (outdims == 4 && B.dims == 3)
            B2 = B.reshape(1, B.w, B.h, B.c);
    }

    // Validate only after the 1-D/2-D expansion below, otherwise a valid
    // per-channel vector would be rejected.
    if ((A2.w > 1 && B2.w > 1 && A2.w != B2.w) || (A2.h > 1 && B2.h > 1 && A2.h != B2.h) ||
        (A2.c > 1 && B2.c > 1 && A2.c != B2.c)) {
        return -1;
    }

    const std::int32_t outw = std::max(A2.w, B2.w);
    const std::int32_t outh = std::max(A2.h, B2.h);
    const std::int32_t outd = std::max(A2.d, B2.d);
    const std::int32_t outc = std::max(A2.c, B2.c);

    MatT& top_blob = top_blobs[0];
    if (outdims == 1)
    {
        top_blob.create(outw, 4u, opt.blob_allocator);
    }
    if (outdims == 2)
    {
        top_blob.create(outw, outh, 4u, opt.blob_allocator);
    }
    if (outdims == 3)
    {
        top_blob.create(outw, outh, outc, 4u, opt.blob_allocator);
    }
    if (outdims == 4)
    {
        top_blob.create(outw, outh, outd, outc, 4u, opt.blob_allocator);
    }
    if (top_blob.empty())
        return -100;

    binary_op_broadcast(A2, B2, top_blob, op_type, opt);

    return 0;
}

MLAYER_BINARYOP_TEMPLATE
std::int32_t BinaryOp<LayerT, MatT, OptionT, ParamDictT, Validate>::forward_inplace(MatT& bottom_top_blob, const OptionT& opt) const
{
    binary_op_scalar_inplace(bottom_top_blob, b, op_type, opt);

    return 0;
}


#undef MLAYER_BINARYOP_TEMPLATE

}  // namespace mlayer

#endif  // MLAYER_BINARYOP_HPP
