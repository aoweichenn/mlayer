// Copyright 2022 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_MATMUL_HPP
#define MLAYER_MATMUL_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace mlayer {

template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT>
class MatMul : public LayerT {
public:
    MatMul();

    std::int32_t load_param(const ParamDictT& pd) override;
    std::int32_t forward(const std::vector<MatT>& bottom_blobs, std::vector<MatT>& top_blobs, const OptionT& opt) const override;

public:
    std::int32_t transB;
};

#define MLAYER_MATMUL_TEMPLATE \
    template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT>

MLAYER_MATMUL_TEMPLATE
MatMul<LayerT, MatT, OptionT, ParamDictT>::MatMul()
{
    this->one_blob_only = false;
    this->support_inplace = false;
}

MLAYER_MATMUL_TEMPLATE
std::int32_t MatMul<LayerT, MatT, OptionT, ParamDictT>::load_param(const ParamDictT& pd)
{
    transB = pd.get(0, 0);

    return 0;
}

template<typename MatT, typename OptionT>
static void transpose(const MatT& X, MatT& XT, const OptionT& opt)
{
    const std::int32_t w = X.w;
    const std::int32_t h = X.h;

    const float* pX = X;
    float* pXT = XT;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (std::int32_t i = 0; i < w; i++)
    {
        float* ptr = pXT + i * h;
        for (std::int32_t j = 0; j < h; j++)
        {
            ptr[j] = pX[j * w + i];
        }
    }
}

template<typename MatT, typename OptionT>
static void matmul_transb(const MatT& A, const MatT& B, MatT& top_blob, const OptionT& opt)
{
    const std::int32_t M = A.h;
    const std::int32_t K = A.w; // assert A.w == B.w
    const std::int32_t N = B.h;

    const float* pA = A;
    const float* pB = B;
    float* pOut = top_blob;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (std::int32_t i = 0; i < M; i++)
    {
        const float* ptrA = pA + i * K;
        float* outptr = pOut + i * N;

        for (std::int32_t j = 0; j < N; j++)
        {
            const float* ptrB = pB + j * K;

            float sum = 0.f;
            for (std::int32_t k = 0; k < K; k++)
            {
                sum += ptrA[k] * ptrB[k];
            }

            *outptr++ = sum;
        }
    }
}

MLAYER_MATMUL_TEMPLATE
std::int32_t MatMul<LayerT, MatT, OptionT, ParamDictT>::forward(const std::vector<MatT>& bottom_blobs, std::vector<MatT>& top_blobs, const OptionT& opt) const
{
    const MatT& A = bottom_blobs[0];
    const MatT& B = bottom_blobs[1];
    MatT& top_blob = top_blobs[0];

    const std::int32_t Adims = A.dims;
    const std::int32_t Bdims = B.dims;
    const std::int32_t max_ABdims = std::max(Adims, Bdims);
    const std::size_t elemsize = A.elemsize;

    if (Adims == 1 && Bdims == 1)
    {
        // dot product
        top_blob.create(1, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const std::int32_t K = A.w; // assert A.w == B.w
        const float* ptrA = A;
        const float* ptrB = B;

        float sum = 0.f;
        for (std::int32_t k = 0; k < K; k++)
        {
            sum += ptrA[k] * ptrB[k];
        }

        top_blob[0] = sum;
    }
    else if (Adims == 2 && Bdims == 2)
    {
        // matrix multiply
        const std::int32_t M = A.h;
        const std::int32_t N = transB == 0 ? B.w : B.h;

        top_blob.create(N, M, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        MatT BT;
        if (transB == 0)
        {
            BT.create(B.h, B.w, elemsize, opt.workspace_allocator);
            if (BT.empty())
                return -100;

            transpose(B, BT, opt);
        }
        else
        {
            BT = B;
        }

        matmul_transb(A, BT, top_blob, opt);
    }
    else if (Adims == 1 && Bdims == 2)
    {
        // matrix multiply
        const std::int32_t N = transB == 0 ? B.w : B.h;

        MatT top_blob1(N, 1, elemsize, opt.blob_allocator);
        if (top_blob1.empty())
            return -100;

        MatT A1 = A.reshape(A.w, 1);

        MatT BT;
        if (transB == 0)
        {
            BT.create(B.h, B.w, elemsize, opt.workspace_allocator);
            if (BT.empty())
                return -100;

            transpose(B, BT, opt);
        }
        else
        {
            BT = B;
        }

        matmul_transb(A1, BT, top_blob1, opt);

        top_blob = top_blob1.reshape(N);
    }
    else if (Adims == 2 && Bdims == 1)
    {
        // matrix multiply
        const std::int32_t M = A.h;

        MatT top_blob1(1, M, elemsize, opt.blob_allocator);
        if (top_blob1.empty())
            return -100;

        MatT BT = B.reshape(B.w, 1);

        matmul_transb(A, BT, top_blob1, opt);

        top_blob = top_blob1.reshape(M);
    }
    else if (Adims == 1 && Bdims > 2)
    {
        // batched matrix multiply
        const std::int32_t N = transB == 0 ? B.w : B.h;
        const std::int32_t batch_size = B.d * B.c;

        MatT top_blob1(N, 1, batch_size, elemsize, opt.blob_allocator);
        if (top_blob1.empty())
            return -100;

        MatT A1 = A.reshape(A.w, 1);
        MatT B1 = B.reshape(B.w, B.h, batch_size);

        for (std::int32_t p = 0; p < batch_size; p++)
        {
            MatT BT;
            if (transB == 0)
            {
                BT.create(B.h, B.w, elemsize, opt.workspace_allocator);
                if (BT.empty())
                    return -100;

                transpose(B1.channel(p), BT, opt);
            }
            else
            {
                BT = B1.channel(p);
            }

            MatT top_blob1_p = top_blob1.channel(p);
            matmul_transb(A1, BT, top_blob1_p, opt);
        }

        if (Bdims == 3)
            top_blob = top_blob1.reshape(N, B.d * B.c);
        else
            top_blob = top_blob1.reshape(N, B.d, B.c);
    }
    else if (Adims > 2 && Bdims == 1)
    {
        // batched matrix multiply
        const std::int32_t M = A.h;
        const std::int32_t batch_size = A.d * A.c;

        MatT top_blob1(1, M, batch_size, elemsize, opt.blob_allocator);
        if (top_blob1.empty())
            return -100;

        MatT A1 = A.reshape(A.w, A.h, batch_size);
        MatT BT = B.reshape(B.w, 1);

        for (std::int32_t p = 0; p < batch_size; p++)
        {
            MatT top_blob1_p = top_blob1.channel(p);
            matmul_transb(A1.channel(p), BT, top_blob1_p, opt);
        }

        if (Adims == 3)
            top_blob = top_blob1.reshape(M, A.d * A.c);
        else
            top_blob = top_blob1.reshape(M, A.d, A.c);
    }
    else if (max_ABdims == 3)
    {
        MatT A1 = Adims == 2 ? A.reshape(A.w, A.h, 1) : A;
        MatT B1 = Bdims == 2 ? B.reshape(B.w, B.h, 1) : B;

        const std::int32_t M = A1.h;
        const std::int32_t N = transB == 0 ? B1.w : B1.h;
        const std::int32_t batch_size = std::max(A1.c, B1.c);

        top_blob.create(N, M, batch_size, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        MatT BT0;
        if (B1.c == 1)
        {
            if (transB == 0)
            {
                BT0.create(B1.h, B1.w, elemsize, opt.workspace_allocator);
                if (BT0.empty())
                    return -100;

                transpose(B1.channel(0), BT0, opt);
            }
            else
            {
                BT0 = B1.channel(0);
            }
        }

        for (std::int32_t p = 0; p < batch_size; p++)
        {
            std::int32_t Ap = A1.c == 1 ? 0 : p;
            std::int32_t Bp = B1.c == 1 ? 0 : p;

            MatT BT;
            if (B1.c == 1)
            {
                BT = BT0;
            }
            else
            {
                if (transB == 0)
                {
                    BT.create(B1.h, B1.w, elemsize, opt.workspace_allocator);
                    if (BT.empty())
                        return -100;

                    transpose(B1.channel(Bp), BT, opt);
                }
                else
                {
                    BT = B1.channel(Bp);
                }
            }

            MatT top_blob_p = top_blob.channel(p);
            matmul_transb(A1.channel(Ap), BT, top_blob_p, opt);
        }
    }
    else if (max_ABdims == 4)
    {
        MatT A1 = Adims == 3 ? A.reshape(A.w, A.h, A.c, 1) : A;
        MatT B1 = Bdims == 3 ? B.reshape(B.w, B.h, B.c, 1) : B;

        const std::int32_t M = A1.h;
        const std::int32_t N = transB == 0 ? B1.w : B1.h;
        const std::int32_t batch_size_d = std::max(A1.d, B1.d);
        const std::int32_t batch_size_c = std::max(A1.c, B1.c);

        top_blob.create(N, M, batch_size_d, batch_size_c, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        MatT BT00;
        if (B1.d == 1 && B1.c == 1)
        {
            if (transB == 0)
            {
                BT00.create(B1.h, B1.w, elemsize, opt.workspace_allocator);
                if (BT00.empty())
                    return -100;

                transpose(B1.channel(0).depth(0), BT00, opt);
            }
            else
            {
                BT00 = B1.channel(0).depth(0);
            }
        }

        for (std::int32_t p = 0; p < batch_size_c; p++)
        {
            std::int32_t Ap = A1.c == 1 ? 0 : p;
            std::int32_t Bp = B1.c == 1 ? 0 : p;

            MatT BT0x;
            if (B1.d == 1 && B1.c != 1)
            {
                if (transB == 0)
                {
                    BT0x.create(B1.h, B1.w, elemsize, opt.workspace_allocator);
                    if (BT0x.empty())
                        return -100;

                    transpose(B1.channel(Bp).depth(0), BT0x, opt);
                }
                else
                {
                    BT0x = B1.channel(Bp).depth(0);
                }
            }

            for (std::int32_t q = 0; q < batch_size_d; q++)
            {
                std::int32_t Ad = A1.d == 1 ? 0 : q;
                std::int32_t Bd = B1.d == 1 ? 0 : q;

                MatT BT;
                if (B1.d == 1 && B1.c == 1)
                {
                    BT = BT00;
                }
                else if (B1.d == 1 && B1.c != 1)
                {
                    BT = BT0x;
                }
                else
                {
                    if (transB == 0)
                    {
                        BT.create(B1.h, B1.w, elemsize, opt.workspace_allocator);
                        if (BT.empty())
                            return -100;

                        transpose(B1.channel(Bp).depth(Bd), BT, opt);
                    }
                    else
                    {
                        BT = B1.channel(Bp).depth(Bd);
                    }
                }

                MatT top_blob_p_q = top_blob.channel(p).depth(q);
                matmul_transb(A1.channel(Ap).depth(Ad), BT, top_blob_p_q, opt);
            }
        }
    }
    else
    {

        return -1;
    }

    return 0;
}


#undef MLAYER_MATMUL_TEMPLATE

}  // namespace mlayer

#endif  // MLAYER_MATMUL_HPP
