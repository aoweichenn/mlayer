// Copyright 2020 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_GEMM_HPP
#define MLAYER_GEMM_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mlayer {

// Standard fp32/fp16 Gemm path. Weight-block int8 and classic int8 paths are
// intentionally rejected until the quantized kernels are ported.
template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, typename ModelBinT>
class Gemm : public LayerT {
public:
    Gemm() {
        this->one_blob_only = false;
        this->support_inplace = false;
        this->alpha = 1.0F;
        this->beta = 1.0F;
        this->transA = 0;
        this->transB = 0;
        this->constantA = 0;
        this->constantB = 0;
        this->constantC = 0;
        this->constantM = 0;
        this->constantN = 0;
        this->constantK = 0;
        this->constant_broadcast_type_C = 0;
        this->output_N1M = 0;
        this->output_transpose = 0;
        this->quantize_term = 0;
    }

    std::int32_t load_param(const ParamDictT& pd) override {
        this->alpha = pd.get(0, 1.0F);
        this->beta = pd.get(1, 1.0F);
        this->transA = pd.get(2, 0);
        this->transB = pd.get(3, 0);
        this->constantA = pd.get(4, 0);
        this->constantB = pd.get(5, 0);
        this->constantC = pd.get(6, 0);
        this->constantM = pd.get(7, 0);
        this->constantN = pd.get(8, 0);
        this->constantK = pd.get(9, 0);
        this->constant_broadcast_type_C = pd.get(10, 0);
        this->output_N1M = pd.get(11, 0);
        this->output_transpose = pd.get(14, 0);
        this->quantize_term = pd.get(18, 0);

        if (this->quantize_term != 0) {
            return -1;
        }

        if (this->constantA == 0 && this->constantB == 1 && this->constantC == 1) {
            this->one_blob_only = true;
        }
        if (this->constantA == 1 && this->constantB == 0 && this->constantC == 1) {
            this->one_blob_only = true;
        }
        if (this->constantA == 1 && this->constantB == 1 && this->constantC == 0) {
            this->one_blob_only = true;
        }
        return 0;
    }

    std::int32_t load_model(const ModelBinT& mb) override {
        if (this->constantA != 0) {
            this->A_data = this->transA == 0 ? mb.load(this->constantK, this->constantM, 0)
                                             : mb.load(this->constantM, this->constantK, 0);
            if (this->A_data.empty()) {
                return -100;
            }
        }
        if (this->constantB != 0) {
            this->B_data = this->transB == 0 ? mb.load(this->constantN, this->constantK, 0)
                                             : mb.load(this->constantK, this->constantN, 0);
            if (this->B_data.empty()) {
                return -100;
            }
        }
        if (this->constantC != 0 && this->constant_broadcast_type_C != -1) {
            if (this->constant_broadcast_type_C == 0) {
                this->C_data = mb.load(1, 0);
            } else if (this->constant_broadcast_type_C == 1) {
                this->C_data = mb.load(this->constantM, 0);
            } else if (this->constant_broadcast_type_C == 2) {
                this->C_data = mb.load(1, this->constantM, 0);
            } else if (this->constant_broadcast_type_C == 3) {
                this->C_data = mb.load(this->constantN, this->constantM, 0);
            } else if (this->constant_broadcast_type_C == 4) {
                this->C_data = mb.load(this->constantN, 1, 0);
            }
            if (this->C_data.empty()) {
                return -100;
            }
        }
        return 0;
    }

    std::int32_t forward(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const override {
        std::vector<MatT> bottoms(1);
        bottoms[0] = bottom_blob;
        std::vector<MatT> tops(1);
        const std::int32_t ret = this->forward(bottoms, tops, opt);
        top_blob = tops[0];
        return ret;
    }

    std::int32_t forward(const std::vector<MatT>& bottom_blobs, std::vector<MatT>& top_blobs,
                         const OptionT& opt) const override {
        const MatT& A0 = this->constantA != 0 ? this->A_data : bottom_blobs[0];
        const MatT& B0 = this->constantB != 0 ? this->B_data
                         : this->constantA != 0 ? bottom_blobs[0]
                                                : bottom_blobs[1];
        const std::size_t elemsize = A0.elemsize;

        MatT A;
        if (this->transA == 0) {
            A = A0;
        } else {
            A.create(A0.dims == 3 ? A0.c : A0.h, A0.w, elemsize, opt.workspace_allocator);
            if (A.empty()) {
                return -100;
            }
            const std::size_t hstep = A0.dims == 3 ? A0.cstep : static_cast<std::size_t>(A0.w);
            for (std::int32_t i = 0; i < A.h; ++i) {
                float* ptr = A.row(i);
                for (std::int32_t j = 0; j < A.w; ++j) {
                    ptr[j] = A0[static_cast<std::size_t>(j) * hstep + static_cast<std::size_t>(i)];
                }
            }
        }

        MatT BT;
        if (this->transB == 0) {
            BT.create(B0.dims == 3 ? B0.c : B0.h, B0.w, elemsize, opt.workspace_allocator);
            if (BT.empty()) {
                return -100;
            }
            const std::size_t hstep = B0.dims == 3 ? B0.cstep : static_cast<std::size_t>(B0.w);
            for (std::int32_t i = 0; i < BT.h; ++i) {
                float* ptr = BT.row(i);
                for (std::int32_t j = 0; j < BT.w; ++j) {
                    ptr[j] = B0[static_cast<std::size_t>(j) * hstep + static_cast<std::size_t>(i)];
                }
            }
        } else {
            BT = B0;
        }

        const std::int32_t M = A.dims == 3 ? A.c : A.h;
        const std::int32_t N = BT.dims == 3 ? BT.c : BT.h;

        MatT C;
        std::int32_t broadcast_type_C = 0;
        if (this->constantC != 0) {
            C = this->C_data;
            broadcast_type_C = this->constant_broadcast_type_C;
        } else {
            if (this->constantA != 0 && this->constantB != 0 && bottom_blobs.size() == 1) {
                C = bottom_blobs[0];
            } else if ((this->constantA != 0 || this->constantB != 0) && bottom_blobs.size() == 2) {
                C = bottom_blobs[1];
            } else if (bottom_blobs.size() == 3) {
                C = bottom_blobs[2];
            }

            if (!C.empty()) {
                if (C.dims == 1 && C.w == 1) {
                    broadcast_type_C = 0;
                } else if (C.dims == 1 && C.w == M) {
                    broadcast_type_C = 1;
                } else if (C.dims == 1 && C.w == N) {
                    broadcast_type_C = 4;
                } else if (C.dims == 2 && C.w == 1 && C.h == M) {
                    broadcast_type_C = 2;
                } else if (C.dims == 2 && C.w == N && C.h == M) {
                    broadcast_type_C = 3;
                } else if (C.dims == 2 && C.w == N && C.h == 1) {
                    broadcast_type_C = 4;
                } else {
                    return -1;
                }
            }
        }

        MatT& top = top_blobs[0];
        if (this->output_transpose != 0) {
            if (this->output_N1M != 0) {
                top.create(M, 1, N, elemsize, opt.blob_allocator);
            } else {
                top.create(M, N, elemsize, opt.blob_allocator);
            }
        } else {
            if (this->output_N1M != 0) {
                top.create(N, 1, M, elemsize, opt.blob_allocator);
            } else {
                top.create(N, M, elemsize, opt.blob_allocator);
            }
        }
        if (top.empty()) {
            return -100;
        }

        const std::int32_t K = A.w;
#pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t i = 0; i < M; ++i) {
            const std::size_t out_hstep = top.dims == 3 ? top.cstep : static_cast<std::size_t>(top.w);
            const std::size_t A_hstep = A.dims == 3 ? A.cstep : static_cast<std::size_t>(A.w);
            const std::size_t BT_hstep = BT.dims == 3 ? BT.cstep : static_cast<std::size_t>(BT.w);
            const float* ptrA = static_cast<const float*>(A) + static_cast<std::size_t>(i) * A_hstep;
            const float* ptrC = C.empty() ? nullptr : static_cast<const float*>(C);

            for (std::int32_t j = 0; j < N; ++j) {
                const float* ptrBT = static_cast<const float*>(BT) + static_cast<std::size_t>(j) * BT_hstep;
                float sum = 0.0F;
                if (ptrC != nullptr) {
                    if (broadcast_type_C == 0) {
                        sum = ptrC[0];
                    } else if (broadcast_type_C == 1 || broadcast_type_C == 2) {
                        sum = ptrC[static_cast<std::size_t>(i)];
                    } else if (broadcast_type_C == 3) {
                        sum = ptrC[static_cast<std::size_t>(i) * static_cast<std::size_t>(N) +
                                   static_cast<std::size_t>(j)];
                    } else if (broadcast_type_C == 4) {
                        sum = ptrC[static_cast<std::size_t>(j)];
                    }
                    sum *= this->beta;
                }
                for (std::int32_t k = 0; k < K; ++k) {
                    sum += ptrA[k] * ptrBT[k];
                }
                sum *= this->alpha;

                if (this->output_transpose != 0) {
                    top[static_cast<std::size_t>(j) * out_hstep + static_cast<std::size_t>(i)] = sum;
                } else {
                    top[static_cast<std::size_t>(i) * out_hstep + static_cast<std::size_t>(j)] = sum;
                }
            }
        }
        return 0;
    }

public:
    float alpha;
    float beta;
    std::int32_t transA;
    std::int32_t transB;
    std::int32_t constantA;
    std::int32_t constantB;
    std::int32_t constantC;
    std::int32_t constantM;
    std::int32_t constantN;
    std::int32_t constantK;
    std::int32_t constant_broadcast_type_C;
    std::int32_t output_N1M;
    std::int32_t output_transpose;
    std::int32_t quantize_term;
    MatT A_data;
    MatT B_data;
    MatT C_data;
};

}  // namespace mlayer

#endif  // MLAYER_GEMM_HPP
