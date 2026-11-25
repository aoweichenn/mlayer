// Copyright 2020 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_GEMM_HPP
#define MLAYER_GEMM_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>

#if defined(__AVX2__)
#include <immintrin.h>
#endif
#include <cstdint>
#include <vector>

#include <mlayer/int8_quantize.hpp>

namespace mlayer {

// Standard fp32/fp16/weight-block-quantized Gemm. Weight-block int4/int6/int8
// weights are unpacked from ncnn's little-endian format; the classic int8
// (quantize_term 1..3) path is still not ported.
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
        this->weight_block_quantize = false;
        this->weight_block_quantize_bits = 0;
        this->weight_block_quantize_block_size = 0;
        this->weight_block_quantize_has_input_scale = false;
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

        this->weight_block_quantize = false;
        if (this->quantize_term != 0) {
            this->weight_block_quantize_bits = this->quantize_term / 100;
            const std::int32_t format_code = this->quantize_term % 100 / 10;
            const std::int32_t block_size_code = this->quantize_term % 10;
            if ((this->weight_block_quantize_bits != 4 && this->weight_block_quantize_bits != 6 &&
                 this->weight_block_quantize_bits != 8) ||
                (format_code != 0 && format_code != 1) || block_size_code < 0 || block_size_code > 2) {
                return -1;
            }
            this->weight_block_quantize_block_size =
                block_size_code == 0 ? 32 : (block_size_code == 1 ? 64 : 128);
            this->weight_block_quantize_has_input_scale = format_code == 1;
            this->weight_block_quantize = true;
            if (this->constantB == 0 || this->transB != 1) {
                return -1;
            }
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
            if (this->weight_block_quantize) {
                const std::int32_t packed_k_bytes =
                    (this->constantK * this->weight_block_quantize_bits + 7) / 8;
                this->B_data = mb.load(packed_k_bytes, this->constantN, this->weight_block_quantize_bits);
            } else {
                this->B_data = this->transB == 0 ? mb.load(this->constantN, this->constantK, 0)
                                                 : mb.load(this->constantK, this->constantN, 0);
            }
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
        if (this->weight_block_quantize) {
            const std::int32_t block_count =
                (this->constantK + this->weight_block_quantize_block_size - 1) /
                this->weight_block_quantize_block_size;
            this->B_data_quantize_scales = mb.load(block_count, this->constantN, 1);
            if (this->B_data_quantize_scales.empty()) {
                return -100;
            }
            if (this->weight_block_quantize_has_input_scale) {
                this->B_data_input_scales = mb.load(this->constantK, 1);
                if (this->B_data_input_scales.empty()) {
                    return -100;
                }
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
        if (this->weight_block_quantize) {
            return this->forward_weight_block_quantize(bottom_blobs, top_blobs, opt);
        }

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
        const std::size_t out_hstep = top.dims == 3 ? top.cstep : static_cast<std::size_t>(top.w);
        const std::size_t A_hstep = A.dims == 3 ? A.cstep : static_cast<std::size_t>(A.w);
        const std::size_t BT_hstep = BT.dims == 3 ? BT.cstep : static_cast<std::size_t>(BT.w);
        const float* ptrC = C.empty() ? nullptr : static_cast<const float*>(C);
        const float* ptrA_all = static_cast<const float*>(A);
        const float* ptrBT_all = static_cast<const float*>(BT);

#pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t mn = 0; mn < M * N; ++mn) {
            const std::int32_t i = mn / N;
            const std::int32_t j = mn % N;
            const float* ptrA = ptrA_all + static_cast<std::size_t>(i) * A_hstep;
            const float* ptrBT = ptrBT_all + static_cast<std::size_t>(j) * BT_hstep;

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
#if defined(__AVX2__)
            {
                __m256 acc = _mm256_setzero_ps();
                std::int32_t k = 0;
                for (; k + 8 <= K; k += 8) {
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(ptrA + k), _mm256_loadu_ps(ptrBT + k), acc);
                }
                __m128 lo = _mm256_castps256_ps128(acc);
                __m128 hi = _mm256_extractf128_ps(acc, 1);
                __m128 s128 = _mm_add_ps(lo, hi);
                s128 = _mm_hadd_ps(s128, s128);
                s128 = _mm_hadd_ps(s128, s128);
                sum += _mm_cvtss_f32(s128);
                for (; k < K; ++k) {
                    sum += ptrA[k] * ptrBT[k];
                }
            }
#else
            for (std::int32_t k = 0; k < K; ++k) {
                sum += ptrA[k] * ptrBT[k];
            }
#endif
            sum *= this->alpha;

            if (this->output_transpose != 0) {
                top[static_cast<std::size_t>(j) * out_hstep + static_cast<std::size_t>(i)] = sum;
            } else {
                top[static_cast<std::size_t>(i) * out_hstep + static_cast<std::size_t>(j)] = sum;
            }
        }
        return 0;
    }

private:
    std::int32_t forward_weight_block_quantize(const std::vector<MatT>& bottom_blobs,
                                               std::vector<MatT>& top_blobs, const OptionT& opt) const {
        const MatT& A = bottom_blobs[0];
        const std::int32_t K = this->transA != 0
                                   ? (A.dims == 3 ? A.c : A.h) * A.elempack
                                   : A.w;
        const std::int32_t M = this->transA != 0
                                   ? A.w
                                   : (A.dims == 3 ? A.c : A.h) * A.elempack;
        const std::int32_t N = this->constantN;
        const std::int32_t block_size = this->weight_block_quantize_block_size;

        MatT C;
        std::int32_t broadcast_type_C = 0;
        if (this->constantC != 0) {
            C = this->C_data;
            broadcast_type_C = this->constant_broadcast_type_C;
        } else if (bottom_blobs.size() == 2) {
            C = bottom_blobs[1];
        }

        MatT& top = top_blobs[0];
        if (this->output_transpose != 0) {
            top.create(this->output_N1M != 0 ? M : M, this->output_N1M != 0 ? 1 : N, 4u, opt.blob_allocator);
            if (this->output_N1M != 0) {
                top = top.reshape(M, 1, N, opt.blob_allocator);
            }
        } else {
            if (this->output_N1M != 0) {
                top.create(N, 1, M, 4u, opt.blob_allocator);
            } else {
                top.create(N, M, 4u, opt.blob_allocator);
            }
        }
        if (top.empty()) {
            return -100;
        }

        return this->weight_block_quantize_gemm(
            A, this->transA, this->B_data, this->B_data_quantize_scales, this->B_data_input_scales, C, top, M,
            N, K, block_size, this->weight_block_quantize_bits, this->alpha, this->beta, broadcast_type_C,
            this->output_transpose, opt);
    }

    static std::int32_t weight_block_quantize_gemm(
        const MatT& A, std::int32_t transA, const MatT& BT, const MatT& BT_scales, const MatT& input_scales,
        const MatT& C, MatT& top_blob, std::int32_t M, std::int32_t N, std::int32_t K,
        std::int32_t block_size, std::int32_t bits, float alpha, float beta,
        std::int32_t broadcast_type_C, std::int32_t output_transpose, const OptionT& opt) {
        const std::int32_t block_count = (K + block_size - 1) / block_size;
        // Process kMTile activation rows per weight row read. Prefill passes
        // M=seqlen and previously re-read every weight row M times; tiling
        // reduces weight traffic by up to kMTile with identical arithmetic
        // (per output element the block/k accumulation order is unchanged).
        constexpr std::int32_t kMTile = 8;

        const float* ptrC = C.empty() ? nullptr : static_cast<const float*>(C);
        const std::size_t out_hstep =
            top_blob.dims == 3 ? top_blob.cstep : static_cast<std::size_t>(top_blob.w);
        const std::size_t a_hstep = A.dims == 3 ? A.cstep : static_cast<std::size_t>(A.w);
        const float* input_scale_ptr =
            input_scales.empty() ? nullptr : static_cast<const float*>(input_scales);
        const std::int32_t mt_count = (M + kMTile - 1) / kMTile;

        const auto init_sum = [&](std::int32_t i, std::int32_t j) -> float {
            if (ptrC == nullptr) {
                return 0.0F;
            }
            float sum = 0.0F;
            if (broadcast_type_C == 0) {
                sum = ptrC[0];
            } else if (broadcast_type_C == 1 || broadcast_type_C == 2) {
                sum = ptrC[static_cast<std::size_t>(i)];
            } else if (broadcast_type_C == 3) {
                sum = ptrC[static_cast<std::size_t>(i) * static_cast<std::size_t>(N) +
                           static_cast<std::size_t>(j)];
            } else {
                sum = ptrC[static_cast<std::size_t>(j)];
            }
            return sum * beta;
        };
        const auto store_value = [&](std::int32_t i, std::int32_t j, float v) {
            if (output_transpose != 0) {
                top_blob[static_cast<std::size_t>(j) * out_hstep + static_cast<std::size_t>(i)] = v;
            } else {
                top_blob[static_cast<std::size_t>(i) * out_hstep + static_cast<std::size_t>(j)] = v;
            }
        };

        if (bits != 8) {
            // ncnn dequantizes 4/6-bit weights to fp32 and multiplies them with
            // the fp32 activation (no activation quantization on this path).
#pragma omp parallel for num_threads(opt.num_threads)
            for (std::int32_t work = 0; work < mt_count * N; ++work) {
                const std::int32_t t = work / N;
                const std::int32_t j = work % N;
                const std::int32_t i0 = t * kMTile;
                const std::int32_t mi = std::min(kMTile, M - i0);

                float sums[kMTile];
                const float* arows[kMTile];
                for (std::int32_t ii = 0; ii < mi; ++ii) {
                    sums[ii] = init_sum(i0 + ii, j);
                    arows[ii] = static_cast<const float*>(A) +
                                static_cast<std::size_t>(i0 + ii) * a_hstep;
                }

                const auto* ptrB = BT.template row<const unsigned char>(j);
                const float* scale_ptr = BT_scales.row(j);

                for (std::int32_t k0 = 0; k0 < K; k0 += block_size) {
                    const std::int32_t max_kk = block_size < K - k0 ? block_size : K - k0;
                    const float descale = 1.0F / scale_ptr[k0 / block_size];
                    for (std::int32_t kk = 0; kk < max_kk; ++kk) {
                        const std::int32_t k = k0 + kk;
                        const auto q = detail::unpack_weight_block(ptrB, k, bits);
                        const float w = static_cast<float>(q) * descale;
                        const float in_scale = input_scale_ptr != nullptr ? input_scale_ptr[k] : 1.0F;
                        for (std::int32_t ii = 0; ii < mi; ++ii) {
                            sums[ii] += arows[ii][k] * in_scale * w;
                        }
                    }
                }
                for (std::int32_t ii = 0; ii < mi; ++ii) {
                    store_value(i0 + ii, j, sums[ii] * alpha);
                }
            }
            return 0;
        }

        MatT A_int8;
        A_int8.create(K, M, 1u, opt.workspace_allocator);
        if (A_int8.empty()) {
            return -100;
        }
        MatT A_descales;
        A_descales.create(block_count, M, 4u, opt.workspace_allocator);
        if (A_descales.empty()) {
            return -100;
        }

        const bool has_input_scale = !input_scales.empty();

#pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t i = 0; i < M; ++i) {
            auto* outptr = A_int8.template row<std::int8_t>(i);
            float* descale_ptr = A_descales.row(i);
            for (std::int32_t g = 0; g < block_count; ++g) {
                const std::int32_t k0 = g * block_size;
                const std::int32_t max_kk = block_size < K - k0 ? block_size : K - k0;
                float absmax = 0.0F;
                for (std::int32_t kk = 0; kk < max_kk; ++kk) {
                    const std::int32_t k = k0 + kk;
                    float v = transA != 0
                                  ? static_cast<const float*>(A)[static_cast<std::size_t>(k) * a_hstep +
                                                                 static_cast<std::size_t>(i)]
                                  : static_cast<const float*>(A)[static_cast<std::size_t>(i) * a_hstep +
                                                                 static_cast<std::size_t>(k)];
                    if (has_input_scale) {
                        v *= input_scale_ptr[k];
                    }
                    absmax = std::max(absmax, std::fabs(v));
                }
                if (absmax == 0.0F) {
                    descale_ptr[g] = 0.0F;
                    for (std::int32_t kk = 0; kk < max_kk; ++kk) {
                        outptr[k0 + kk] = 0;
                    }
                    continue;
                }
                const float scale = 127.0F / absmax;
                descale_ptr[g] = absmax / 127.0F;
                for (std::int32_t kk = 0; kk < max_kk; ++kk) {
                    const std::int32_t k = k0 + kk;
                    float v = transA != 0
                                  ? static_cast<const float*>(A)[static_cast<std::size_t>(k) * a_hstep +
                                                                 static_cast<std::size_t>(i)]
                                  : static_cast<const float*>(A)[static_cast<std::size_t>(i) * a_hstep +
                                                                 static_cast<std::size_t>(k)];
                    if (has_input_scale) {
                        v *= input_scale_ptr[k];
                    }
                    const auto q = static_cast<std::int32_t>(std::round(v * scale));
                    outptr[k0 + kk] = static_cast<std::int8_t>(std::max(-127, std::min(127, q)));
                }
            }
        }

#pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t work = 0; work < mt_count * N; ++work) {
            const std::int32_t t = work / N;
            const std::int32_t j = work % N;
            const std::int32_t i0 = t * kMTile;
            const std::int32_t mi = std::min(kMTile, M - i0);

            float sums[kMTile];
            const std::int8_t* arows[kMTile];
            const float* adescs[kMTile];
            for (std::int32_t ii = 0; ii < mi; ++ii) {
                sums[ii] = init_sum(i0 + ii, j);
                arows[ii] = A_int8.template row<const std::int8_t>(i0 + ii);
                adescs[ii] = A_descales.row(i0 + ii);
            }

            const auto* ptrB = BT.template row<const std::int8_t>(j);
            const float* b_scale_ptr = BT_scales.row(j);

            for (std::int32_t g = 0; g < block_count; ++g) {
                const std::int32_t k0 = g * block_size;
                const std::int32_t max_kk = block_size < K - k0 ? block_size : K - k0;
                std::int32_t sums_int32[kMTile] = {0};
                std::int32_t kk = 0;
#if defined(__AVX2__)
                const __m256i ones = _mm256_set1_epi16(1);
                __m256i accv[kMTile];
                for (std::int32_t ii = 0; ii < mi; ++ii) {
                    accv[ii] = _mm256_setzero_si256();
                }
                for (; kk + 32 <= max_kk; kk += 32) {
                    _mm_prefetch(reinterpret_cast<const char*>(ptrB + k0 + kk + 256), _MM_HINT_T0);
                    const __m256i b8 =
                        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptrB + k0 + kk));
                    for (std::int32_t ii = 0; ii < mi; ++ii) {
                        const __m256i a8 =
                            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(arows[ii] + k0 + kk));
                        const __m256i a_abs = _mm256_abs_epi8(a8);
                        const __m256i b_signed = _mm256_sign_epi8(b8, a8);
                        const __m256i pairs = _mm256_maddubs_epi16(a_abs, b_signed);
                        accv[ii] = _mm256_add_epi32(accv[ii], _mm256_madd_epi16(pairs, ones));
                    }
                }
                for (std::int32_t ii = 0; ii < mi; ++ii) {
                    __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(accv[ii]),
                                                   _mm256_extracti128_si256(accv[ii], 1));
                    sum128 = _mm_add_epi32(sum128, _mm_srli_si128(sum128, 8));
                    sum128 = _mm_add_epi32(sum128, _mm_srli_si128(sum128, 4));
                    sums_int32[ii] = _mm_cvtsi128_si32(sum128);
                }
#endif
                for (; kk < max_kk; ++kk) {
                    const std::int32_t k = k0 + kk;
                    const std::int32_t bv = static_cast<std::int32_t>(ptrB[k]);
                    for (std::int32_t ii = 0; ii < mi; ++ii) {
                        sums_int32[ii] += static_cast<std::int32_t>(arows[ii][k]) * bv;
                    }
                }
                for (std::int32_t ii = 0; ii < mi; ++ii) {
                    sums[ii] += static_cast<float>(sums_int32[ii]) * adescs[ii][g] / b_scale_ptr[g];
                }
            }
            for (std::int32_t ii = 0; ii < mi; ++ii) {
                store_value(i0 + ii, j, sums[ii] * alpha);
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
    bool weight_block_quantize;
    std::int32_t weight_block_quantize_bits;
    std::int32_t weight_block_quantize_block_size;
    bool weight_block_quantize_has_input_scale;
    MatT A_data;
    MatT B_data;
    MatT C_data;
    MatT B_data_quantize_scales;
    MatT B_data_input_scales;
};

}  // namespace mlayer

#endif  // MLAYER_GEMM_HPP
