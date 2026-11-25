// Copyright 2017 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_CONVOLUTION_HPP
#define MLAYER_CONVOLUTION_HPP

#include <climits>
#include <cmath>

#if defined(__AVX2__)
#include <immintrin.h>
#endif
#include <cstddef>
#include <cstdint>
#include <vector>

#include <mlayer/border.hpp>
#include <mlayer/quantize.hpp>

namespace mlayer {

template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, typename ModelBinT, bool Validate,
         bool Int8, float (*Activation)(float, std::int32_t, const MatT&),
         void (*QuantizeInt8)(const MatT&, MatT&, const MatT&, const OptionT&),
         void (*CopyMakeBorder)(const MatT&, MatT&, std::int32_t, std::int32_t, std::int32_t, std::int32_t, std::int32_t,
                                float, const OptionT&)>
class Convolution : public LayerT {
public:
    Convolution();

    std::int32_t load_param(const ParamDictT& pd) override;
    std::int32_t load_model(const ModelBinT& mb) override;
    std::int32_t forward(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const override;
    std::int32_t forward(const std::vector<MatT>& bottom_blobs, std::vector<MatT>& top_blobs, const OptionT& opt) const override;

protected:
    void make_padding(const MatT& bottom_blob, MatT& bottom_blob_bordered, const OptionT& opt) const;
    void make_padding(const MatT& bottom_blob, MatT& bottom_blob_bordered, std::int32_t kernel_w, std::int32_t kernel_h,
                      const OptionT& opt) const;
    std::int32_t forward_int8(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const;

    static std::int32_t convolution(const MatT& bottom_blob, MatT& top_blob, const MatT& weight_data, const MatT& bias_data, std::int32_t kernel_w, std::int32_t kernel_h, std::int32_t stride_w, std::int32_t stride_h, std::int32_t dilation_w, std::int32_t dilation_h, std::int32_t activation_type, const MatT& activation_params, const OptionT& opt);

    static std::int32_t convolution_1x1(const MatT& bottom_blob, MatT& top_blob, const MatT& weight_data, const MatT& bias_data, std::int32_t activation_type, const MatT& activation_params, const OptionT& opt);

    static float fused_activation(float v, std::int32_t type, const MatT& params)
    {
        return type == 0 ? v : Activation(v, type, params);
    }

#if defined(__AVX2__)
    static __m128 hsum8(__m256 v)
    {
        const __m128 lo = _mm256_castps256_ps128(v);
        const __m128 hi = _mm256_extractf128_ps(v, 1);
        const __m128 s = _mm_add_ps(lo, hi);
        return _mm_hadd_ps(_mm_hadd_ps(s, s), _mm_hadd_ps(s, s));
    }
#endif



    static std::int32_t convolution_im2col(const MatT& bottom_blob, MatT& top_blob, const MatT& weight_data, const MatT& bias_data, std::int32_t kernel_w, std::int32_t kernel_h, std::int32_t stride_w, std::int32_t stride_h, std::int32_t dilation_w, std::int32_t dilation_h, std::int32_t activation_type, const MatT& activation_params, const OptionT& opt);
    static std::int32_t convolution_gemm_pack(const MatT& bottom_blob, MatT& top_blob, const MatT& weight_data, const MatT& bias_data, std::int32_t kernel_w, std::int32_t kernel_h, std::int32_t stride_w, std::int32_t stride_h, std::int32_t dilation_w, std::int32_t dilation_h, std::int32_t activation_type, const MatT& activation_params, const OptionT& opt);

public:
    std::int32_t num_output;
    std::int32_t kernel_w;
    std::int32_t kernel_h;
    std::int32_t dilation_w;
    std::int32_t dilation_h;
    std::int32_t stride_w;
    std::int32_t stride_h;

    MatT weight_data;
    MatT bias_data;

    std::int32_t pad_left;
    std::int32_t pad_right;
    std::int32_t pad_top;
    std::int32_t pad_bottom;
    float pad_value;

    std::int32_t bias_term;
    std::int32_t weight_data_size;
    std::int32_t int8_scale_term;
    std::int32_t activation_type;
    MatT activation_params;

    MatT weight_data_int8_scales;
    MatT bottom_blob_int8_scales;
    MatT top_blob_int8_scales;

    bool dynamic_weight;
};

#define MLAYER_CONVOLUTION_TEMPLATE \
    template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, typename ModelBinT, bool Validate, \
             bool Int8, float (*Activation)(float, std::int32_t, const MatT&), \
             void (*QuantizeInt8)(const MatT&, MatT&, const MatT&, const OptionT&), \
             void (*CopyMakeBorder)(const MatT&, MatT&, std::int32_t, std::int32_t, std::int32_t, std::int32_t, \
                                    std::int32_t, float, const OptionT&)>

MLAYER_CONVOLUTION_TEMPLATE
Convolution<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Int8, Activation, QuantizeInt8, CopyMakeBorder>::Convolution()
{
    this->one_blob_only = true;
    this->support_inplace = false;
}

MLAYER_CONVOLUTION_TEMPLATE
std::int32_t Convolution<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Int8, Activation, QuantizeInt8, CopyMakeBorder>::load_param(const ParamDictT& pd)
{
    num_output = pd.get(0, 0);
    kernel_w = pd.get(1, 0);
    kernel_h = pd.get(11, kernel_w);
    dilation_w = pd.get(2, 1);
    dilation_h = pd.get(12, dilation_w);
    stride_w = pd.get(3, 1);
    stride_h = pd.get(13, stride_w);
    pad_left = pd.get(4, 0);
    pad_right = pd.get(15, pad_left);
    pad_top = pd.get(14, pad_left);
    pad_bottom = pd.get(16, pad_top);
    pad_value = pd.get(18, 0.f);
    bias_term = pd.get(5, 0);
    weight_data_size = pd.get(6, 0);
    int8_scale_term = pd.get(8, 0);
    activation_type = pd.get(9, 0);
    activation_params = pd.get(10, MatT());

    dynamic_weight = pd.get(19, 0);

    const std::int32_t activation_params_type = pd.type(10);
if constexpr (Validate) {
    if (activation_params_type != 0 && activation_params_type != 4 && activation_params_type != 5 && activation_params_type != 6)
        return -1;

    if ((activation_params.dims != 0 || activation_params.w != 0 || activation_params.data) && (activation_params.dims != 1 || activation_params.w < 0 || activation_params.elempack != 1 || activation_params.elemsize != 4u || (activation_params.w > 0 && !activation_params.data)))
        return -1;

    if (activation_type < 0 || activation_type > 6)
        return -1;

    if ((activation_type == 2 && activation_params.w < 1) || ((activation_type == 3 || activation_type == 6) && activation_params.w < 2))
        return -1;

    if (dilation_w <= 0 || stride_w <= 0 || dilation_h <= 0 || stride_h <= 0)
        return -1;

    if (!dynamic_weight)
    {
        if (kernel_w <= 0 || kernel_w - 1 > (INT_MAX - 1) / dilation_w || kernel_h <= 0 || kernel_h - 1 > (INT_MAX - 1) / dilation_h || kernel_w > INT_MAX / kernel_h)
            return -1;

        if (num_output <= 0 || weight_data_size <= 0 || weight_data_size % num_output != 0 || (weight_data_size / num_output) % (kernel_w * kernel_h) != 0)
            return -1;
    }
}

    // convert integer text arrays without modifying the shared data
    if (activation_params_type == 5 && !activation_params.empty())
    {
        MatT converted(activation_params.w);
        if (converted.empty())
            return -100;

        const std::int32_t* p = activation_params;
        for (std::int32_t i = 0; i < activation_params.w; i++)
            converted[i] = (float)p[i];

        activation_params = converted;
    }

    if (dynamic_weight)
    {
        this->one_blob_only = false;
    }

    if (int8_scale_term)
    {
        if constexpr (Int8)
        {
            this->support_int8_storage = true;
        }
        else
        {
            return -1;
        }
    }

    return 0;
}

MLAYER_CONVOLUTION_TEMPLATE
std::int32_t Convolution<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Int8, Activation, QuantizeInt8, CopyMakeBorder>::load_model(const ModelBinT& mb)
{
    if (dynamic_weight)
        return 0;

    weight_data = mb.load(weight_data_size, 0);
    if (weight_data.empty())
        return -100;

    if (bias_term)
    {
        bias_data = mb.load(num_output, 1);
        if (bias_data.empty())
            return -100;
    }

    if constexpr (Int8)
    {
        if (int8_scale_term)
        {
            weight_data_int8_scales = mb.load(num_output, 1);
            bottom_blob_int8_scales = mb.load(1, 1);
        }

        if (int8_scale_term > 100)
        {
            top_blob_int8_scales = mb.load(1, 1);
        }
    }

    if constexpr (Int8)
    {
        // runtime quantize the weight data
        if (weight_data.elemsize == (std::size_t)4u && int8_scale_term)
        {
        const std::int32_t maxk = kernel_w * kernel_h;
        const std::int32_t num_input = weight_data_size / num_output / maxk;

        MatT weight_data_r2 = weight_data.reshape(maxk, num_input, num_output);

        MatT weight_data_int8;

        OptionT opt_q;
        opt_q.num_threads = 1;
        opt_q.blob_allocator = static_cast<decltype(opt_q.blob_allocator)>(weight_data.allocator);
        opt_q.use_packing_layout = false;
            QuantizeInt8(weight_data_r2, weight_data_int8, weight_data_int8_scales, opt_q);
            if (weight_data_int8.empty())
                return -100;

            weight_data = weight_data_int8.reshape(weight_data_size);
        }
    }

    return 0;
}

MLAYER_CONVOLUTION_TEMPLATE
std::int32_t Convolution<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Int8, Activation, QuantizeInt8, CopyMakeBorder>::convolution(const MatT& bottom_blob, MatT& top_blob, const MatT& weight_data, const MatT& bias_data, std::int32_t kernel_w, std::int32_t kernel_h, std::int32_t stride_w, std::int32_t stride_h, std::int32_t dilation_w, std::int32_t dilation_h, std::int32_t activation_type, const MatT& activation_params, const OptionT& opt)
{
    const std::int32_t w = bottom_blob.w;
    const std::int32_t inch = bottom_blob.c;

    const std::int32_t outw = top_blob.w;
    const std::int32_t outh = top_blob.h;
    const std::int32_t outch = top_blob.c;

    const std::int32_t bias_term = bias_data.empty() ? 0 : 1;

    const std::int32_t maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<std::int32_t> _space_ofs(maxk);
    std::int32_t* space_ofs = &_space_ofs[0];
    {
        std::int32_t p1 = 0;
        std::int32_t p2 = 0;
        std::int32_t gap = w * dilation_h - kernel_w * dilation_w;
        for (std::int32_t i = 0; i < kernel_h; i++)
        {
            for (std::int32_t j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (std::int32_t p = 0; p < outch; p++)
    {
        float* outptr = top_blob.channel(p);

        for (std::int32_t i = 0; i < outh; i++)
        {
            for (std::int32_t j = 0; j < outw; j++)
            {
                float sum = 0.f;

                if (bias_term)
                    sum = bias_data[p];

                const float* kptr = (const float*)weight_data + maxk * inch * p;

                for (std::int32_t q = 0; q < inch; q++)
                {
                    const MatT m = bottom_blob.channel(q);
                    const float* sptr = m.row(i * stride_h) + j * stride_w;

                    for (std::int32_t k = 0; k < maxk; k++) // 29.23
                    {
                        float val = sptr[space_ofs[k]]; // 20.72
                        float wt = kptr[k];
                        sum += val * wt; // 41.45
                    }

                    kptr += maxk;
                }

                outptr[j] = fused_activation(sum, activation_type, activation_params);
            }

            outptr += outw;
        }
    }

    return 0;
}

MLAYER_CONVOLUTION_TEMPLATE
std::int32_t Convolution<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Int8, Activation, QuantizeInt8, CopyMakeBorder>::convolution_1x1(const MatT& bottom_blob, MatT& top_blob, const MatT& weight_data, const MatT& bias_data, std::int32_t activation_type, const MatT& activation_params, const OptionT& opt)
{
    const std::int32_t outw = bottom_blob.w;
    const std::int32_t outh = bottom_blob.h;
    const std::int32_t inch = bottom_blob.c;
    const std::int32_t outch = top_blob.c;
    const std::int32_t size = outw * outh;

    // Block four output channels so every input channel is streamed once per
    // block instead of once per output channel.
    constexpr std::int32_t kOutputBlock = 4;

    // Split the spatial range too when there are fewer output blocks than
    // threads, otherwise small-output-channel layers only use 2-4 cores.
    const std::int32_t pblocks = (outch + kOutputBlock - 1) / kOutputBlock;
    const std::int32_t m_split = std::max(1, opt.num_threads / pblocks);
    const std::int32_t tasks = pblocks * m_split;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (std::int32_t task = 0; task < tasks; task++)
    {
        const std::int32_t pb = task / m_split;
        const std::int32_t mi = task % m_split;
        const std::int32_t i_begin = size * mi / m_split;
        const std::int32_t i_end = size * (mi + 1) / m_split;
        const std::int32_t p0 = pb * kOutputBlock;
        const std::int32_t np = outch - p0 < kOutputBlock ? outch - p0 : kOutputBlock;

        float* outptr[kOutputBlock];
        const float* wptr[kOutputBlock];
        float bias[kOutputBlock];
        for (std::int32_t t = 0; t < np; t++)
        {
            const std::int32_t p = p0 + t;
            outptr[t] = top_blob.channel(p);
            wptr[t] = static_cast<const float*>(weight_data) + inch * p;
            bias[t] = bias_data.empty() ? 0.f : bias_data[p];
            for (std::int32_t i = i_begin; i < i_end; i++)
            {
                outptr[t][i] = bias[t];
            }
        }

        if (np == kOutputBlock)
        {
#if defined(__AVX2__)
            for (std::int32_t q = 0; q < inch; q++)
            {
                const float* sptr = bottom_blob.channel(q);
                const __m256 wv0 = _mm256_set1_ps(wptr[0][q]);
                const __m256 wv1 = _mm256_set1_ps(wptr[1][q]);
                const __m256 wv2 = _mm256_set1_ps(wptr[2][q]);
                const __m256 wv3 = _mm256_set1_ps(wptr[3][q]);
                std::int32_t i = i_begin;
                for (; i + 8 <= i_end; i += 8)
                {
                    const __m256 v = _mm256_loadu_ps(sptr + i);
                    _mm256_storeu_ps(outptr[0] + i,
                                     _mm256_fmadd_ps(v, wv0, _mm256_loadu_ps(outptr[0] + i)));
                    _mm256_storeu_ps(outptr[1] + i,
                                     _mm256_fmadd_ps(v, wv1, _mm256_loadu_ps(outptr[1] + i)));
                    _mm256_storeu_ps(outptr[2] + i,
                                     _mm256_fmadd_ps(v, wv2, _mm256_loadu_ps(outptr[2] + i)));
                    _mm256_storeu_ps(outptr[3] + i,
                                     _mm256_fmadd_ps(v, wv3, _mm256_loadu_ps(outptr[3] + i)));
                }
                for (; i < i_end; i++)
                {
                    const float v = sptr[i];
                    outptr[0][i] += v * wptr[0][q];
                    outptr[1][i] += v * wptr[1][q];
                    outptr[2][i] += v * wptr[2][q];
                    outptr[3][i] += v * wptr[3][q];
                }
            }
#else
            for (std::int32_t q = 0; q < inch; q++)
            {
                const float* sptr = bottom_blob.channel(q);
                for (std::int32_t i = i_begin; i < i_end; i++)
                {
                    const float v = sptr[i];
                    outptr[0][i] += v * wptr[0][q];
                    outptr[1][i] += v * wptr[1][q];
                    outptr[2][i] += v * wptr[2][q];
                    outptr[3][i] += v * wptr[3][q];
                }
            }
#endif
        }
        else
        {
            for (std::int32_t t = 0; t < np; t++)
            {
                for (std::int32_t q = 0; q < inch; q++)
                {
                    const float w = wptr[t][q];
                    const float* sptr = bottom_blob.channel(q);
                    float* op = outptr[t];
                    for (std::int32_t i = i_begin; i < i_end; i++)
                    {
                        op[i] += sptr[i] * w;
                    }
                }
            }
        }

        for (std::int32_t t = 0; t < np; t++)
        {
            for (std::int32_t i = i_begin; i < i_end; i++)
            {
                outptr[t][i] = fused_activation(outptr[t][i], activation_type, activation_params);
            }
        }
    }

    return 0;
}



MLAYER_CONVOLUTION_TEMPLATE
std::int32_t Convolution<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Int8, Activation, QuantizeInt8, CopyMakeBorder>::convolution_im2col(const MatT& bottom_blob, MatT& top_blob, const MatT& weight_data, const MatT& bias_data, std::int32_t kernel_w, std::int32_t kernel_h, std::int32_t stride_w, std::int32_t stride_h, std::int32_t dilation_w, std::int32_t dilation_h, std::int32_t activation_type, const MatT& activation_params, const OptionT& opt)
{
    const std::int32_t outw = top_blob.w;
    const std::int32_t outh = top_blob.h;
    const std::int32_t inch = bottom_blob.c;
    const std::int32_t outch = top_blob.c;
    const std::int32_t maxk = kernel_w * kernel_h;
    const std::int32_t K = inch * maxk;
    const std::int32_t M = outw * outh;

    MatT col(K, M, 4u, opt.workspace_allocator);
    if (col.empty())
    {
        return -100;
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (std::int32_t m = 0; m < M; m++)
    {
        const std::int32_t oy = m / outw;
        const std::int32_t ox = m % outw;
        float* cp = col.row(m);

        for (std::int32_t q = 0; q < inch; q++)
        {
            const MatT src = bottom_blob.channel(q);
            for (std::int32_t ky = 0; ky < kernel_h; ky++)
            {
                const float* sptr = src.row(oy * stride_h + ky * dilation_h) + ox * stride_w;
                float* dst = cp + q * maxk + ky * kernel_w;
                for (std::int32_t kx = 0; kx < kernel_w; kx++)
                {
                    dst[kx] = sptr[kx * dilation_w];
                }
            }
        }
    }

    // Block eight output channels: each im2col row is loaded once per block
    // instead of once per output channel, which is the main memory bottleneck
    // for 3x3 convolutions.
    constexpr std::int32_t kOutputBlock = 8;

    const std::int32_t pblocks = (outch + kOutputBlock - 1) / kOutputBlock;
    const std::int32_t m_split = std::max(1, opt.num_threads / pblocks);
    const std::int32_t tasks = pblocks * m_split;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (std::int32_t task = 0; task < tasks; task++)
    {
        const std::int32_t pb = task / m_split;
        const std::int32_t mi = task % m_split;
        const std::int32_t m_begin = M * mi / m_split;
        const std::int32_t m_end = M * (mi + 1) / m_split;
        const std::int32_t p0 = pb * kOutputBlock;
        const std::int32_t np = outch - p0 < kOutputBlock ? outch - p0 : kOutputBlock;

        float* outptr[kOutputBlock];
        const float* wptr[kOutputBlock];
        float bias[kOutputBlock];
        for (std::int32_t t = 0; t < np; t++)
        {
            const std::int32_t p = p0 + t;
            outptr[t] = top_blob.channel(p);
            wptr[t] = static_cast<const float*>(weight_data) + p * K;
            bias[t] = bias_data.empty() ? 0.f : bias_data[p];
            for (std::int32_t i = m_begin; i < m_end; i++)
            {
                outptr[t][i] = bias[t];
            }
        }

        if (np == kOutputBlock)
        {
#if defined(__AVX2__)
            for (std::int32_t m = m_begin; m < m_end; m++)
            {
                const float* cp = col.row(m);

                __m256 a0 = _mm256_setzero_ps();
                __m256 a1 = _mm256_setzero_ps();
                __m256 a2 = _mm256_setzero_ps();
                __m256 a3 = _mm256_setzero_ps();
                __m256 a4 = _mm256_setzero_ps();
                __m256 a5 = _mm256_setzero_ps();
                __m256 a6 = _mm256_setzero_ps();
                __m256 a7 = _mm256_setzero_ps();
                std::int32_t k = 0;
                for (; k + 8 <= K; k += 8)
                {
                    const __m256 c = _mm256_loadu_ps(cp + k);
                    a0 = _mm256_fmadd_ps(c, _mm256_loadu_ps(wptr[0] + k), a0);
                    a1 = _mm256_fmadd_ps(c, _mm256_loadu_ps(wptr[1] + k), a1);
                    a2 = _mm256_fmadd_ps(c, _mm256_loadu_ps(wptr[2] + k), a2);
                    a3 = _mm256_fmadd_ps(c, _mm256_loadu_ps(wptr[3] + k), a3);
                    a4 = _mm256_fmadd_ps(c, _mm256_loadu_ps(wptr[4] + k), a4);
                    a5 = _mm256_fmadd_ps(c, _mm256_loadu_ps(wptr[5] + k), a5);
                    a6 = _mm256_fmadd_ps(c, _mm256_loadu_ps(wptr[6] + k), a6);
                    a7 = _mm256_fmadd_ps(c, _mm256_loadu_ps(wptr[7] + k), a7);
                }

                float sum0 = bias[0] + _mm_cvtss_f32(hsum8(a0));
                float sum1 = bias[1] + _mm_cvtss_f32(hsum8(a1));
                float sum2 = bias[2] + _mm_cvtss_f32(hsum8(a2));
                float sum3 = bias[3] + _mm_cvtss_f32(hsum8(a3));
                float sum4 = bias[4] + _mm_cvtss_f32(hsum8(a4));
                float sum5 = bias[5] + _mm_cvtss_f32(hsum8(a5));
                float sum6 = bias[6] + _mm_cvtss_f32(hsum8(a6));
                float sum7 = bias[7] + _mm_cvtss_f32(hsum8(a7));
                for (; k < K; k++)
                {
                    const float c = cp[k];
                    sum0 += c * wptr[0][k];
                    sum1 += c * wptr[1][k];
                    sum2 += c * wptr[2][k];
                    sum3 += c * wptr[3][k];
                    sum4 += c * wptr[4][k];
                    sum5 += c * wptr[5][k];
                    sum6 += c * wptr[6][k];
                    sum7 += c * wptr[7][k];
                }

                outptr[0][m] = fused_activation(sum0, activation_type, activation_params);
                outptr[1][m] = fused_activation(sum1, activation_type, activation_params);
                outptr[2][m] = fused_activation(sum2, activation_type, activation_params);
                outptr[3][m] = fused_activation(sum3, activation_type, activation_params);
                outptr[4][m] = fused_activation(sum4, activation_type, activation_params);
                outptr[5][m] = fused_activation(sum5, activation_type, activation_params);
                outptr[6][m] = fused_activation(sum6, activation_type, activation_params);
                outptr[7][m] = fused_activation(sum7, activation_type, activation_params);
            }
#else
            for (std::int32_t m = 0; m < M; m++)
            {
                const float* cp = col.row(m);
                float sum0 = bias[0];
                float sum1 = bias[1];
                float sum2 = bias[2];
                float sum3 = bias[3];
                float sum4 = bias[4];
                float sum5 = bias[5];
                float sum6 = bias[6];
                float sum7 = bias[7];
                for (std::int32_t k = 0; k < K; k++)
                {
                    const float c = cp[k];
                    sum0 += c * wptr[0][k];
                    sum1 += c * wptr[1][k];
                    sum2 += c * wptr[2][k];
                    sum3 += c * wptr[3][k];
                    sum4 += c * wptr[4][k];
                    sum5 += c * wptr[5][k];
                    sum6 += c * wptr[6][k];
                    sum7 += c * wptr[7][k];
                }
                outptr[0][m] = fused_activation(sum0, activation_type, activation_params);
                outptr[1][m] = fused_activation(sum1, activation_type, activation_params);
                outptr[2][m] = fused_activation(sum2, activation_type, activation_params);
                outptr[3][m] = fused_activation(sum3, activation_type, activation_params);
                outptr[4][m] = fused_activation(sum4, activation_type, activation_params);
                outptr[5][m] = fused_activation(sum5, activation_type, activation_params);
                outptr[6][m] = fused_activation(sum6, activation_type, activation_params);
                outptr[7][m] = fused_activation(sum7, activation_type, activation_params);
            }
#endif
        }
        else
        {
            for (std::int32_t t = 0; t < np; t++)
            {
                const float* wp = wptr[t];
                float* op = outptr[t];
                for (std::int32_t m = m_begin; m < m_end; m++)
                {
                    const float* cp = col.row(m);
                    float sum = bias[t];
                    for (std::int32_t k = 0; k < K; k++)
                    {
                        sum += cp[k] * wp[k];
                    }
                    op[m] = fused_activation(sum, activation_type, activation_params);
                }
            }
        }
    }

    return 0;
}

MLAYER_CONVOLUTION_TEMPLATE
std::int32_t Convolution<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Int8, Activation, QuantizeInt8, CopyMakeBorder>::convolution_gemm_pack(const MatT& bottom_blob, MatT& top_blob, const MatT& weight_data, const MatT& bias_data, std::int32_t kernel_w, std::int32_t kernel_h, std::int32_t stride_w, std::int32_t stride_h, std::int32_t dilation_w, std::int32_t dilation_h, std::int32_t activation_type, const MatT& activation_params, const OptionT& opt)
{
#if !defined(__AVX2__)
    return convolution_im2col(bottom_blob, top_blob, weight_data, bias_data, kernel_w, kernel_h, stride_w, stride_h, dilation_w, dilation_h, activation_type, activation_params, opt);
#else
    const std::int32_t outw = top_blob.w;
    const std::int32_t outh = top_blob.h;
    const std::int32_t bch = bottom_blob.c;
    const std::int32_t outch = top_blob.c;
    const std::int32_t bw = bottom_blob.w;
    const std::int32_t maxk = kernel_w * kernel_h;
    const std::int32_t K = bch * maxk;
    const std::int32_t N = outw * outh;

    // Each tile holds 32 output pixels packed as four 8-pixel subtiles:
    //   bt[sub * K * 8 + k * 8 + j] = bottom pixel (sub * 8 + j) at reduction k
    // The tile is built once and reused by every output-channel block, so the
    // im2col data is read from L1/L2 instead of being streamed from DRAM once
    // per output block.
    constexpr std::int32_t kSubtile = 8;
    constexpr std::int32_t kSubtilesPerTile = 4;
    constexpr std::int32_t kTileN = kSubtile * kSubtilesPerTile;
    constexpr std::int32_t kOutBlock = 4;

    const std::int32_t ntiles = (N + kTileN - 1) / kTileN;
    const std::int32_t nout = (outch + kOutBlock - 1) / kOutBlock;
    const std::int32_t threads = std::max(1, opt.num_threads);
    // Split the output channels across threads only when there are fewer
    // pixel tiles than threads (small feature maps).
    const std::int32_t slices = ntiles >= threads ? 1 : std::min(nout, (threads + ntiles - 1) / ntiles);
    const std::int32_t tasks = ntiles * slices;

    const float* wbase = static_cast<const float*>(weight_data);
    const float* bbase = bias_data.empty() ? nullptr : static_cast<const float*>(bias_data);
    const bool plain_store = activation_type == 0;

    #pragma omp parallel num_threads(threads)
    {
        std::vector<float> bt_storage(static_cast<std::size_t>(K) * kTileN);
        float* bt = bt_storage.data();

        #pragma omp for schedule(static)
        for (std::int32_t task = 0; task < tasks; ++task)
        {
            const std::int32_t tile = task / slices;
            const std::int32_t slice = task % slices;
            const std::int32_t m0 = tile * kTileN;
            const std::int32_t mcount = std::min(kTileN, N - m0);
            const std::int32_t p_begin = nout * slice / slices;
            const std::int32_t p_end = nout * (slice + 1) / slices;

            // im2col gather, packed by output pixel.
            for (std::int32_t sub = 0; sub < kSubtilesPerTile; ++sub)
            {
                const std::int32_t jbase = m0 + sub * kSubtile;
                const std::int32_t jcount = std::min(kSubtile, N - jbase);
                if (jcount <= 0)
                {
                    break;
                }
                float* bsub = bt + static_cast<std::size_t>(sub) * K * kSubtile;

                std::int32_t offs[kSubtile];
                for (std::int32_t j = 0; j < kSubtile; ++j)
                {
                    const std::int32_t m = jbase + std::min(j, jcount - 1);
                    const std::int32_t oy = m / outw;
                    const std::int32_t ox = m % outw;
                    offs[j] = (oy * stride_h) * bw + ox * stride_w;
                }

                // 8 aligned output pixels from one row: one vector load/store.
                const bool vector_ok = jcount == kSubtile && stride_w == 1 && ((jbase % outw) + kSubtile <= outw);

                for (std::int32_t q = 0; q < bch; ++q)
                {
                    const float* ch = static_cast<const float*>(bottom_blob.channel(q));
                    for (std::int32_t ky = 0; ky < kernel_h; ++ky)
                    {
                        const std::int32_t row_off = ky * dilation_h * bw;
                        for (std::int32_t kx = 0; kx < kernel_w; ++kx)
                        {
                            const std::int32_t k = q * maxk + ky * kernel_w + kx;
                            float* dst = bsub + static_cast<std::size_t>(k) * kSubtile;
                            const std::int32_t col_off = row_off + kx * dilation_w;
                            if (vector_ok)
                            {
                                _mm256_storeu_ps(dst, _mm256_loadu_ps(ch + offs[0] + col_off));
                            }
                            else
                            {
                                for (std::int32_t j = 0; j < kSubtile; ++j)
                                {
                                    dst[j] = j < jcount ? ch[offs[j] + col_off] : 0.f;
                                }
                            }
                        }
                    }
                }
            }

            // 4 output channels x 8 pixels per micro tile.
            for (std::int32_t pb = p_begin; pb < p_end; ++pb)
            {
                const std::int32_t p0 = pb * kOutBlock;
                const std::int32_t np = std::min(kOutBlock, outch - p0);

                if (np != kOutBlock)
                {
                    // Rare tail block: one output channel at a time.
                    for (std::int32_t t = 0; t < np; ++t)
                    {
                        const std::int32_t p = p0 + t;
                        const float* wp = wbase + static_cast<std::size_t>(p) * K;
                        const float b = bbase == nullptr ? 0.f : bbase[p];
                        float* op = top_blob.channel(p);
                        for (std::int32_t m = m0; m < m0 + mcount; ++m)
                        {
                            const std::int32_t sub = (m - m0) / kSubtile;
                            const std::int32_t j = (m - m0) % kSubtile;
                            const float* bsub = bt + static_cast<std::size_t>(sub) * K * kSubtile;
                            float sum = b;
                            for (std::int32_t k = 0; k < K; ++k)
                            {
                                sum += bsub[k * kSubtile + j] * wp[k];
                            }
                            op[m] = fused_activation(sum, activation_type, activation_params);
                        }
                    }
                    continue;
                }

                const float* w0 = wbase + static_cast<std::size_t>(p0 + 0) * K;
                const float* w1 = wbase + static_cast<std::size_t>(p0 + 1) * K;
                const float* w2 = wbase + static_cast<std::size_t>(p0 + 2) * K;
                const float* w3 = wbase + static_cast<std::size_t>(p0 + 3) * K;
                const float b0 = bbase == nullptr ? 0.f : bbase[p0 + 0];
                const float b1 = bbase == nullptr ? 0.f : bbase[p0 + 1];
                const float b2 = bbase == nullptr ? 0.f : bbase[p0 + 2];
                const float b3 = bbase == nullptr ? 0.f : bbase[p0 + 3];
                float* o0 = top_blob.channel(p0 + 0);
                float* o1 = top_blob.channel(p0 + 1);
                float* o2 = top_blob.channel(p0 + 2);
                float* o3 = top_blob.channel(p0 + 3);

                // Pair two 8-pixel subtiles per reduction step so each
                // broadcast weight feeds 16 multiply-adds.
                for (std::int32_t sub = 0; sub < kSubtilesPerTile; sub += 2)
                {
                    const std::int32_t jbase0 = m0 + sub * kSubtile;
                    const std::int32_t jcount0 = std::min(kSubtile, N - jbase0);
                    if (jcount0 <= 0)
                    {
                        break;
                    }
                    const std::int32_t jbase1 = jbase0 + kSubtile;
                    const std::int32_t jcount1 = std::min(kSubtile, N - jbase1);
                    const float* bsub0 = bt + static_cast<std::size_t>(sub) * K * kSubtile;
                    const float* bsub1 = bsub0 + static_cast<std::size_t>(K) * kSubtile;

                    __m256 a00 = _mm256_setzero_ps();
                    __m256 a01 = _mm256_setzero_ps();
                    __m256 a02 = _mm256_setzero_ps();
                    __m256 a03 = _mm256_setzero_ps();
                    __m256 a10 = _mm256_setzero_ps();
                    __m256 a11 = _mm256_setzero_ps();
                    __m256 a12 = _mm256_setzero_ps();
                    __m256 a13 = _mm256_setzero_ps();
                    for (std::int32_t k = 0; k < K; ++k)
                    {
                        const __m256 v0 = _mm256_loadu_ps(bsub0 + k * kSubtile);
                        const __m256 v1 = jcount1 > 0 ? _mm256_loadu_ps(bsub1 + k * kSubtile) : _mm256_setzero_ps();
                        const __m256 wv0 = _mm256_broadcast_ss(w0 + k);
                        const __m256 wv1 = _mm256_broadcast_ss(w1 + k);
                        const __m256 wv2 = _mm256_broadcast_ss(w2 + k);
                        const __m256 wv3 = _mm256_broadcast_ss(w3 + k);
                        a00 = _mm256_fmadd_ps(v0, wv0, a00);
                        a01 = _mm256_fmadd_ps(v0, wv1, a01);
                        a02 = _mm256_fmadd_ps(v0, wv2, a02);
                        a03 = _mm256_fmadd_ps(v0, wv3, a03);
                        a10 = _mm256_fmadd_ps(v1, wv0, a10);
                        a11 = _mm256_fmadd_ps(v1, wv1, a11);
                        a12 = _mm256_fmadd_ps(v1, wv2, a12);
                        a13 = _mm256_fmadd_ps(v1, wv3, a13);
                    }

                    if (jcount0 == kSubtile && jcount1 == kSubtile && plain_store)
                    {
                        const __m256 vb0 = _mm256_set1_ps(b0);
                        const __m256 vb1 = _mm256_set1_ps(b1);
                        const __m256 vb2 = _mm256_set1_ps(b2);
                        const __m256 vb3 = _mm256_set1_ps(b3);
                        _mm256_storeu_ps(o0 + jbase0, _mm256_add_ps(a00, vb0));
                        _mm256_storeu_ps(o1 + jbase0, _mm256_add_ps(a01, vb1));
                        _mm256_storeu_ps(o2 + jbase0, _mm256_add_ps(a02, vb2));
                        _mm256_storeu_ps(o3 + jbase0, _mm256_add_ps(a03, vb3));
                        _mm256_storeu_ps(o0 + jbase1, _mm256_add_ps(a10, vb0));
                        _mm256_storeu_ps(o1 + jbase1, _mm256_add_ps(a11, vb1));
                        _mm256_storeu_ps(o2 + jbase1, _mm256_add_ps(a12, vb2));
                        _mm256_storeu_ps(o3 + jbase1, _mm256_add_ps(a13, vb3));
                    }
                    else
                    {
                        alignas(32) float r0[kSubtile];
                        alignas(32) float r1[kSubtile];
                        alignas(32) float r2[kSubtile];
                        alignas(32) float r3[kSubtile];
                        _mm256_store_ps(r0, a00);
                        _mm256_store_ps(r1, a01);
                        _mm256_store_ps(r2, a02);
                        _mm256_store_ps(r3, a03);
                        for (std::int32_t j = 0; j < jcount0; ++j)
                        {
                            o0[jbase0 + j] = fused_activation(r0[j] + b0, activation_type, activation_params);
                            o1[jbase0 + j] = fused_activation(r1[j] + b1, activation_type, activation_params);
                            o2[jbase0 + j] = fused_activation(r2[j] + b2, activation_type, activation_params);
                            o3[jbase0 + j] = fused_activation(r3[j] + b3, activation_type, activation_params);
                        }
                        if (jcount1 > 0)
                        {
                            _mm256_store_ps(r0, a10);
                            _mm256_store_ps(r1, a11);
                            _mm256_store_ps(r2, a12);
                            _mm256_store_ps(r3, a13);
                            for (std::int32_t j = 0; j < jcount1; ++j)
                            {
                                o0[jbase1 + j] = fused_activation(r0[j] + b0, activation_type, activation_params);
                                o1[jbase1 + j] = fused_activation(r1[j] + b1, activation_type, activation_params);
                                o2[jbase1 + j] = fused_activation(r2[j] + b2, activation_type, activation_params);
                                o3[jbase1 + j] = fused_activation(r3[j] + b3, activation_type, activation_params);
                            }
                        }
                    }
                }
            }
        }
    }

    return 0;
#endif
}

MLAYER_CONVOLUTION_TEMPLATE
std::int32_t Convolution<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Int8, Activation, QuantizeInt8, CopyMakeBorder>::forward(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const
{
    if constexpr (Int8)
    {
        if (opt.use_int8_inference && weight_data.elemsize == (std::size_t)1u)
        {
            return forward_int8(bottom_blob, top_blob, opt);
        }
    }

    // ncnn compatibility: a 1D blob fed into a 1x1 convolution is treated as
    // a 1x1xC feature map (the old InnerProduct path was equivalent).
    MatT bottom_blob_reshaped;
    const MatT* bottom_ptr = &bottom_blob;
    if (bottom_blob.dims == 1 && kernel_w == 1 && kernel_h == 1)
    {
        bottom_blob_reshaped = bottom_blob.reshape(1, 1, bottom_blob.w, opt.workspace_allocator);
        if (bottom_blob_reshaped.empty())
            return -100;
        bottom_ptr = &bottom_blob_reshaped;
    }

    MatT bottom_blob_bordered;
    make_padding(*bottom_ptr, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    const std::int32_t w = bottom_blob_bordered.w;
    const std::int32_t h = bottom_blob_bordered.h;
    const std::size_t elemsize = bottom_blob_bordered.elemsize;

    const std::int32_t kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const std::int32_t kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    const std::int32_t outw = (w - kernel_extent_w) / stride_w + 1;
    const std::int32_t outh = (h - kernel_extent_h) / stride_h + 1;

    top_blob.create(outw, outh, num_output, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (elemsize == 4u && dilation_w == 1 && dilation_h == 1 &&
        ((kernel_w == 1 && kernel_h == 1 && stride_w == 1 && stride_h == 1) ||
         (kernel_w == 3 && kernel_h == 3 && stride_w == stride_h && (stride_w == 1 || stride_w == 2))))
    {
        return convolution_gemm_pack(bottom_blob_bordered, top_blob, weight_data, bias_data, kernel_w, kernel_h, stride_w, stride_h, dilation_w, dilation_h, activation_type, activation_params, opt);
    }

    if (kernel_w == 1 && kernel_h == 1 && stride_w == 1 && stride_h == 1 && dilation_w == 1 && dilation_h == 1)
    {
        return convolution_1x1(bottom_blob_bordered, top_blob, weight_data, bias_data, activation_type, activation_params, opt);
    }

    const std::size_t im2col_bytes = static_cast<std::size_t>(kernel_w) * kernel_h * bottom_blob_bordered.c *
                                     top_blob.w * top_blob.h * sizeof(float);
    if (kernel_w * kernel_h * bottom_blob_bordered.c >= 16 && im2col_bytes <= 512ull * 1024ull * 1024ull)
    {
        return convolution_im2col(bottom_blob_bordered, top_blob, weight_data, bias_data, kernel_w, kernel_h, stride_w, stride_h, dilation_w, dilation_h, activation_type, activation_params, opt);
    }

    std::int32_t ret = convolution(bottom_blob_bordered, top_blob, weight_data, bias_data, kernel_w, kernel_h, stride_w, stride_h, dilation_w, dilation_h, activation_type, activation_params, opt);
    if (ret != 0)
        return ret;

    return 0;
}

MLAYER_CONVOLUTION_TEMPLATE
std::int32_t Convolution<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Int8, Activation, QuantizeInt8, CopyMakeBorder>::forward(const std::vector<MatT>& bottom_blobs, std::vector<MatT>& top_blobs, const OptionT& opt) const
{
    const MatT& bottom_blob = bottom_blobs[0];
    const MatT& _weight_data = bottom_blobs[1];
    MatT& top_blob = top_blobs[0];

    const std::int32_t _kernel_w = _weight_data.w;
    const std::int32_t _kernel_h = _weight_data.h;
    const std::int32_t _num_output = _weight_data.c;

    MatT weight_data_flattened;
    flatten(_weight_data, weight_data_flattened, opt);
    if (weight_data_flattened.empty())
        return -100;

    MatT bias_data_flattened;
    if (bias_term)
    {
        const MatT& _bias_data = bottom_blobs[2];
        flatten(_bias_data, bias_data_flattened, opt);
        if (bias_data_flattened.empty())
            return -100;
    }

    MatT bottom_blob_bordered;
    make_padding(bottom_blob, bottom_blob_bordered, _kernel_w, _kernel_h, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    const std::int32_t w = bottom_blob_bordered.w;
    const std::int32_t h = bottom_blob_bordered.h;
    const std::size_t elemsize = bottom_blob_bordered.elemsize;

    const std::int32_t kernel_extent_w = dilation_w * (_kernel_w - 1) + 1;
    const std::int32_t kernel_extent_h = dilation_h * (_kernel_h - 1) + 1;

    const std::int32_t outw = (w - kernel_extent_w) / stride_w + 1;
    const std::int32_t outh = (h - kernel_extent_h) / stride_h + 1;

    top_blob.create(outw, outh, _num_output, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    std::int32_t ret = convolution(bottom_blob_bordered, top_blob, weight_data_flattened, bias_data_flattened, _kernel_w, _kernel_h, stride_w, stride_h, dilation_w, dilation_h, activation_type, activation_params, opt);
    if (ret != 0)
        return ret;

    return 0;
}

MLAYER_CONVOLUTION_TEMPLATE
void Convolution<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Int8, Activation, QuantizeInt8, CopyMakeBorder>::make_padding(const MatT& bottom_blob, MatT& bottom_blob_bordered, const OptionT& opt) const
{
    make_padding(bottom_blob, bottom_blob_bordered, kernel_w, kernel_h, opt);
}

MLAYER_CONVOLUTION_TEMPLATE
void Convolution<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Int8, Activation, QuantizeInt8, CopyMakeBorder>::make_padding(const MatT& bottom_blob, MatT& bottom_blob_bordered, std::int32_t _kernel_w, std::int32_t _kernel_h, const OptionT& opt) const
{
    std::int32_t w = bottom_blob.w;
    std::int32_t h = bottom_blob.h;

    const std::int32_t kernel_extent_w = dilation_w * (_kernel_w - 1) + 1;
    const std::int32_t kernel_extent_h = dilation_h * (_kernel_h - 1) + 1;

    bottom_blob_bordered = bottom_blob;
    if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0)
    {
        OptionT opt_b = opt;
        opt_b.blob_allocator = opt.workspace_allocator;
        CopyMakeBorder(bottom_blob, bottom_blob_bordered, pad_top, pad_bottom, pad_left, pad_right, kBorderConstant, pad_value, opt_b);
    }
    else if (pad_left == -233 && pad_right == -233 && pad_top == -233 && pad_bottom == -233)
    {
        // tensorflow padding=SAME or onnx padding=SAME_UPPER
        std::int32_t wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        std::int32_t hpad = kernel_extent_h + (h - 1) / stride_h * stride_h - h;
        if (wpad > 0 || hpad > 0)
        {
            OptionT opt_b = opt;
            opt_b.blob_allocator = opt.workspace_allocator;
            CopyMakeBorder(bottom_blob, bottom_blob_bordered, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, kBorderConstant, pad_value, opt_b);
        }
    }
    else if (pad_left == -234 && pad_right == -234 && pad_top == -234 && pad_bottom == -234)
    {
        // onnx padding=SAME_LOWER
        std::int32_t wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        std::int32_t hpad = kernel_extent_h + (h - 1) / stride_h * stride_h - h;
        if (wpad > 0 || hpad > 0)
        {
            OptionT opt_b = opt;
            opt_b.blob_allocator = opt.workspace_allocator;
            CopyMakeBorder(bottom_blob, bottom_blob_bordered, hpad - hpad / 2, hpad / 2, wpad - wpad / 2, wpad / 2, kBorderConstant, pad_value, opt_b);
        }
    }
}

MLAYER_CONVOLUTION_TEMPLATE
std::int32_t Convolution<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Int8, Activation, QuantizeInt8, CopyMakeBorder>::forward_int8(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const
{
    std::int32_t w = bottom_blob.w;
    std::int32_t h = bottom_blob.h;
    std::int32_t channels = bottom_blob.c;
    std::size_t elemsize = bottom_blob.elemsize;


    const std::int32_t kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const std::int32_t kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    MatT bottom_blob_unbordered = bottom_blob;
    if (elemsize != 1)
    {
        OptionT opt_g = opt;
        opt_g.blob_allocator = opt.workspace_allocator;

        QuantizeInt8(bottom_blob, bottom_blob_unbordered, bottom_blob_int8_scales, opt_g);
        if (bottom_blob_unbordered.empty())
            return -100;
    }

    MatT bottom_blob_bordered;
    make_padding(bottom_blob_unbordered, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;

    std::int32_t outw = (w - kernel_extent_w) / stride_w + 1;
    std::int32_t outh = (h - kernel_extent_h) / stride_h + 1;

    const std::int32_t maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<std::int32_t> _space_ofs(maxk);
    std::int32_t* space_ofs = &_space_ofs[0];
    {
        std::int32_t p1 = 0;
        std::int32_t p2 = 0;
        std::int32_t gap = w * dilation_h - kernel_w * dilation_w;
        for (std::int32_t i = 0; i < kernel_h; i++)
        {
            for (std::int32_t j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }

    // int8
    bool use_int8_requantize = int8_scale_term > 100;
    std::size_t out_elemsize = use_int8_requantize ? 1u : 4u;

    top_blob.create(outw, outh, num_output, out_elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // num_output
    #pragma omp parallel for num_threads(opt.num_threads)
    for (std::int32_t p = 0; p < num_output; p++)
    {
        std::int8_t* outptr = top_blob.channel(p);

        for (std::int32_t i = 0; i < outh; i++)
        {
            for (std::int32_t j = 0; j < outw; j++)
            {
                std::int32_t sum = 0;

                const std::int8_t* kptr = (const std::int8_t*)weight_data + maxk * channels * p;

                // channels
                for (std::int32_t q = 0; q < channels; q++)
                {
                    const MatT m = bottom_blob_bordered.channel(q);
                    const std::int8_t* sptr = m.template row<std::int8_t>(i * stride_h) + j * stride_w;

                    for (std::int32_t k = 0; k < maxk; k++)
                    {
                        std::int32_t val = sptr[space_ofs[k]];
                        std::int32_t wt = kptr[k];
                        sum += val * wt;
                    }

                    kptr += maxk;
                }

                float scale_in;
                if (weight_data_int8_scales[p] == 0)
                    scale_in = 0;
                else
                    scale_in = 1.f / (bottom_blob_int8_scales[0] * weight_data_int8_scales[p]);

                float sumfp32 = sum * scale_in;

                if (bias_term)
                    sumfp32 += bias_data[p];

                sumfp32 = fused_activation(sumfp32, activation_type, activation_params);

                if (use_int8_requantize)
                {
                    // requantize
                    float scale_out = top_blob_int8_scales[0];
                    std::int8_t sums8 = float_to_int8(sumfp32 * scale_out);
                    outptr[0] = sums8;
                    outptr += 1;
                }
                else
                {
                    // dequantize
                    ((float*)outptr)[0] = sumfp32;
                    outptr += 4;
                }
            }
        }
    }

    return 0;
}


#undef MLAYER_CONVOLUTION_TEMPLATE

}  // namespace mlayer

#endif  // MLAYER_CONVOLUTION_HPP
