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

    static std::int32_t convolution_3x3s1_winograd43(const MatT& bottom_blob, MatT& top_blob, const MatT& weight_data, const MatT& bias_data, std::int32_t activation_type, const MatT& activation_params, const OptionT& opt);

    static std::int32_t convolution_3x3s1_winograd(const MatT& bottom_blob, MatT& top_blob, const MatT& weight_data, const MatT& bias_data, std::int32_t activation_type, const MatT& activation_params, const OptionT& opt);

    static std::int32_t convolution_im2col(const MatT& bottom_blob, MatT& top_blob, const MatT& weight_data, const MatT& bias_data, std::int32_t kernel_w, std::int32_t kernel_h, std::int32_t stride_w, std::int32_t stride_h, std::int32_t dilation_w, std::int32_t dilation_h, std::int32_t activation_type, const MatT& activation_params, const OptionT& opt);

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
std::int32_t Convolution<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Int8, Activation, QuantizeInt8, CopyMakeBorder>::convolution_3x3s1_winograd43(const MatT& bottom_blob, MatT& top_blob, const MatT& weight_data, const MatT& bias_data, std::int32_t activation_type, const MatT& activation_params, const OptionT& opt)
{
    // F(4x4, 3x3) winograd for stride-1 fp32 convolution.
    const std::int32_t outw = top_blob.w;
    const std::int32_t outh = top_blob.h;
    const std::int32_t inch = bottom_blob.c;
    const std::int32_t outch = top_blob.c;
    const std::int32_t K = inch * 9;
    const float* weight_ptr = static_cast<const float*>(weight_data);

    static constexpr float kBT[6][6] = {
        {4.f, 0.f, -5.f, 0.f, 1.f, 0.f},
        {0.f, -4.f, -4.f, 1.f, 1.f, 0.f},
        {0.f, 4.f, -4.f, -1.f, 1.f, 0.f},
        {0.f, -2.f, -1.f, 2.f, 1.f, 0.f},
        {0.f, 2.f, -1.f, -2.f, 1.f, 0.f},
        {0.f, 4.f, 0.f, -5.f, 0.f, 1.f},
    };
    static constexpr float kG[6][3] = {
        {1.f / 4.f, 0.f, 0.f},
        {-1.f / 6.f, -1.f / 6.f, -1.f / 6.f},
        {-1.f / 6.f, 1.f / 6.f, -1.f / 6.f},
        {1.f / 24.f, 1.f / 12.f, 1.f / 6.f},
        {1.f / 24.f, -1.f / 12.f, 1.f / 6.f},
        {0.f, 0.f, 1.f},
    };
    static constexpr float kAT[4][6] = {
        {1.f, 1.f, 1.f, 1.f, 1.f, 0.f},
        {0.f, 1.f, -1.f, 2.f, -2.f, 0.f},
        {0.f, 1.f, 1.f, 4.f, 4.f, 0.f},
        {0.f, 1.f, -1.f, 8.f, -8.f, 1.f},
    };

    const std::int32_t tiles_x = (outw + 3) / 4;
    const std::int32_t tiles_y = (outh + 3) / 4;
    const std::int32_t num_tiles = tiles_x * tiles_y;
    const std::int32_t pad_w = tiles_x * 4 + 2;
    const std::int32_t pad_h = tiles_y * 4 + 2;
    constexpr std::int32_t kChannelBlock = 8;
    constexpr std::int32_t kTilePositions = 36;

    MatT input_pad(pad_w, pad_h, inch, 4u, opt.workspace_allocator);
    if (input_pad.empty())
    {
        return -100;
    }
    const std::int32_t copy_w = std::min(pad_w, bottom_blob.w);
    const std::int32_t copy_h = std::min(pad_h, bottom_blob.h);
    #pragma omp parallel for num_threads(opt.num_threads)
    for (std::int32_t q = 0; q < inch; q++)
    {
        for (std::int32_t y = 0; y < pad_h; y++)
        {
            float* dst = input_pad.channel(q).row(y);
            for (std::int32_t x = 0; x < pad_w; x++)
            {
                dst[x] = (y < copy_h && x < copy_w) ? bottom_blob.channel(q).row(y)[x] : 0.f;
            }
        }
    }

    const std::int32_t pblocks = (outch + kChannelBlock - 1) / kChannelBlock;
    MatT U(static_cast<std::int32_t>(static_cast<std::size_t>(pblocks) * kTilePositions * inch * kChannelBlock),
           4u, opt.workspace_allocator);
    if (U.empty())
    {
        return -100;
    }
    float* up = U;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (std::int32_t p = 0; p < outch; p++)
    {
        float* uph = up + static_cast<std::size_t>(p / kChannelBlock) * kTilePositions * inch * kChannelBlock +
                     (p % kChannelBlock);
        for (std::int32_t q = 0; q < inch; q++)
        {
            const float* g = weight_ptr + p * K + q * 9;
            const float gv[3][3] = {{g[0], g[1], g[2]}, {g[3], g[4], g[5]}, {g[6], g[7], g[8]}};

            float t[6][3];
            for (std::int32_t j = 0; j < 3; j++)
            {
                const float a = gv[0][j], b = gv[1][j], c = gv[2][j];
                t[0][j] = 0.25f * a;
                t[1][j] = -(a + b + c) / 6.f;
                t[2][j] = (-a + b - c) / 6.f;
                t[3][j] = a / 24.f + b / 12.f + c / 6.f;
                t[4][j] = a / 24.f - b / 12.f + c / 6.f;
                t[5][j] = c;
            }
            for (std::int32_t i = 0; i < 6; i++)
            {
                const float t0 = t[i][0], t1 = t[i][1], t2 = t[i][2];
                const float u0 = 0.25f * t0;
                const float u1 = -(t0 + t1 + t2) / 6.f;
                const float u2 = (-t0 + t1 - t2) / 6.f;
                const float u3 = t0 / 24.f + t1 / 12.f + t2 / 6.f;
                const float u4 = t0 / 24.f - t1 / 12.f + t2 / 6.f;
                const float u5 = t2;
                float* row = uph + (static_cast<std::size_t>(i * 6) * inch + q) * kChannelBlock;
                row[0 * inch * kChannelBlock] = u0;
                row[1 * inch * kChannelBlock] = u1;
                row[2 * inch * kChannelBlock] = u2;
                row[3 * inch * kChannelBlock] = u3;
                row[4 * inch * kChannelBlock] = u4;
                row[5 * inch * kChannelBlock] = u5;
            }
        }
    }

    const std::int32_t chunk_tiles = std::max(1, 256 / inch);
    const std::int32_t num_chunks = (num_tiles + chunk_tiles - 1) / chunk_tiles;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (std::int32_t chunk = 0; chunk < num_chunks; chunk++)
    {
        const std::int32_t tile_begin = chunk * chunk_tiles;
        const std::int32_t tile_end = std::min(num_tiles, tile_begin + chunk_tiles);
        const std::int32_t chunk_count = tile_end - tile_begin;

        std::vector<float> vbuf(static_cast<std::size_t>(chunk_count) * kTilePositions * inch);

        for (std::int32_t lt = 0; lt < chunk_count; lt++)
        {
            const std::int32_t tile = tile_begin + lt;
            const std::int32_t ty = tile / tiles_x;
            const std::int32_t tx = tile % tiles_x;
            float* vtile = vbuf.data() + static_cast<std::size_t>(lt) * kTilePositions * inch;

            for (std::int32_t q = 0; q < inch; q++)
            {
                const MatT src = input_pad.channel(q);
                const float* d0 = src.row(ty * 4 + 0) + tx * 4;
                const float* d1 = src.row(ty * 4 + 1) + tx * 4;
                const float* d2 = src.row(ty * 4 + 2) + tx * 4;
                const float* d3 = src.row(ty * 4 + 3) + tx * 4;
                const float* d4 = src.row(ty * 4 + 4) + tx * 4;
                const float* d5 = src.row(ty * 4 + 5) + tx * 4;
                const float* d[6] = {d0, d1, d2, d3, d4, d5};

                float r[6][6];
                for (std::int32_t j = 0; j < 6; j++)
                {
                    const float d0 = d[0][j], d1 = d[1][j], d2 = d[2][j];
                    const float d3 = d[3][j], d4 = d[4][j], d5 = d[5][j];
                    r[0][j] = 4.f * d0 - 5.f * d2 + d4;
                    r[1][j] = -4.f * d1 - 4.f * d2 + d3 + d4;
                    r[2][j] = 4.f * d1 - 4.f * d2 - d3 + d4;
                    r[3][j] = -2.f * d1 - d2 + 2.f * d3 + d4;
                    r[4][j] = 2.f * d1 - d2 - 2.f * d3 + d4;
                    r[5][j] = 4.f * d1 - 5.f * d3 + d5;
                }
                for (std::int32_t i = 0; i < 6; i++)
                {
                    const float r0 = r[i][0], r1 = r[i][1], r2 = r[i][2];
                    const float r3 = r[i][3], r4 = r[i][4], r5 = r[i][5];
                    vtile[(static_cast<std::size_t>((i * 6 + 0) * inch)) + q] = 4.f * r0 - 5.f * r2 + r4;
                    vtile[(static_cast<std::size_t>((i * 6 + 1) * inch)) + q] =
                        -4.f * r1 - 4.f * r2 + r3 + r4;
                    vtile[(static_cast<std::size_t>((i * 6 + 2) * inch)) + q] =
                        4.f * r1 - 4.f * r2 - r3 + r4;
                    vtile[(static_cast<std::size_t>((i * 6 + 3) * inch)) + q] =
                        -2.f * r1 - r2 + 2.f * r3 + r4;
                    vtile[(static_cast<std::size_t>((i * 6 + 4) * inch)) + q] =
                        2.f * r1 - r2 - 2.f * r3 + r4;
                    vtile[(static_cast<std::size_t>((i * 6 + 5) * inch)) + q] = 4.f * r1 - 5.f * r3 + r5;
                }
            }
        }

        float mbuf[kTilePositions * kChannelBlock];
        for (std::int32_t pb = 0; pb < pblocks; pb++)
        {
            const std::int32_t p0 = pb * kChannelBlock;
            const std::int32_t np = outch - p0 < kChannelBlock ? outch - p0 : kChannelBlock;
            const float* upb = up + static_cast<std::size_t>(pb) * kTilePositions * inch * kChannelBlock;

            for (std::int32_t lt = 0; lt < chunk_count; lt++)
            {
                const float* vtile = vbuf.data() + static_cast<std::size_t>(lt) * kTilePositions * inch;

                for (std::int32_t pos = 0; pos < kTilePositions; pos++)
                {
                    const float* vrow = vtile + static_cast<std::size_t>(pos) * inch;
                    const float* urow = upb + static_cast<std::size_t>(pos) * inch * kChannelBlock;
                    float* mrow = mbuf + static_cast<std::size_t>(pos) * kChannelBlock;
#if defined(__AVX2__)
                    if (np == kChannelBlock)
                    {
                        for (std::int32_t g = 0; g < kChannelBlock; g += 8)
                        {
                            __m256 acc = _mm256_setzero_ps();
                            for (std::int32_t q = 0; q < inch; q++)
                            {
                                const __m256 v = _mm256_set1_ps(vrow[q]);
                                acc = _mm256_fmadd_ps(v, _mm256_loadu_ps(urow + q * kChannelBlock + g), acc);
                            }
                            _mm256_storeu_ps(mrow + g, acc);
                        }
                        continue;
                    }
#endif
                    for (std::int32_t p = 0; p < np; p++)
                    {
                        float sum = 0.f;
                        for (std::int32_t q = 0; q < inch; q++)
                        {
                            sum += vrow[q] * urow[q * kChannelBlock + p];
                        }
                        mrow[p] = sum;
                    }
                }

                const std::int32_t tile = tile_begin + lt;
                const std::int32_t ty = tile / tiles_x;
                const std::int32_t tx = tile % tiles_x;

                float yv[4][4][kChannelBlock];
#if defined(__AVX2__)
                if (np == kChannelBlock)
                {
                    for (std::int32_t g = 0; g < kChannelBlock; g += 8)
                    {
                        __m256 tmp[4][6];
                        for (std::int32_t r = 0; r < 4; r++)
                        {
                            for (std::int32_t j = 0; j < 6; j++)
                            {
                                __m256 acc = _mm256_set1_ps(0.f);
                                for (std::int32_t i = 0; i < 6; i++)
                                {
                                    acc = _mm256_fmadd_ps(_mm256_set1_ps(kAT[r][i]),
                                                          _mm256_loadu_ps(mbuf + (i * 6 + j) * kChannelBlock + g),
                                                          acc);
                                }
                                tmp[r][j] = acc;
                            }
                        }
                        for (std::int32_t r = 0; r < 4; r++)
                        {
                            for (std::int32_t c = 0; c < 4; c++)
                            {
                                __m256 acc = _mm256_set1_ps(0.f);
                                for (std::int32_t j = 0; j < 6; j++)
                                {
                                    acc = _mm256_fmadd_ps(_mm256_set1_ps(kAT[c][j]), tmp[r][j], acc);
                                }
                                _mm256_storeu_ps(yv[r][c] + g, acc);
                            }
                        }
                    }
                }
                else
#endif
                {
                    for (std::int32_t p = 0; p < np; p++)
                    {
                        float tmp[4][6];
                        for (std::int32_t r = 0; r < 4; r++)
                        {
                            for (std::int32_t j = 0; j < 6; j++)
                            {
                                float acc = 0.f;
                                for (std::int32_t i = 0; i < 6; i++)
                                {
                                    acc += kAT[r][i] * mbuf[(i * 6 + j) * kChannelBlock + p];
                                }
                                tmp[r][j] = acc;
                            }
                        }
                        for (std::int32_t r = 0; r < 4; r++)
                        {
                            for (std::int32_t c = 0; c < 4; c++)
                            {
                                float acc = 0.f;
                                for (std::int32_t j = 0; j < 6; j++)
                                {
                                    acc += kAT[c][j] * tmp[r][j];
                                }
                                yv[r][c][p] = acc;
                            }
                        }
                    }
                }

                for (std::int32_t r = 0; r < 4; r++)
                {
                    const std::int32_t oy = ty * 4 + r;
                    if (oy >= outh)
                    {
                        continue;
                    }
                    for (std::int32_t c = 0; c < 4; c++)
                    {
                        const std::int32_t ox = tx * 4 + c;
                        if (ox >= outw)
                        {
                            continue;
                        }
                        for (std::int32_t t = 0; t < np; t++)
                        {
                            const std::int32_t p = p0 + t;
                            const float bias = bias_data.empty() ? 0.f : bias_data[p];
                            top_blob.channel(p)[oy * outw + ox] =
                                fused_activation(yv[r][c][t] + bias, activation_type, activation_params);
                        }
                    }
                }
            }
        }
    }

    return 0;
}

MLAYER_CONVOLUTION_TEMPLATE
std::int32_t Convolution<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Int8, Activation, QuantizeInt8, CopyMakeBorder>::convolution_3x3s1_winograd(const MatT& bottom_blob, MatT& top_blob, const MatT& weight_data, const MatT& bias_data, std::int32_t activation_type, const MatT& activation_params, const OptionT& opt)
{
    // F(2x2, 3x3) winograd for stride-1 fp32 convolution.
    const std::int32_t outw = top_blob.w;
    const std::int32_t outh = top_blob.h;
    const std::int32_t inch = bottom_blob.c;
    const std::int32_t outch = top_blob.c;
    const std::int32_t K = inch * 9;
    const float* weight_ptr = static_cast<const float*>(weight_data);

    const std::int32_t tiles_x = (outw + 1) / 2;
    const std::int32_t tiles_y = (outh + 1) / 2;
    const std::int32_t num_tiles = tiles_x * tiles_y;
    const std::int32_t pad_w = tiles_x * 2 + 2;
    const std::int32_t pad_h = tiles_y * 2 + 2;
    constexpr std::int32_t kChannelBlock = 8;

    MatT input_pad(pad_w, pad_h, inch, 4u, opt.workspace_allocator);
    if (input_pad.empty())
    {
        return -100;
    }
    const std::int32_t copy_w = std::min(pad_w, bottom_blob.w);
    const std::int32_t copy_h = std::min(pad_h, bottom_blob.h);
    #pragma omp parallel for num_threads(opt.num_threads)
    for (std::int32_t q = 0; q < inch; q++)
    {
        for (std::int32_t y = 0; y < pad_h; y++)
        {
            float* dst = input_pad.channel(q).row(y);
            if (y < copy_h)
            {
                const float* src = bottom_blob.channel(q).row(y);
                for (std::int32_t x = 0; x < copy_w; x++)
                {
                    dst[x] = src[x];
                }
                for (std::int32_t x = copy_w; x < pad_w; x++)
                {
                    dst[x] = 0.f;
                }
            }
            else
            {
                for (std::int32_t x = 0; x < pad_w; x++)
                {
                    dst[x] = 0.f;
                }
            }
        }
    }

    // Transformed weights, blocked as [pblock][pos][q][lane] so one output
    // channel block streams contiguously while the tile loop runs.
    const std::int32_t pblocks = (outch + kChannelBlock - 1) / kChannelBlock;
    MatT U(static_cast<std::int32_t>(static_cast<std::size_t>(pblocks) * 16u * static_cast<std::size_t>(inch) *
                                     static_cast<std::size_t>(kChannelBlock)),
           4u, opt.workspace_allocator);
    if (U.empty())
    {
        return -100;
    }
    float* up = U;
    #pragma omp parallel for num_threads(opt.num_threads)
    for (std::int32_t p = 0; p < outch; p++)
    {
        float* uph = up + static_cast<std::size_t>(p / kChannelBlock) * 16 * inch * kChannelBlock +
                     (p % kChannelBlock);
        for (std::int32_t q = 0; q < inch; q++)
        {
            const float* g = weight_ptr + p * K + q * 9;
            const float g00 = g[0], g01 = g[1], g02 = g[2];
            const float g10 = g[3], g11 = g[4], g12 = g[5];
            const float g20 = g[6], g21 = g[7], g22 = g[8];

            float t[4][3];
            t[0][0] = g00; t[0][1] = g01; t[0][2] = g02;
            t[1][0] = (g00 + g10 + g20) * 0.5f;
            t[1][1] = (g01 + g11 + g21) * 0.5f;
            t[1][2] = (g02 + g12 + g22) * 0.5f;
            t[2][0] = (g00 - g10 + g20) * 0.5f;
            t[2][1] = (g01 - g11 + g21) * 0.5f;
            t[2][2] = (g02 - g12 + g22) * 0.5f;
            t[3][0] = g20; t[3][1] = g21; t[3][2] = g22;

            for (std::int32_t i = 0; i < 4; i++)
            {
                const float u0 = t[i][0];
                const float u1 = (t[i][0] + t[i][1] + t[i][2]) * 0.5f;
                const float u2 = (t[i][0] - t[i][1] + t[i][2]) * 0.5f;
                const float u3 = t[i][2];
                uph[((i * 4 + 0) * inch + q) * kChannelBlock] = u0;
                uph[((i * 4 + 1) * inch + q) * kChannelBlock] = u1;
                uph[((i * 4 + 2) * inch + q) * kChannelBlock] = u2;
                uph[((i * 4 + 3) * inch + q) * kChannelBlock] = u3;
            }
        }
    }

    // Tile-blocked: each V tile is computed once and reused for every output
    // channel block while it is still resident in L1.
    const std::int32_t chunk_tiles = std::max(1, 1024 / inch);
    const std::int32_t num_chunks = (num_tiles + chunk_tiles - 1) / chunk_tiles;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (std::int32_t chunk = 0; chunk < num_chunks; chunk++)
    {
        const std::int32_t tile_begin = chunk * chunk_tiles;
        const std::int32_t tile_end = std::min(num_tiles, tile_begin + chunk_tiles);
        const std::int32_t chunk_count = tile_end - tile_begin;

        std::vector<float> vbuf(static_cast<std::size_t>(chunk_count) * 16 * inch);

        for (std::int32_t lt = 0; lt < chunk_count; lt++)
        {
            const std::int32_t tile = tile_begin + lt;
        const std::int32_t ty = tile / tiles_x;
        const std::int32_t tx = tile % tiles_x;
        float* vtile = vbuf.data() + static_cast<std::size_t>(lt) * 16 * inch;

        for (std::int32_t q = 0; q < inch; q++)
        {
            const MatT src = input_pad.channel(q);
            const float* d0 = src.row(ty * 2 + 0) + tx * 2;
            const float* d1 = src.row(ty * 2 + 1) + tx * 2;
            const float* d2 = src.row(ty * 2 + 2) + tx * 2;
            const float* d3 = src.row(ty * 2 + 3) + tx * 2;

            float t0[4], t1[4], t2[4], t3[4];
            for (std::int32_t c = 0; c < 4; c++)
            {
                t0[c] = d0[c] - d2[c];
                t1[c] = d1[c] + d2[c];
                t2[c] = d2[c] - d1[c];
                t3[c] = d1[c] - d3[c];
            }
            const float* tr[4] = {t0, t1, t2, t3};
            for (std::int32_t i = 0; i < 4; i++)
            {
                const float* r = tr[i];
                vtile[static_cast<std::size_t>((i * 4 + 0) * inch + q)] = r[0] - r[2];
                vtile[static_cast<std::size_t>((i * 4 + 1) * inch + q)] = r[1] + r[2];
                vtile[static_cast<std::size_t>((i * 4 + 2) * inch + q)] = r[2] - r[1];
                vtile[static_cast<std::size_t>((i * 4 + 3) * inch + q)] = r[1] - r[3];
            }
        }
        }

        float mbuf[16 * kChannelBlock];
        for (std::int32_t pb = 0; pb < pblocks; pb++)
        {
            const std::int32_t p0 = pb * kChannelBlock;
            const std::int32_t np = outch - p0 < kChannelBlock ? outch - p0 : kChannelBlock;
            const float* upb = up + static_cast<std::size_t>(pb) * 16 * inch * kChannelBlock;

            for (std::int32_t lt = 0; lt < chunk_count; lt++)
            {
                const std::int32_t tile = tile_begin + lt;
                const float* vtile = vbuf.data() + static_cast<std::size_t>(lt) * 16 * inch;
            for (std::int32_t pos = 0; pos < 16; pos++)
            {
                const float* vrow = vtile + static_cast<std::size_t>(pos) * inch;
                const float* urow = upb + static_cast<std::size_t>(pos) * inch * kChannelBlock;
                float* mrow = mbuf + static_cast<std::size_t>(pos) * kChannelBlock;
#if defined(__AVX2__)
                if (np == kChannelBlock)
                {
                    for (std::int32_t g = 0; g < kChannelBlock; g += 8)
                    {
                        __m256 acc = _mm256_setzero_ps();
                        for (std::int32_t q = 0; q < inch; q++)
                        {
                            const __m256 v = _mm256_set1_ps(vrow[q]);
                            acc = _mm256_fmadd_ps(v, _mm256_loadu_ps(urow + q * kChannelBlock + g), acc);
                        }
                        _mm256_storeu_ps(mrow + g, acc);
                    }
                    continue;
                }
#endif
                for (std::int32_t p = 0; p < np; p++)
                {
                    float sum = 0.f;
                    for (std::int32_t q = 0; q < inch; q++)
                    {
                        sum += vrow[q] * urow[q * kChannelBlock + p];
                    }
                    mrow[p] = sum;
                }
            }

            const std::int32_t ty = tile / tiles_x;
            const std::int32_t tx = tile % tiles_x;
            const std::int32_t oy = ty * 2;
            const std::int32_t ox = tx * 2;

            float yv[4][kChannelBlock];
#if defined(__AVX2__)
            if (np == kChannelBlock)
            {
                for (std::int32_t g = 0; g < kChannelBlock; g += 8)
                {
                    __m256 t0[4];
                    __m256 t1[4];
                    for (std::int32_t j = 0; j < 4; j++)
                    {
                        const __m256 m0 = _mm256_loadu_ps(mbuf + (0 * 4 + j) * kChannelBlock + g);
                        const __m256 m1 = _mm256_loadu_ps(mbuf + (1 * 4 + j) * kChannelBlock + g);
                        const __m256 m2 = _mm256_loadu_ps(mbuf + (2 * 4 + j) * kChannelBlock + g);
                        const __m256 m3 = _mm256_loadu_ps(mbuf + (3 * 4 + j) * kChannelBlock + g);
                        t0[j] = _mm256_add_ps(_mm256_add_ps(m0, m1), m2);
                        t1[j] = _mm256_sub_ps(_mm256_sub_ps(m1, m2), m3);
                    }
                    _mm256_storeu_ps(yv[0] + g, _mm256_add_ps(_mm256_add_ps(t0[0], t0[1]), t0[2]));
                    _mm256_storeu_ps(yv[1] + g, _mm256_sub_ps(_mm256_sub_ps(t0[1], t0[2]), t0[3]));
                    _mm256_storeu_ps(yv[2] + g, _mm256_add_ps(_mm256_add_ps(t1[0], t1[1]), t1[2]));
                    _mm256_storeu_ps(yv[3] + g, _mm256_sub_ps(_mm256_sub_ps(t1[1], t1[2]), t1[3]));
                }
            }
            else
#endif
            {
                for (std::int32_t p = 0; p < np; p++)
                {
                    float t0[4], t1[4];
                    for (std::int32_t j = 0; j < 4; j++)
                    {
                        t0[j] = mbuf[(0 * 4 + j) * kChannelBlock + p] +
                                mbuf[(1 * 4 + j) * kChannelBlock + p] +
                                mbuf[(2 * 4 + j) * kChannelBlock + p];
                        t1[j] = mbuf[(1 * 4 + j) * kChannelBlock + p] -
                                mbuf[(2 * 4 + j) * kChannelBlock + p] -
                                mbuf[(3 * 4 + j) * kChannelBlock + p];
                    }
                    yv[0][p] = t0[0] + t0[1] + t0[2];
                    yv[1][p] = t0[1] - t0[2] - t0[3];
                    yv[2][p] = t1[0] + t1[1] + t1[2];
                    yv[3][p] = t1[1] - t1[2] - t1[3];
                }
            }

            for (std::int32_t t = 0; t < np; t++)
            {
                const std::int32_t p = p0 + t;
                const float bias = bias_data.empty() ? 0.f : bias_data[p];
                float* op = top_blob.channel(p);
                if (oy < outh && ox < outw)
                {
                    op[oy * outw + ox] = fused_activation(yv[0][t] + bias, activation_type, activation_params);
                }
                if (oy < outh && ox + 1 < outw)
                {
                    op[oy * outw + ox + 1] = fused_activation(yv[1][t] + bias, activation_type, activation_params);
                }
                if (oy + 1 < outh && ox < outw)
                {
                    op[(oy + 1) * outw + ox] = fused_activation(yv[2][t] + bias, activation_type, activation_params);
                }
                if (oy + 1 < outh && ox + 1 < outw)
                {
                    op[(oy + 1) * outw + ox + 1] = fused_activation(yv[3][t] + bias, activation_type, activation_params);
                }
            }
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

    if (kernel_w == 1 && kernel_h == 1 && stride_w == 1 && stride_h == 1 && dilation_w == 1 && dilation_h == 1)
    {
        return convolution_1x1(bottom_blob_bordered, top_blob, weight_data, bias_data, activation_type, activation_params, opt);
    }

    if (kernel_w == 3 && kernel_h == 3 && stride_w == 1 && stride_h == 1 && dilation_w == 1 && dilation_h == 1 &&
        elemsize == 4u && num_output >= 8 && bottom_blob_bordered.c >= 4)
    {
        if (outw * outh >= 4096 && bottom_blob_bordered.c >= 32)
        {
            return convolution_3x3s1_winograd43(bottom_blob_bordered, top_blob, weight_data, bias_data, activation_type, activation_params, opt);
        }
        return convolution_3x3s1_winograd(bottom_blob_bordered, top_blob, weight_data, bias_data, activation_type, activation_params, opt);
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
