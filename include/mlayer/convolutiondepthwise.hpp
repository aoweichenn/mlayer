// Copyright 2017 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_CONVOLUTIONDEPTHWISE_HPP
#define MLAYER_CONVOLUTIONDEPTHWISE_HPP

#include <climits>
#include <cmath>

#if defined(__AVX2__)
#include <immintrin.h>
#endif
#if defined(__ARM_NEON)
#include <arm_neon.h>
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
                                float, const OptionT&),
         void (*Flatten)(const MatT&, MatT&, const OptionT&)>
class ConvolutionDepthWise : public LayerT {
public:
    ConvolutionDepthWise();

    std::int32_t load_param(const ParamDictT& pd) override;
    std::int32_t load_model(const ModelBinT& mb) override;
    std::int32_t forward(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const override;
    std::int32_t forward(const std::vector<MatT>& bottom_blobs, std::vector<MatT>& top_blobs, const OptionT& opt) const override;

protected:
    void make_padding(const MatT& bottom_blob, MatT& bottom_blob_bordered, const OptionT& opt) const;
    void make_padding(const MatT& bottom_blob, MatT& bottom_blob_bordered, std::int32_t kernel_w, std::int32_t kernel_h,
                      const OptionT& opt) const;
    std::int32_t forward_int8(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const;

    static std::int32_t convolutiondepthwise(const MatT& bottom_blob, MatT& top_blob, const MatT& weight_data,
                                             const MatT& bias_data, std::int32_t kernel_w, std::int32_t kernel_h,
                                             std::int32_t stride_w, std::int32_t stride_h, std::int32_t dilation_w,
                                             std::int32_t dilation_h, std::int32_t group, std::int32_t activation_type,
                                             const MatT& activation_params, const OptionT& opt);

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
    std::int32_t group;
    std::int32_t int8_scale_term;
    std::int32_t activation_type;
    MatT activation_params;

    MatT weight_data_int8_scales;
    MatT bottom_blob_int8_scales;
    MatT top_blob_int8_scales;

    bool dynamic_weight;
};

#define MLAYER_CONVOLUTIONDEPTHWISE_TEMPLATE \
    template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, typename ModelBinT, bool Validate, \
             bool Int8, float (*Activation)(float, std::int32_t, const MatT&), \
             void (*QuantizeInt8)(const MatT&, MatT&, const MatT&, const OptionT&), \
             void (*CopyMakeBorder)(const MatT&, MatT&, std::int32_t, std::int32_t, std::int32_t, std::int32_t, \
                                    std::int32_t, float, const OptionT&), \
             void (*Flatten)(const MatT&, MatT&, const OptionT&)>

MLAYER_CONVOLUTIONDEPTHWISE_TEMPLATE
ConvolutionDepthWise<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Int8, Activation, QuantizeInt8, CopyMakeBorder, Flatten>::ConvolutionDepthWise()
{
    this->one_blob_only = true;
    this->support_inplace = false;
}

MLAYER_CONVOLUTIONDEPTHWISE_TEMPLATE
std::int32_t ConvolutionDepthWise<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Int8, Activation, QuantizeInt8, CopyMakeBorder, Flatten>::load_param(const ParamDictT& pd)
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
    group = pd.get(7, 1);
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

    if (int8_scale_term != 0 && int8_scale_term != 1 && int8_scale_term != 2 && int8_scale_term != 101 && int8_scale_term != 102)
        return -1;

    if (group <= 0)
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

        if (num_output <= 0 || num_output % group != 0 || weight_data_size <= 0 || weight_data_size % num_output != 0 || (weight_data_size / num_output) % (kernel_w * kernel_h) != 0)
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

MLAYER_CONVOLUTIONDEPTHWISE_TEMPLATE
std::int32_t ConvolutionDepthWise<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Int8, Activation, QuantizeInt8, CopyMakeBorder, Flatten>::load_model(const ModelBinT& mb)
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
        if (int8_scale_term == 1 || int8_scale_term == 101)
        {
            weight_data_int8_scales = mb.load(group, 1);
            bottom_blob_int8_scales = mb.load(1, 1);

            float bottom_blob_int8_scale = bottom_blob_int8_scales[0];
            bottom_blob_int8_scales = MatT(group);
            bottom_blob_int8_scales.fill(bottom_blob_int8_scale);
        }
        else if (int8_scale_term == 2 || int8_scale_term == 102)
        {
            weight_data_int8_scales = mb.load(1, 1);
            bottom_blob_int8_scales = mb.load(1, 1);

            // extend group if only one provided
            float weight_data_int8_scale = weight_data_int8_scales[0];
            weight_data_int8_scales = MatT(group);
            weight_data_int8_scales.fill(weight_data_int8_scale);

            float bottom_blob_int8_scale = bottom_blob_int8_scales[0];
            bottom_blob_int8_scales = MatT(group);
            bottom_blob_int8_scales.fill(bottom_blob_int8_scale);
        }

        if (int8_scale_term > 100)
        {
            top_blob_int8_scales = mb.load(1, 1);

            float top_blob_int8_scale = top_blob_int8_scales[0];
            top_blob_int8_scales = MatT(group);
            top_blob_int8_scales.fill(top_blob_int8_scale);
        }
    }

    if constexpr (Int8)
    {
        // runtime quantize the weight data
        if (weight_data.elemsize == (std::size_t)4u && int8_scale_term)
        {
            MatT int8_weight_data(weight_data_size, (std::size_t)1u);
            if (int8_weight_data.empty())
                return -100;

            const std::int32_t weight_data_size_g = weight_data_size / group;

            for (std::int32_t g = 0; g < group; g++)
            {
                OptionT opt_q;
                opt_q.num_threads = 1;
                opt_q.blob_allocator = static_cast<decltype(opt_q.blob_allocator)>(int8_weight_data.allocator);
                opt_q.use_packing_layout = false;

                const MatT weight_data_g = weight_data.range(weight_data_size_g * g, weight_data_size_g);
                MatT int8_weight_data_g = int8_weight_data.range(weight_data_size_g * g, weight_data_size_g);
                const MatT weight_data_int8_scales_g = weight_data_int8_scales.range(g, 1);
                QuantizeInt8(weight_data_g, int8_weight_data_g, weight_data_int8_scales_g, opt_q);
            }

            weight_data = int8_weight_data;
        }
    }

    return 0;
}

MLAYER_CONVOLUTIONDEPTHWISE_TEMPLATE
std::int32_t ConvolutionDepthWise<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Int8, Activation, QuantizeInt8, CopyMakeBorder, Flatten>::convolutiondepthwise(const MatT& bottom_blob, MatT& top_blob, const MatT& weight_data, const MatT& bias_data, std::int32_t kernel_w, std::int32_t kernel_h, std::int32_t stride_w, std::int32_t stride_h, std::int32_t dilation_w, std::int32_t dilation_h, std::int32_t group, std::int32_t activation_type, const MatT& activation_params, const OptionT& opt)
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

    // depth-wise
    if (inch == group && group == outch)
    {
        // One channel per output and per-tap weights shared by all pixels, so
        // eight output pixels can be produced per vector iteration.
        const bool vectorizable = stride_w == 1 && activation_type == 0 && maxk <= 9;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t g = 0; g < group; g++)
        {
            float* outptr = top_blob.channel(g);
            const float* kptr = (const float*)weight_data + maxk * g;
            const MatT m = bottom_blob.channel(g);

            if (vectorizable)
            {
#if defined(__AVX2__)
                __m256 wv[9];
                for (std::int32_t k = 0; k < maxk; k++)
                {
                    wv[k] = _mm256_set1_ps(kptr[k]);
                }
                const __m256 bv = _mm256_set1_ps(bias_term ? bias_data[g] : 0.f);

                for (std::int32_t i = 0; i < outh; i++)
                {
                    const float* sptr = m.row(i * stride_h);
                    float* optr = outptr + i * outw;
                    std::int32_t j = 0;
                    for (; j + 8 <= outw; j += 8)
                    {
                        __m256 acc = bv;
                        for (std::int32_t k = 0; k < maxk; k++)
                        {
                            acc = _mm256_fmadd_ps(_mm256_loadu_ps(sptr + j + space_ofs[k]), wv[k], acc);
                        }
                        _mm256_storeu_ps(optr + j, acc);
                    }
                    for (; j < outw; j++)
                    {
                        float sum = bias_term ? bias_data[g] : 0.f;
                        for (std::int32_t k = 0; k < maxk; k++)
                        {
                            sum += sptr[j + space_ofs[k]] * kptr[k];
                        }
                        optr[j] = sum;
                    }
                }
                continue;
#elif defined(__ARM_NEON)
                float32x4_t wv[9];
                for (std::int32_t k = 0; k < maxk; k++)
                {
                    wv[k] = vdupq_n_f32(kptr[k]);
                }
                const float32x4_t bv = vdupq_n_f32(bias_term ? bias_data[g] : 0.f);

                for (std::int32_t i = 0; i < outh; i++)
                {
                    const float* sptr = m.row(i * stride_h);
                    float* optr = outptr + i * outw;
                    std::int32_t j = 0;
                    for (; j + 4 <= outw; j += 4)
                    {
                        float32x4_t acc = bv;
                        for (std::int32_t k = 0; k < maxk; k++)
                        {
                            acc = vfmaq_f32(acc, vld1q_f32(sptr + j + space_ofs[k]), wv[k]);
                        }
                        vst1q_f32(optr + j, acc);
                    }
                    for (; j < outw; j++)
                    {
                        float sum = bias_term ? bias_data[g] : 0.f;
                        for (std::int32_t k = 0; k < maxk; k++)
                        {
                            sum += sptr[j + space_ofs[k]] * kptr[k];
                        }
                        optr[j] = sum;
                    }
                }
                continue;
#endif
            }

            for (std::int32_t i = 0; i < outh; i++)
            {
                for (std::int32_t j = 0; j < outw; j++)
                {
                    float sum = 0.f;

                    if (bias_term)
                        sum = bias_data[g];

                    const float* sptr = m.row(i * stride_h) + j * stride_w;

                    for (std::int32_t k = 0; k < maxk; k++)
                    {
                        float val = sptr[space_ofs[k]];
                        float w = kptr[k];
                        sum += val * w;
                    }

                    outptr[j] = Activation(sum, activation_type, activation_params);
                }

                outptr += outw;
            }
        }
    }
    else
    {
        // group convolution
        const std::int32_t inch_g = inch / group;
        const std::int32_t outch_g = outch / group;

#ifdef _WIN32
        #pragma omp parallel for num_threads(opt.num_threads)
#else
        #pragma omp parallel for collapse(2) num_threads(opt.num_threads)
#endif
        for (std::int32_t g = 0; g < group; g++)
        {
            for (std::int32_t p = 0; p < outch_g; p++)
            {
                float* outptr = top_blob.channel(g * outch_g + p);
                const float* weight_data_ptr = (const float*)weight_data + maxk * inch_g * outch_g * g;

                // shadowed variable for less openmp task args
                const std::int32_t outw = top_blob.w;
                const std::int32_t outh = top_blob.h;

                for (std::int32_t i = 0; i < outh; i++)
                {
                    for (std::int32_t j = 0; j < outw; j++)
                    {
                        float sum = 0.f;

                        if (bias_term)
                            sum = bias_data[outch_g * g + p];

                        const float* kptr = weight_data_ptr + maxk * inch_g * p;

                        for (std::int32_t q = 0; q < inch_g; q++)
                        {
                            const MatT m = bottom_blob.channel(inch_g * g + q);
                            const float* sptr = m.row(i * stride_h) + j * stride_w;

                            for (std::int32_t k = 0; k < maxk; k++)
                            {
                                float val = sptr[space_ofs[k]];
                                float w = kptr[k];
                                sum += val * w;
                            }

                            kptr += maxk;
                        }

                        outptr[j] = Activation(sum, activation_type, activation_params);
                    }

                    outptr += outw;
                }
            }
        }
    }

    return 0;
}

MLAYER_CONVOLUTIONDEPTHWISE_TEMPLATE
std::int32_t ConvolutionDepthWise<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Int8, Activation, QuantizeInt8, CopyMakeBorder, Flatten>::forward(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const
{
    // convolv with NxN kernel
    // value = value + bias

    if constexpr (Int8)
    {
        // int8 weights can only be interpreted by the int8 kernel; the
        // use_int8_inference flag is not required for them.
        if (weight_data.elemsize == (std::size_t)1u)
        {
            return forward_int8(bottom_blob, top_blob, opt);
        }
    }


    MatT bottom_blob_bordered;
    make_padding(bottom_blob, bottom_blob_bordered, opt);
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

    std::int32_t ret = convolutiondepthwise(bottom_blob_bordered, top_blob, weight_data, bias_data, kernel_w, kernel_h, stride_w, stride_h, dilation_w, dilation_h, group, activation_type, activation_params, opt);
    if (ret != 0)
        return ret;

    return 0;
}

MLAYER_CONVOLUTIONDEPTHWISE_TEMPLATE
std::int32_t ConvolutionDepthWise<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Int8, Activation, QuantizeInt8, CopyMakeBorder, Flatten>::forward(const std::vector<MatT>& bottom_blobs, std::vector<MatT>& top_blobs, const OptionT& opt) const
{
    const MatT& bottom_blob = bottom_blobs[0];
    const MatT& _weight_data = bottom_blobs[1];
    MatT& top_blob = top_blobs[0];

    const std::int32_t _kernel_w = _weight_data.w;
    const std::int32_t _kernel_h = _weight_data.h;
    const std::int32_t _num_output = _weight_data.c;

    MatT weight_data_flattened;
    Flatten(_weight_data, weight_data_flattened, opt);
    if (weight_data_flattened.empty())
        return -100;

    MatT bias_data_flattened;
    if (bias_term)
    {
        const MatT& _bias_data = bottom_blobs[2];
        Flatten(_bias_data, bias_data_flattened, opt);
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

    std::int32_t ret = convolutiondepthwise(bottom_blob_bordered, top_blob, weight_data_flattened, bias_data_flattened, _kernel_w, _kernel_h, stride_w, stride_h, dilation_w, dilation_h, group, activation_type, activation_params, opt);
    if (ret != 0)
        return ret;

    return 0;
}

MLAYER_CONVOLUTIONDEPTHWISE_TEMPLATE
void ConvolutionDepthWise<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Int8, Activation, QuantizeInt8, CopyMakeBorder, Flatten>::make_padding(const MatT& bottom_blob, MatT& bottom_blob_bordered, const OptionT& opt) const
{
    make_padding(bottom_blob, bottom_blob_bordered, kernel_w, kernel_h, opt);
}

MLAYER_CONVOLUTIONDEPTHWISE_TEMPLATE
void ConvolutionDepthWise<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Int8, Activation, QuantizeInt8, CopyMakeBorder, Flatten>::make_padding(const MatT& bottom_blob, MatT& bottom_blob_bordered, std::int32_t _kernel_w, std::int32_t _kernel_h, const OptionT& opt) const
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

MLAYER_CONVOLUTIONDEPTHWISE_TEMPLATE
std::int32_t ConvolutionDepthWise<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Int8, Activation, QuantizeInt8, CopyMakeBorder, Flatten>::forward_int8(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const
{
    // convolv with NxN kernel
    // value = value + bias

    std::int32_t w = bottom_blob.w;
    std::int32_t h = bottom_blob.h;
    std::int32_t channels = bottom_blob.c;
    std::size_t elemsize = bottom_blob.elemsize;

    if (channels % group != 0 || num_output % group != 0)
    {
        // reject invalid group
        return -100;
    }


    const std::int32_t kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const std::int32_t kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    MatT bottom_blob_int8 = bottom_blob;
    if (elemsize != 1)
    {
        const std::int32_t channels_g = channels / group;

        MatT scales(channels);
        {
            float* ps = scales;
            for (std::int32_t g = 0; g < group; g++)
            {
                float scale = bottom_blob_int8_scales[g];
                for (std::int32_t q = 0; q < channels_g; q++)
                {
                    *ps++ = scale;
                }
            }
        }

        OptionT opt_q = opt;
        opt_q.blob_allocator = opt.workspace_allocator;
        QuantizeInt8(bottom_blob, bottom_blob_int8, scales, opt_q);
        if (bottom_blob_int8.empty())
            return -100;
    }

    MatT bottom_blob_bordered;
    make_padding(bottom_blob_int8, bottom_blob_bordered, opt);
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

    // depth-wise
    if (channels == group && group == num_output)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t g = 0; g < group; g++)
        {
            std::int8_t* outptr = top_blob.channel(g);
            const std::int8_t* kptr = (const std::int8_t*)weight_data + maxk * g;
            const MatT m = bottom_blob_bordered.channel(g);

            for (std::int32_t i = 0; i < outh; i++)
            {
                for (std::int32_t j = 0; j < outw; j++)
                {
                    std::int32_t sum = 0;

                    const std::int8_t* sptr = m.template row<std::int8_t>(i * stride_h) + j * stride_w;

                    for (std::int32_t k = 0; k < maxk; k++)
                    {
                        std::int8_t val = sptr[space_ofs[k]];
                        std::int8_t w = kptr[k];
                        sum += val * w;
                    }

                    float scale_in;
                    if (weight_data_int8_scales[g] == 0)
                        scale_in = 0;
                    else
                        scale_in = 1.f / (bottom_blob_int8_scales[g] * weight_data_int8_scales[g]);

                    float sumfp32 = sum * scale_in;

                    if (bias_term)
                        sumfp32 += bias_data[g];

                    sumfp32 = Activation(sumfp32, activation_type, activation_params);

                    if (use_int8_requantize)
                    {
                        // requantize
                        float scale_out = top_blob_int8_scales[g];
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
    }
    else
    {
        // group convolution
        const std::int32_t channels_g = channels / group;
        const std::int32_t num_output_g = num_output / group;

#ifdef _WIN32
        #pragma omp parallel for num_threads(opt.num_threads)
#else // _WIN32
        #pragma omp parallel for collapse(2) num_threads(opt.num_threads)
#endif // _WIN32
        for (std::int32_t g = 0; g < group; g++)
        {
            for (std::int32_t p = 0; p < num_output_g; p++)
            {
                std::int8_t* outptr = top_blob.channel(g * num_output_g + p);
                const std::int8_t* weight_data_ptr = (const std::int8_t*)weight_data + maxk * channels_g * num_output_g * g;

                for (std::int32_t i = 0; i < outh; i++)
                {
                    for (std::int32_t j = 0; j < outw; j++)
                    {
                        std::int32_t sum = 0;

                        const std::int8_t* kptr = weight_data_ptr + maxk * channels_g * p;

                        // channels_g
                        for (std::int32_t q = 0; q < channels_g; q++)
                        {
                            const MatT m = bottom_blob_bordered.channel(channels_g * g + q);
                            const std::int8_t* sptr = m.template row<std::int8_t>(i * stride_h) + j * stride_w;

                            for (std::int32_t k = 0; k < maxk; k++)
                            {
                                std::int8_t val = sptr[space_ofs[k]];
                                std::int8_t w = kptr[k];
                                sum += val * w;
                            }

                            kptr += maxk;
                        }

                        float scale_in;
                        if (weight_data_int8_scales[g] == 0)
                            scale_in = 0;
                        else
                            scale_in = 1.f / (bottom_blob_int8_scales[g] * weight_data_int8_scales[g]);

                        float sumfp32 = sum * scale_in;

                        if (bias_term)
                            sumfp32 += bias_data[g * num_output_g + p];

                        sumfp32 = Activation(sumfp32, activation_type, activation_params);

                        if (use_int8_requantize)
                        {
                            // requantize
                            float scale_out = top_blob_int8_scales[g];
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
        }
    }

    return 0;
}


#undef MLAYER_CONVOLUTIONDEPTHWISE_TEMPLATE

}  // namespace mlayer

#endif  // MLAYER_CONVOLUTIONDEPTHWISE_HPP
