// Copyright 2017 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_DECONVOLUTION_HPP
#define MLAYER_DECONVOLUTION_HPP

#include <climits>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <mlayer/border.hpp>

namespace mlayer {

template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, typename ModelBinT, bool Validate,
         float (*Activation)(float, std::int32_t, const MatT&),
         void (*CopyCutBorder)(const MatT&, MatT&, std::int32_t, std::int32_t, std::int32_t, std::int32_t, const OptionT&),
         void (*Flatten)(const MatT&, MatT&, const OptionT&)>
class Deconvolution : public LayerT {
public:
    Deconvolution();

    std::int32_t load_param(const ParamDictT& pd) override;
    std::int32_t load_model(const ModelBinT& mb) override;
    std::int32_t forward(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const override;
    std::int32_t forward(const std::vector<MatT>& bottom_blobs, std::vector<MatT>& top_blobs, const OptionT& opt) const override;

protected:
    void cut_padding(const MatT& top_blob_bordered, MatT& top_blob, const OptionT& opt) const;

    static std::int32_t deconvolution(const MatT& bottom_blob, MatT& top_blob, const MatT& weight_data,
                                      const MatT& bias_data, std::int32_t kernel_w, std::int32_t kernel_h,
                                      std::int32_t stride_w, std::int32_t stride_h, std::int32_t dilation_w,
                                      std::int32_t dilation_h, std::int32_t group,
                                      std::int32_t activation_type, const MatT& activation_params,
                                      const OptionT& opt);

public:
    std::int32_t num_output;
    std::int32_t kernel_w;
    std::int32_t kernel_h;
    std::int32_t dilation_w;
    std::int32_t dilation_h;
    std::int32_t stride_w;
    std::int32_t stride_h;
    std::int32_t pad_left;
    std::int32_t pad_right;
    std::int32_t pad_top;
    std::int32_t pad_bottom;
    std::int32_t output_pad_right;
    std::int32_t output_pad_bottom;
    std::int32_t output_w;
    std::int32_t output_h;
    std::int32_t bias_term;
    std::int32_t group;

    std::int32_t weight_data_size;

    std::int32_t activation_type;
    MatT activation_params;

    std::int32_t dynamic_weight;

    MatT weight_data;
    MatT bias_data;
};

#define MLAYER_DECONVOLUTION_TEMPLATE \
    template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, typename ModelBinT, bool Validate, \
             float (*Activation)(float, std::int32_t, const MatT&), \
             void (*CopyCutBorder)(const MatT&, MatT&, std::int32_t, std::int32_t, std::int32_t, std::int32_t, \
                                   const OptionT&), \
             void (*Flatten)(const MatT&, MatT&, const OptionT&)>

MLAYER_DECONVOLUTION_TEMPLATE
Deconvolution<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Activation, CopyCutBorder, Flatten>::Deconvolution()
{
    this->one_blob_only = true;
    this->support_inplace = false;
}

MLAYER_DECONVOLUTION_TEMPLATE
std::int32_t Deconvolution<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Activation, CopyCutBorder, Flatten>::load_param(const ParamDictT& pd)
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
    output_pad_right = pd.get(18, 0);
    output_pad_bottom = pd.get(19, output_pad_right);
    output_w = pd.get(20, 0);
    output_h = pd.get(21, output_w);
    bias_term = pd.get(5, 0);
    weight_data_size = pd.get(6, 0);
    group = pd.get(7, 1);
    activation_type = pd.get(9, 0);
    activation_params = pd.get(10, MatT());

    dynamic_weight = pd.get(28, 0);

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

    return 0;
}

MLAYER_DECONVOLUTION_TEMPLATE
std::int32_t Deconvolution<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Activation, CopyCutBorder, Flatten>::load_model(const ModelBinT& mb)
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

    return 0;
}

MLAYER_DECONVOLUTION_TEMPLATE
std::int32_t Deconvolution<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Activation, CopyCutBorder, Flatten>::deconvolution(const MatT& bottom_blob, MatT& top_blob, const MatT& weight_data, const MatT& bias_data, std::int32_t kernel_w, std::int32_t kernel_h, std::int32_t stride_w, std::int32_t stride_h, std::int32_t dilation_w, std::int32_t dilation_h, std::int32_t group, std::int32_t activation_type, const MatT& activation_params, const OptionT& opt)
{
    const std::int32_t outw = top_blob.w;
    const std::int32_t outch = top_blob.c;

    const std::int32_t maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<std::int32_t> _space_ofs(maxk);
    std::int32_t* space_ofs = &_space_ofs[0];
    {
        std::int32_t p1 = 0;
        std::int32_t p2 = 0;
        std::int32_t gap = outw * dilation_h - kernel_w * dilation_w;
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
        MatT out = top_blob.channel(p);

        const float bias = bias_data.empty() ? 0.f : bias_data[p];

        out.fill(bias);

        // shadowed variable for less openmp task args
        const std::int32_t w = bottom_blob.w;
        const std::int32_t h = bottom_blob.h;
        const std::int32_t inch = bottom_blob.c;
        const std::int32_t inch_g = inch / group;
        const std::int32_t g = p / (outch / group);
        const std::int32_t outw = top_blob.w;
        const std::int32_t outh = top_blob.h;

        for (std::int32_t i = 0; i < h; i++)
        {
            for (std::int32_t j = 0; j < w; j++)
            {
                float* outptr = out.row(i * stride_h) + j * stride_w;

                const float* kptr = static_cast<const float*>(weight_data) + maxk * inch_g * p;

                for (std::int32_t q = g * inch_g; q < (g + 1) * inch_g; q++)
                {
                    const float val = bottom_blob.channel(q).row(i)[j];

                    for (std::int32_t k = 0; k < maxk; k++)
                    {
                        float w = kptr[k];
                        outptr[space_ofs[k]] += val * w;
                    }

                    kptr += maxk;
                }
            }
        }

        {
            float* outptr = out;
            std::int32_t size = outw * outh;

            for (std::int32_t i = 0; i < size; i++)
            {
                outptr[i] = Activation(outptr[i], activation_type, activation_params);
            }
        }
    }

    return 0;
}

MLAYER_DECONVOLUTION_TEMPLATE
std::int32_t Deconvolution<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Activation, CopyCutBorder, Flatten>::forward(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const
{
    if (group <= 0 || num_output % group != 0 || bottom_blob.c % group != 0) {
        return -1;
    }
    std::int32_t w = bottom_blob.w;
    std::int32_t h = bottom_blob.h;
    std::size_t elemsize = bottom_blob.elemsize;

    const std::int32_t kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const std::int32_t kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    std::int32_t outw = (w - 1) * stride_w + kernel_extent_w + output_pad_right;
    std::int32_t outh = (h - 1) * stride_h + kernel_extent_h + output_pad_bottom;

    MatT top_blob_bordered;
    if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0 || (output_w > 0 && output_h > 0))
    {
        top_blob_bordered.create(outw, outh, num_output, elemsize, opt.workspace_allocator);
    }
    else
    {
        top_blob_bordered = top_blob;
        top_blob_bordered.create(outw, outh, num_output, elemsize, opt.blob_allocator);
    }
    if (top_blob_bordered.empty())
        return -100;

    std::int32_t ret = deconvolution(bottom_blob, top_blob_bordered, weight_data, bias_data, kernel_w, kernel_h,
                                     stride_w, stride_h, dilation_w, dilation_h, group, activation_type,
                                     activation_params, opt);
    if (ret != 0)
        return ret;

    cut_padding(top_blob_bordered, top_blob, opt);
    if (top_blob.empty())
        return -100;

    return 0;
}

MLAYER_DECONVOLUTION_TEMPLATE
std::int32_t Deconvolution<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Activation, CopyCutBorder, Flatten>::forward(const std::vector<MatT>& bottom_blobs, std::vector<MatT>& top_blobs, const OptionT& opt) const
{
    const MatT& bottom_blob = bottom_blobs[0];
    const MatT& _weight_data = bottom_blobs[1];
    MatT& top_blob = top_blobs[0];

    const std::int32_t _num_input = bottom_blob.c;
    const std::int32_t _kernel_w = _weight_data.w;
    const std::int32_t _kernel_h = _weight_data.h;
    const std::int32_t _num_output = _weight_data.d * group;

    MatT weight_data_flattened;
    Flatten(_weight_data, weight_data_flattened, opt);
    if (weight_data_flattened.empty())
        return -100;

    // transpose group-inch/group-outch/group-kh-kw to group-outch/group-inch/group-kh-kw
    MatT weight_data_transposed;
    {
        weight_data_transposed.create(_kernel_w * _kernel_h * _num_output * _num_input / group, 4u, opt.workspace_allocator);
        if (weight_data_transposed.empty())
            return -100;

        const std::int32_t outch_g = _num_output / group;
        const std::int32_t inch_g = _num_input / group;
        const std::int32_t maxk = _kernel_h * _kernel_w;

        for (std::int32_t g = 0; g < group; g++)
        {
            // reorder weight from inch-outch to outch-inch
            float* wg2 = (float*)weight_data_transposed + g * outch_g * inch_g * maxk;
            const float* wg = (const float*)weight_data_flattened + g * inch_g * outch_g * maxk;
            for (std::int32_t i = 0; i < outch_g; i++)
            {
                for (std::int32_t j = 0; j < inch_g; j++)
                {
                    for (std::int32_t k = 0; k < maxk; k++)
                    {
                        wg2[(i * inch_g + j) * maxk + k] = wg[(j * outch_g + i) * maxk + k];
                    }
                }
            }
        }
    }

    MatT bias_data_flattened;
    if (bias_term)
    {
        const MatT& _bias_data = bottom_blobs[2];
        Flatten(_bias_data, bias_data_flattened, opt);
        if (bias_data_flattened.empty())
            return -100;
    }

    const std::int32_t w = bottom_blob.w;
    const std::int32_t h = bottom_blob.h;

    const std::int32_t kernel_extent_w = dilation_w * (_kernel_w - 1) + 1;
    const std::int32_t kernel_extent_h = dilation_h * (_kernel_h - 1) + 1;

    std::int32_t outw = (w - 1) * stride_w + kernel_extent_w + output_pad_right;
    std::int32_t outh = (h - 1) * stride_h + kernel_extent_h + output_pad_bottom;

    MatT top_blob_bordered;
    if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0 || (output_w > 0 && output_h > 0))
    {
        top_blob_bordered.create(outw, outh, _num_output, 4u, opt.workspace_allocator);
    }
    else
    {
        top_blob_bordered = top_blob;
        top_blob_bordered.create(outw, outh, _num_output, 4u, opt.blob_allocator);
    }
    if (top_blob_bordered.empty())
        return -100;

    std::int32_t ret = deconvolution(bottom_blob, top_blob_bordered, weight_data_transposed, bias_data_flattened, _kernel_w, _kernel_h, stride_w, stride_h,
                                     dilation_w, dilation_h, group, activation_type, activation_params, opt);
    if (ret != 0)
        return ret;

    cut_padding(top_blob_bordered, top_blob, opt);
    if (top_blob.empty())
        return -100;

    return 0;
}

MLAYER_DECONVOLUTION_TEMPLATE
void Deconvolution<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Activation, CopyCutBorder, Flatten>::cut_padding(const MatT& top_blob_bordered, MatT& top_blob, const OptionT& opt) const
{
    if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0)
    {
        CopyCutBorder(top_blob_bordered, top_blob, pad_top, pad_bottom, pad_left, pad_right, opt);
    }
    else if (output_w > 0 && output_h > 0)
    {
        std::int32_t wcut = top_blob_bordered.w - output_w;
        std::int32_t hcut = top_blob_bordered.h - output_h;

        if (pad_left == -233 || pad_right == -233 || pad_top == -233 || pad_bottom == -233)
        {
            // onnx padding=SAME_UPPER
            CopyCutBorder(top_blob_bordered, top_blob, hcut / 2, hcut - hcut / 2, wcut / 2, wcut - wcut / 2, opt);
        }
        else if (pad_left == -234 || pad_right == -234 || pad_top == -234 || pad_bottom == -234)
        {
            // onnx padding=SAME_LOWER
            CopyCutBorder(top_blob_bordered, top_blob, hcut - hcut / 2, hcut / 2, wcut - wcut / 2, wcut / 2, opt);
        }
    }
    else
    {
        top_blob = top_blob_bordered;
    }
}


#undef MLAYER_DECONVOLUTION_TEMPLATE

}  // namespace mlayer

#endif  // MLAYER_DECONVOLUTION_HPP
