// Copyright 2021 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_CONVOLUTION1D_HPP
#define MLAYER_CONVOLUTION1D_HPP

#include <climits>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <mlayer/border.hpp>


namespace mlayer {

template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, typename ModelBinT, bool Validate,
         float (*Activation)(float, std::int32_t, const MatT&),
         void (*CopyMakeBorder)(const MatT&, MatT&, std::int32_t, std::int32_t, std::int32_t, std::int32_t, std::int32_t,
                                float, const OptionT&),
         void (*Flatten)(const MatT&, MatT&, const OptionT&)>
class Convolution1D : public LayerT {
public:
    Convolution1D();

    std::int32_t load_param(const ParamDictT& pd) override;
    std::int32_t load_model(const ModelBinT& mb) override;
    std::int32_t forward(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const override;
    std::int32_t forward(const std::vector<MatT>& bottom_blobs, std::vector<MatT>& top_blobs, const OptionT& opt) const override;

protected:
    void make_padding(const MatT& bottom_blob, MatT& bottom_blob_bordered, const OptionT& opt) const;
    void make_padding(const MatT& bottom_blob, MatT& bottom_blob_bordered, std::int32_t kernel_w, const OptionT& opt) const;

    static std::int32_t convolution1d(const MatT& bottom_blob, MatT& top_blob, const MatT& weight_data,
                                      const MatT& bias_data, std::int32_t kernel_w, std::int32_t stride_w,
                                      std::int32_t dilation_w, std::int32_t activation_type,
                                      const MatT& activation_params, const OptionT& opt);

public:
    std::int32_t num_output;
    std::int32_t kernel_w;
    std::int32_t dilation_w;
    std::int32_t stride_w;
    std::int32_t pad_left;
    std::int32_t pad_right;
    float pad_value;
    std::int32_t bias_term;
    std::int32_t weight_data_size;
    std::int32_t activation_type;
    MatT activation_params;
    bool dynamic_weight;
    MatT weight_data;
    MatT bias_data;
};

#define MLAYER_CONVOLUTION1D_TEMPLATE \
    template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, typename ModelBinT, bool Validate, \
             float (*Activation)(float, std::int32_t, const MatT&), \
             void (*CopyMakeBorder)(const MatT&, MatT&, std::int32_t, std::int32_t, std::int32_t, std::int32_t, std::int32_t, \
                                    float, const OptionT&), \
             void (*Flatten)(const MatT&, MatT&, const OptionT&)>

MLAYER_CONVOLUTION1D_TEMPLATE
Convolution1D<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Activation, CopyMakeBorder, Flatten>::Convolution1D()
{
    this->one_blob_only = true;
    this->support_inplace = false;
}

MLAYER_CONVOLUTION1D_TEMPLATE
std::int32_t Convolution1D<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Activation, CopyMakeBorder, Flatten>::load_param(const ParamDictT& pd)
{
    num_output = pd.get(0, 0);
    kernel_w = pd.get(1, 0);
    dilation_w = pd.get(2, 1);
    stride_w = pd.get(3, 1);
    pad_left = pd.get(4, 0);
    pad_right = pd.get(15, pad_left);
    pad_value = pd.get(18, 0.f);
    bias_term = pd.get(5, 0);
    weight_data_size = pd.get(6, 0);
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

    if (dilation_w <= 0 || stride_w <= 0)
        return -1;

    if (!dynamic_weight)
    {
        if (kernel_w <= 0 || kernel_w - 1 > (INT_MAX - 1) / dilation_w)
            return -1;

        if (num_output <= 0 || weight_data_size <= 0 || weight_data_size % num_output != 0 || (weight_data_size / num_output) % kernel_w != 0)
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

MLAYER_CONVOLUTION1D_TEMPLATE
std::int32_t Convolution1D<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Activation, CopyMakeBorder, Flatten>::load_model(const ModelBinT& mb)
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

MLAYER_CONVOLUTION1D_TEMPLATE
std::int32_t Convolution1D<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Activation, CopyMakeBorder, Flatten>::convolution1d(const MatT& bottom_blob, MatT& top_blob, const MatT& weight_data, const MatT& bias_data, std::int32_t kernel_w, std::int32_t stride_w, std::int32_t dilation_w, std::int32_t activation_type, const MatT& activation_params, const OptionT& opt)
{
    const std::int32_t h = bottom_blob.h;

    const std::int32_t outw = top_blob.w;
    const std::int32_t outh = top_blob.h;

    const std::int32_t bias_term = bias_data.empty() ? 0 : 1;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (std::int32_t p = 0; p < outh; p++)
    {
        float* outptr = top_blob.row(p);

        for (std::int32_t j = 0; j < outw; j++)
        {
            float sum = 0.f;

            if (bias_term)
                sum = bias_data[p];

            const float* kptr = (const float*)weight_data + kernel_w * h * p;

            for (std::int32_t q = 0; q < h; q++)
            {
                const float* sptr = bottom_blob.row(q) + j * stride_w;

                for (std::int32_t k = 0; k < kernel_w; k++)
                {
                    float val = *sptr;
                    float wt = kptr[k];
                    sum += val * wt;

                    sptr += dilation_w;
                }

                kptr += kernel_w;
            }

            sum = Activation(sum, activation_type, activation_params);

            outptr[j] = sum;
        }
    }

    return 0;
}

MLAYER_CONVOLUTION1D_TEMPLATE
std::int32_t Convolution1D<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Activation, CopyMakeBorder, Flatten>::forward(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const
{
    MatT bottom_blob_bordered;
    make_padding(bottom_blob, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    const std::int32_t w = bottom_blob_bordered.w;
    const std::size_t elemsize = bottom_blob_bordered.elemsize;

    const std::int32_t kernel_extent_w = dilation_w * (kernel_w - 1) + 1;

    const std::int32_t outw = (w - kernel_extent_w) / stride_w + 1;

    top_blob.create(outw, num_output, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    std::int32_t ret = convolution1d(bottom_blob_bordered, top_blob, weight_data, bias_data, kernel_w, stride_w, dilation_w, activation_type, activation_params, opt);
    if (ret != 0)
        return ret;

    return 0;
}

MLAYER_CONVOLUTION1D_TEMPLATE
std::int32_t Convolution1D<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Activation, CopyMakeBorder, Flatten>::forward(const std::vector<MatT>& bottom_blobs, std::vector<MatT>& top_blobs, const OptionT& opt) const
{
    const MatT& bottom_blob = bottom_blobs[0];
    const MatT& _weight_data = bottom_blobs[1];
    MatT& top_blob = top_blobs[0];

    const std::int32_t _kernel_w = _weight_data.w;
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
    make_padding(bottom_blob, bottom_blob_bordered, _kernel_w, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    const std::int32_t w = bottom_blob_bordered.w;
    const std::size_t elemsize = bottom_blob_bordered.elemsize;

    const std::int32_t kernel_extent_w = dilation_w * (_kernel_w - 1) + 1;

    const std::int32_t outw = (w - kernel_extent_w) / stride_w + 1;

    top_blob.create(outw, _num_output, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    std::int32_t ret = convolution1d(bottom_blob_bordered, top_blob, weight_data_flattened, bias_data_flattened, _kernel_w, stride_w, dilation_w, activation_type, activation_params, opt);
    if (ret != 0)
        return ret;

    return 0;
}

MLAYER_CONVOLUTION1D_TEMPLATE
void Convolution1D<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Activation, CopyMakeBorder, Flatten>::make_padding(const MatT& bottom_blob, MatT& bottom_blob_bordered, const OptionT& opt) const
{
    make_padding(bottom_blob, bottom_blob_bordered, kernel_w, opt);
}

MLAYER_CONVOLUTION1D_TEMPLATE
void Convolution1D<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Activation, CopyMakeBorder, Flatten>::make_padding(const MatT& bottom_blob, MatT& bottom_blob_bordered, std::int32_t _kernel_w, const OptionT& opt) const
{
    std::int32_t w = bottom_blob.w;

    const std::int32_t kernel_extent_w = dilation_w * (_kernel_w - 1) + 1;

    bottom_blob_bordered = bottom_blob;
    if (pad_left > 0 || pad_right > 0)
    {
        OptionT opt_b = opt;
        opt_b.blob_allocator = opt.workspace_allocator;
        CopyMakeBorder(bottom_blob, bottom_blob_bordered, 0, 0, pad_left, pad_right, kBorderConstant, pad_value, opt_b);
    }
    else if (pad_left == -233 && pad_right == -233)
    {
        // tensorflow padding=SAME or onnx padding=SAME_UPPER
        std::int32_t wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        if (wpad > 0)
        {
            OptionT opt_b = opt;
            opt_b.blob_allocator = opt.workspace_allocator;
            CopyMakeBorder(bottom_blob, bottom_blob_bordered, 0, 0, wpad / 2, wpad - wpad / 2, kBorderConstant, pad_value, opt_b);
        }
    }
    else if (pad_left == -234 && pad_right == -234)
    {
        // onnx padding=SAME_LOWER
        std::int32_t wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        if (wpad > 0)
        {
            OptionT opt_b = opt;
            opt_b.blob_allocator = opt.workspace_allocator;
            CopyMakeBorder(bottom_blob, bottom_blob_bordered, 0, 0, wpad - wpad / 2, wpad / 2, kBorderConstant, pad_value, opt_b);
        }
    }
}


#undef MLAYER_CONVOLUTION1D_TEMPLATE

}  // namespace mlayer

#endif  // MLAYER_CONVOLUTION1D_HPP
