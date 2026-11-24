// Copyright 2021 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_CONVOLUTION3D_HPP
#define MLAYER_CONVOLUTION3D_HPP

#include <climits>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <mlayer/border.hpp>


namespace mlayer {

template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, typename ModelBinT, bool Validate,
         float (*Activation)(float, std::int32_t, const MatT&),
         void (*CopyMakeBorder3D)(const MatT&, MatT&, std::int32_t, std::int32_t, std::int32_t, std::int32_t,
                                  std::int32_t, std::int32_t, std::int32_t, float, const OptionT&)>
class Convolution3D : public LayerT {
public:
    Convolution3D();

    std::int32_t load_param(const ParamDictT& pd) override;
    std::int32_t load_model(const ModelBinT& mb) override;
    std::int32_t forward(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const override;

protected:
    void make_padding(const MatT& bottom_blob, MatT& bottom_blob_bordered, const OptionT& opt) const;

public:
    std::int32_t num_output;
    std::int32_t kernel_w;
    std::int32_t kernel_h;
    std::int32_t kernel_d;
    std::int32_t dilation_w;
    std::int32_t dilation_h;
    std::int32_t dilation_d;
    std::int32_t stride_w;
    std::int32_t stride_h;
    std::int32_t stride_d;
    std::int32_t pad_left;
    std::int32_t pad_right;
    std::int32_t pad_top;
    std::int32_t pad_bottom;
    std::int32_t pad_front;
    std::int32_t pad_behind;
    float pad_value;
    std::int32_t bias_term;
    std::int32_t weight_data_size;
    std::int32_t activation_type;
    MatT activation_params;
    MatT weight_data;
    MatT bias_data;
};

#define MLAYER_CONVOLUTION3D_TEMPLATE \
    template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, typename ModelBinT, bool Validate, \
             float (*Activation)(float, std::int32_t, const MatT&), \
             void (*CopyMakeBorder3D)(const MatT&, MatT&, std::int32_t, std::int32_t, std::int32_t, std::int32_t, \
                                      std::int32_t, std::int32_t, std::int32_t, float, const OptionT&)>

MLAYER_CONVOLUTION3D_TEMPLATE
Convolution3D<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Activation, CopyMakeBorder3D>::Convolution3D()
{
    this->one_blob_only = true;
    this->support_inplace = false;
}

MLAYER_CONVOLUTION3D_TEMPLATE
std::int32_t Convolution3D<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Activation, CopyMakeBorder3D>::load_param(const ParamDictT& pd)
{
    num_output = pd.get(0, 0);
    kernel_w = pd.get(1, 0);
    kernel_h = pd.get(11, kernel_w);
    kernel_d = pd.get(21, kernel_w);
    dilation_w = pd.get(2, 1);
    dilation_h = pd.get(12, dilation_w);
    dilation_d = pd.get(22, dilation_w);
    stride_w = pd.get(3, 1);
    stride_h = pd.get(13, stride_w);
    stride_d = pd.get(23, stride_w);
    pad_left = pd.get(4, 0);
    pad_right = pd.get(15, pad_left);
    pad_top = pd.get(14, pad_left);
    pad_bottom = pd.get(16, pad_top);
    pad_front = pd.get(24, pad_left);
    pad_behind = pd.get(17, pad_front);
    pad_value = pd.get(18, 0.f);
    bias_term = pd.get(5, 0);
    weight_data_size = pd.get(6, 0);
    activation_type = pd.get(9, 0);
    activation_params = pd.get(10, MatT());

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

    if (kernel_w <= 0 || dilation_w <= 0 || stride_w <= 0 || kernel_w - 1 > (INT_MAX - 1) / dilation_w || kernel_h <= 0 || dilation_h <= 0 || stride_h <= 0 || kernel_h - 1 > (INT_MAX - 1) / dilation_h || kernel_d <= 0 || dilation_d <= 0 || stride_d <= 0 || kernel_d - 1 > (INT_MAX - 1) / dilation_d || kernel_w > INT_MAX / kernel_h || kernel_w * kernel_h > INT_MAX / kernel_d)
        return -1;

    if (num_output <= 0 || weight_data_size <= 0 || weight_data_size % num_output != 0 || (weight_data_size / num_output) % (kernel_w * kernel_h * kernel_d) != 0)
        return -1;
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

    return 0;
}

MLAYER_CONVOLUTION3D_TEMPLATE
std::int32_t Convolution3D<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Activation, CopyMakeBorder3D>::load_model(const ModelBinT& mb)
{
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

MLAYER_CONVOLUTION3D_TEMPLATE
std::int32_t Convolution3D<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Activation, CopyMakeBorder3D>::forward(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const
{
    std::int32_t w = bottom_blob.w;
    std::int32_t h = bottom_blob.h;
    std::int32_t d = bottom_blob.d;
    std::int32_t channels = bottom_blob.c;
    std::size_t elemsize = bottom_blob.elemsize;

    const std::int32_t kernel_extend_w = dilation_w * (kernel_w - 1) + 1;
    const std::int32_t kernel_extend_h = dilation_h * (kernel_h - 1) + 1;
    const std::int32_t kernel_extend_d = dilation_d * (kernel_d - 1) + 1;

    MatT bottom_blob_bordered;
    OptionT opt_pad = opt;
    opt_pad.use_packing_layout = false;
    make_padding(bottom_blob, bottom_blob_bordered, opt_pad);
    if (bottom_blob_bordered.empty())
        return -100;

    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;
    d = bottom_blob_bordered.d;

    std::int32_t outw = (w - kernel_extend_w) / stride_w + 1;
    std::int32_t outh = (h - kernel_extend_h) / stride_h + 1;
    std::int32_t outd = (d - kernel_extend_d) / stride_d + 1;

    const std::int32_t maxk = kernel_w * kernel_h * kernel_d;

    // kernel offsets
    std::vector<std::int32_t> _space_ofs(maxk);
    std::int32_t* space_ofs = &_space_ofs[0];
    {
        std::int32_t p1 = 0;
        std::int32_t p2 = 0;
        std::int32_t gap0 = w * dilation_h - kernel_w * dilation_w;
        std::int32_t gap1 = h * w * dilation_d - w * kernel_h * dilation_h;
        for (std::int32_t z = 0; z < kernel_d; z++)
        {
            for (std::int32_t i = 0; i < kernel_h; i++)
            {
                for (std::int32_t j = 0; j < kernel_w; j++)
                {
                    space_ofs[p1] = p2;
                    p1++;
                    p2 += dilation_w;
                }
                p2 += gap0;
            }
            p2 += gap1;
        }
    }

    top_blob.create(outw, outh, outd, num_output, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (std::int32_t p = 0; p < num_output; p++)
    {
        float* outptr = top_blob.channel(p);

        for (std::int32_t z = 0; z < outd; z++)
        {
            for (std::int32_t i = 0; i < outh; i++)
            {
                for (std::int32_t j = 0; j < outw; j++)
                {
                    float sum = 0.f;

                    if (bias_term)
                        sum = bias_data[p];

                    const float* kptr = (const float*)weight_data + maxk * channels * p;

                    for (std::int32_t q = 0; q < channels; q++)
                    {
                        const MatT m = bottom_blob_bordered.channel(q);
                        const float* sptr = m.depth(z * stride_d).row(i * stride_h) + j * stride_w;

                        for (std::int32_t l = 0; l < maxk; l++)
                        {
                            float val = sptr[space_ofs[l]];

                            float wt = kptr[l];
                            sum += val * wt;
                        }

                        kptr += maxk;
                    }

                    outptr[j] = Activation(sum, activation_type, activation_params);
                }

                outptr += outw;
            }
        }
    }

    return 0;
}

MLAYER_CONVOLUTION3D_TEMPLATE
void Convolution3D<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Activation, CopyMakeBorder3D>::make_padding(const MatT& bottom_blob, MatT& bottom_blob_bordered, const OptionT& opt) const
{
    std::int32_t w = bottom_blob.w;
    std::int32_t h = bottom_blob.h;
    std::int32_t d = bottom_blob.d;

    const std::int32_t kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const std::int32_t kernel_extent_h = dilation_h * (kernel_h - 1) + 1;
    const std::int32_t kernel_extent_d = dilation_d * (kernel_d - 1) + 1;

    bottom_blob_bordered = bottom_blob;
    if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0 || pad_front > 0 || pad_behind > 0)
    {
        OptionT opt_b = opt;
        opt_b.blob_allocator = opt.workspace_allocator;
        CopyMakeBorder3D(bottom_blob, bottom_blob_bordered, pad_top, pad_bottom, pad_left, pad_right, pad_front, pad_behind, kBorderConstant, pad_value, opt_b);
    }
    else if (pad_left == -233 && pad_right == -233 && pad_top == -233 && pad_bottom == -233 && pad_front == -233 && pad_behind == -233)
    {
        // tensorflow padding=SAME or onnx padding=SAME_UPPER
        std::int32_t wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        std::int32_t hpad = kernel_extent_h + (h - 1) / stride_h * stride_h - h;
        std::int32_t dpad = kernel_extent_d + (d - 1) / stride_d * stride_d - d;
        if (wpad > 0 || hpad > 0 || dpad > 0)
        {
            OptionT opt_b = opt;
            opt_b.blob_allocator = opt.workspace_allocator;
            CopyMakeBorder3D(bottom_blob, bottom_blob_bordered, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, dpad / 2, dpad - dpad / 2, kBorderConstant, pad_value, opt_b);
        }
    }
    else if (pad_left == -234 && pad_right == -234 && pad_top == -234 && pad_bottom == -234 && pad_front == -234 && pad_behind == -234)
    {
        // onnx padding=SAME_LOWER
        std::int32_t wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        std::int32_t hpad = kernel_extent_h + (h - 1) / stride_h * stride_h - h;
        std::int32_t dpad = kernel_extent_d + (d - 1) / stride_d * stride_d - d;
        if (wpad > 0 || hpad > 0 || dpad > 0)
        {
            OptionT opt_b = opt;
            opt_b.blob_allocator = opt.workspace_allocator;
            CopyMakeBorder3D(bottom_blob, bottom_blob_bordered, hpad - hpad / 2, hpad / 2, wpad - wpad / 2, wpad / 2, dpad / 2, dpad - dpad / 2, kBorderConstant, pad_value, opt_b);
        }
    }
}


#undef MLAYER_CONVOLUTION3D_TEMPLATE

}  // namespace mlayer

#endif  // MLAYER_CONVOLUTION3D_HPP
