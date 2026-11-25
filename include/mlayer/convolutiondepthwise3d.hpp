// Copyright 2022 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_CONVOLUTIONDEPTHWISE3D_HPP
#define MLAYER_CONVOLUTIONDEPTHWISE3D_HPP

#include <algorithm>
#include <cstdint>

namespace mlayer {

// 3D depthwise / group convolution over a (w, h, d, channels) blob.
template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, typename ModelBinT,
         bool Validate = false, float (*Activation)(float, std::int32_t, const MatT&) = nullptr>
class ConvolutionDepthWise3D : public LayerT {
public:
    ConvolutionDepthWise3D();

    std::int32_t load_param(const ParamDictT& pd) override;
    std::int32_t load_model(const ModelBinT& mb) override;
    std::int32_t forward(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const override;

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
    std::int32_t group;
    std::int32_t activation_type;
    MatT activation_params;
    MatT weight_data;
    MatT bias_data;
};

#define MLAYER_CONVDW3D_TEMPLATE                                                                       \
    template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, typename ModelBinT, \
             bool Validate, float (*Activation)(float, std::int32_t, const MatT&)>

MLAYER_CONVDW3D_TEMPLATE
ConvolutionDepthWise3D<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Activation>::
    ConvolutionDepthWise3D() {
    this->one_blob_only = true;
    this->support_inplace = false;
    this->num_output = 0;
    this->kernel_w = 0;
    this->kernel_h = 0;
    this->kernel_d = 0;
    this->dilation_w = 1;
    this->dilation_h = 1;
    this->dilation_d = 1;
    this->stride_w = 1;
    this->stride_h = 1;
    this->stride_d = 1;
    this->pad_left = 0;
    this->pad_right = 0;
    this->pad_top = 0;
    this->pad_bottom = 0;
    this->pad_front = 0;
    this->pad_behind = 0;
    this->pad_value = 0.0F;
    this->bias_term = 0;
    this->weight_data_size = 0;
    this->group = 1;
    this->activation_type = 0;
}

MLAYER_CONVDW3D_TEMPLATE
std::int32_t ConvolutionDepthWise3D<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Activation>::
    load_param(const ParamDictT& pd) {
    this->num_output = pd.get(0, 0);
    this->kernel_w = pd.get(1, 0);
    this->kernel_h = pd.get(11, this->kernel_w);
    this->kernel_d = pd.get(21, this->kernel_w);
    this->dilation_w = pd.get(2, 1);
    this->dilation_h = pd.get(12, this->dilation_w);
    this->dilation_d = pd.get(22, this->dilation_w);
    this->stride_w = pd.get(3, 1);
    this->stride_h = pd.get(13, this->stride_w);
    this->stride_d = pd.get(23, this->stride_w);
    this->pad_left = pd.get(4, 0);
    this->pad_right = pd.get(15, this->pad_left);
    this->pad_top = pd.get(14, this->pad_left);
    this->pad_bottom = pd.get(16, this->pad_top);
    this->pad_front = pd.get(24, this->pad_left);
    this->pad_behind = pd.get(17, this->pad_front);
    this->pad_value = pd.get(18, 0.0F);
    this->bias_term = pd.get(5, 0);
    this->weight_data_size = pd.get(6, 0);
    this->group = pd.get(7, 1);
    this->activation_type = pd.get(9, 0);
    this->activation_params = pd.get(10, MatT());
    if (this->kernel_w <= 0 || this->num_output <= 0 || this->group <= 0) {
        return -1;
    }
    return 0;
}

MLAYER_CONVDW3D_TEMPLATE
std::int32_t ConvolutionDepthWise3D<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Activation>::
    load_model(const ModelBinT& mb) {
    this->weight_data = mb.load(this->weight_data_size, 0);
    if (this->weight_data.empty()) {
        return -100;
    }
    if (this->bias_term != 0) {
        this->bias_data = mb.load(this->num_output, 1);
        if (this->bias_data.empty()) {
            return -100;
        }
    }
    return 0;
}

MLAYER_CONVDW3D_TEMPLATE
std::int32_t ConvolutionDepthWise3D<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Activation>::
    forward(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const {
    const std::int32_t w = bottom_blob.w;
    const std::int32_t h = bottom_blob.h;
    const std::int32_t d = bottom_blob.d;
    const std::int32_t inch = bottom_blob.c;
    const std::int32_t outch = this->num_output;
    const std::size_t elemsize = bottom_blob.elemsize;
    const std::int32_t maxk = this->kernel_w * this->kernel_h * this->kernel_d;

    const std::int32_t extent_w = this->dilation_w * (this->kernel_w - 1) + 1;
    const std::int32_t extent_h = this->dilation_h * (this->kernel_h - 1) + 1;
    const std::int32_t extent_d = this->dilation_d * (this->kernel_d - 1) + 1;
    const std::int32_t outw =
        (w + this->pad_left + this->pad_right - extent_w) / this->stride_w + 1;
    const std::int32_t outh = (h + this->pad_top + this->pad_bottom - extent_h) / this->stride_h + 1;
    const std::int32_t outd = (d + this->pad_front + this->pad_behind - extent_d) / this->stride_d + 1;
    if (outw <= 0 || outh <= 0 || outd <= 0 || inch % this->group != 0 || outch % this->group != 0) {
        return -1;
    }

    top_blob.create(outw, outh, outd, outch, elemsize, opt.blob_allocator);
    if (top_blob.empty()) {
        return -100;
    }

    const std::int32_t inch_g = inch / this->group;
    const std::int32_t outch_g = outch / this->group;
    const float* weight = this->weight_data;

#pragma omp parallel for num_threads(opt.num_threads)
    for (std::int32_t p = 0; p < outch; ++p) {
        const std::int32_t g = p / outch_g;
        const std::int32_t p_in_group = p - g * outch_g;
        const float* kptr =
            weight + static_cast<std::size_t>(g) * outch_g * inch_g * maxk +
            static_cast<std::size_t>(p_in_group) * inch_g * maxk;
        MatT out = top_blob.channel(p);
        for (std::int32_t oz = 0; oz < outd; ++oz) {
            for (std::int32_t oy = 0; oy < outh; ++oy) {
                float* outptr = out.depth(oz).row(oy);
                for (std::int32_t ox = 0; ox < outw; ++ox) {
                    float sum = this->bias_term != 0 ? this->bias_data[p] : 0.0F;
                    for (std::int32_t q = 0; q < inch_g; ++q) {
                        const MatT src = bottom_blob.channel(g * inch_g + q);
                        const float* kk = kptr + static_cast<std::size_t>(q) * maxk;
                        for (std::int32_t kz = 0; kz < this->kernel_d; ++kz) {
                            const std::int32_t sz = oz * this->stride_d + kz * this->dilation_d -
                                                    this->pad_front;
                            for (std::int32_t ky = 0; ky < this->kernel_h; ++ky) {
                                const std::int32_t sy = oy * this->stride_h + ky * this->dilation_h -
                                                        this->pad_top;
                                const float* row = (sz < 0 || sz >= d || sy < 0 || sy >= h)
                                                       ? nullptr
                                                       : src.depth(sz).row(sy);
                                for (std::int32_t kx = 0; kx < this->kernel_w; ++kx) {
                                    const std::int32_t sx = ox * this->stride_w + kx * this->dilation_w -
                                                            this->pad_left;
                                    const float kv = kk[(kz * this->kernel_h + ky) * this->kernel_w + kx];
                                    if (row == nullptr || sx < 0 || sx >= w) {
                                        sum += kv * this->pad_value;
                                    } else {
                                        sum += row[sx] * kv;
                                    }
                                }
                            }
                        }
                    }
                    outptr[ox] = this->activation_type == 0
                                     ? sum
                                     : Activation(sum, this->activation_type, this->activation_params);
                }
            }
        }
    }
    return 0;
}

}  // namespace mlayer

#endif  // MLAYER_CONVOLUTIONDEPTHWISE3D_HPP
