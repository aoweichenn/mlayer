// Copyright 2022 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_DEFORMABLECONV2D_HPP
#define MLAYER_DEFORMABLECONV2D_HPP

#include <cmath>
#include <cstdint>
#include <vector>

namespace mlayer {

// Deformable convolution: bilinear sampling at per-tap offsets, optional mask.
template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, typename ModelBinT,
         bool Validate = false, float (*Activation)(float, std::int32_t, const MatT&) = nullptr>
class DeformableConv2D : public LayerT {
public:
    DeformableConv2D();

    std::int32_t load_param(const ParamDictT& pd) override;
    std::int32_t load_model(const ModelBinT& mb) override;
    std::int32_t forward(const std::vector<MatT>& bottom_blobs, std::vector<MatT>& top_blobs,
                         const OptionT& opt) const override;

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
    std::int32_t bias_term;
    std::int32_t weight_data_size;
    std::int32_t activation_type;
    MatT activation_params;
    MatT weight_data;
    MatT bias_data;
};

#define MLAYER_DEFORMCONV_TEMPLATE                                                                     \
    template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, typename ModelBinT, \
             bool Validate, float (*Activation)(float, std::int32_t, const MatT&)>

MLAYER_DEFORMCONV_TEMPLATE
DeformableConv2D<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Activation>::DeformableConv2D() {
    this->one_blob_only = false;
    this->support_inplace = false;
    this->num_output = 0;
    this->kernel_w = 0;
    this->kernel_h = 0;
    this->dilation_w = 1;
    this->dilation_h = 1;
    this->stride_w = 1;
    this->stride_h = 1;
    this->pad_left = 0;
    this->pad_right = 0;
    this->pad_top = 0;
    this->pad_bottom = 0;
    this->bias_term = 0;
    this->weight_data_size = 0;
    this->activation_type = 0;
}

MLAYER_DEFORMCONV_TEMPLATE
std::int32_t DeformableConv2D<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Activation>::load_param(
    const ParamDictT& pd) {
    this->num_output = pd.get(0, 0);
    this->kernel_w = pd.get(1, 0);
    this->kernel_h = pd.get(11, this->kernel_w);
    this->dilation_w = pd.get(2, 1);
    this->dilation_h = pd.get(12, this->dilation_w);
    this->stride_w = pd.get(3, 1);
    this->stride_h = pd.get(13, this->stride_w);
    this->pad_left = pd.get(4, 0);
    this->pad_right = pd.get(15, this->pad_left);
    this->pad_top = pd.get(14, this->pad_left);
    this->pad_bottom = pd.get(16, this->pad_top);
    this->bias_term = pd.get(5, 0);
    this->weight_data_size = pd.get(6, 0);
    this->activation_type = pd.get(9, 0);
    this->activation_params = pd.get(10, MatT());
    if (this->kernel_w <= 0 || this->num_output <= 0) {
        return -1;
    }
    return 0;
}

MLAYER_DEFORMCONV_TEMPLATE
std::int32_t DeformableConv2D<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Activation>::load_model(
    const ModelBinT& mb) {
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

MLAYER_DEFORMCONV_TEMPLATE
std::int32_t DeformableConv2D<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Activation>::forward(
    const std::vector<MatT>& bottom_blobs, std::vector<MatT>& top_blobs, const OptionT& opt) const {
    if (bottom_blobs.size() < 2 || top_blobs.empty()) {
        return -1;
    }
    const MatT& bottom_blob = bottom_blobs[0];
    const MatT& offset = bottom_blobs[1];
    const bool has_mask = bottom_blobs.size() == 3;

    const std::int32_t w = bottom_blob.w;
    const std::int32_t h = bottom_blob.h;
    const std::int32_t in_c = bottom_blob.c;
    const std::size_t elemsize = bottom_blob.elemsize;
    const std::int32_t maxk = this->kernel_w * this->kernel_h;

    const std::int32_t kernel_extent_w = this->dilation_w * (this->kernel_w - 1) + 1;
    const std::int32_t kernel_extent_h = this->dilation_h * (this->kernel_h - 1) + 1;
    const std::int32_t out_w =
        (w + this->pad_left + this->pad_right - kernel_extent_w) / this->stride_w + 1;
    const std::int32_t out_h =
        (h + this->pad_top + this->pad_bottom - kernel_extent_h) / this->stride_h + 1;
    if (out_w <= 0 || out_h <= 0) {
        return -1;
    }

    MatT& output = top_blobs[0];
    output.create(out_w, out_h, this->num_output, elemsize, opt.blob_allocator);
    if (output.empty()) {
        return -100;
    }

    const float* weight_ptr = this->weight_data;
    const float* bias_ptr = this->bias_term != 0 ? static_cast<const float*>(this->bias_data.data)
                                                 : static_cast<const float*>(this->weight_data.data);
    const MatT& mask = has_mask ? bottom_blobs[2] : MatT();

#pragma omp parallel for num_threads(opt.num_threads)
    for (std::int32_t h_col = 0; h_col < out_h; ++h_col) {
        for (std::int32_t w_col = 0; w_col < out_w; ++w_col) {
            const std::int32_t h_in = h_col * this->stride_h - this->pad_top;
            const std::int32_t w_in = w_col * this->stride_w - this->pad_left;
            for (std::int32_t oc = 0; oc < this->num_output; ++oc) {
                float sum = this->bias_term != 0 ? bias_ptr[oc] : 0.0F;
                for (std::int32_t i = 0; i < this->kernel_h; ++i) {
                    for (std::int32_t j = 0; j < this->kernel_w; ++j) {
                        const std::int32_t tap = i * this->kernel_w + j;
                        const float offset_h = offset.channel(tap * 2).row(h_col)[w_col];
                        const float offset_w = offset.channel(tap * 2 + 1).row(h_col)[w_col];
                        const float mask_v = has_mask ? mask.channel(tap).row(h_col)[w_col] : 1.0F;
                        const float h_im = static_cast<float>(h_in + i * this->dilation_h) + offset_h;
                        const float w_im = static_cast<float>(w_in + j * this->dilation_w) + offset_w;

                        const bool cond = h_im > -1.0F && w_im > -1.0F && h_im < static_cast<float>(h) &&
                                          w_im < static_cast<float>(w);
                        std::int32_t h_low = 0;
                        std::int32_t w_low = 0;
                        std::int32_t h_high = 0;
                        std::int32_t w_high = 0;
                        float w1 = 0.0F;
                        float w2 = 0.0F;
                        float w3 = 0.0F;
                        float w4 = 0.0F;
                        bool v1_cond = false;
                        bool v2_cond = false;
                        bool v3_cond = false;
                        bool v4_cond = false;
                        if (cond) {
                            h_low = static_cast<std::int32_t>(std::floor(h_im));
                            w_low = static_cast<std::int32_t>(std::floor(w_im));
                            h_high = h_low + 1;
                            w_high = w_low + 1;
                            const float lh = h_im - static_cast<float>(h_low);
                            const float lw = w_im - static_cast<float>(w_low);
                            const float hh = 1.0F - lh;
                            const float hw = 1.0F - lw;
                            v1_cond = h_low >= 0 && w_low >= 0;
                            v2_cond = h_low >= 0 && w_high <= w - 1;
                            v3_cond = h_high <= h - 1 && w_low >= 0;
                            v4_cond = h_high <= h - 1 && w_high <= w - 1;
                            w1 = hh * hw;
                            w2 = hh * lw;
                            w3 = lh * hw;
                            w4 = lh * lw;
                        }

                        for (std::int32_t c_im = 0; c_im < in_c; ++c_im) {
                            float val = 0.0F;
                            if (cond) {
                                const MatT ch = bottom_blob.channel(c_im);
                                const float v1 = v1_cond ? ch.row(h_low)[w_low] : 0.0F;
                                const float v2 = v2_cond ? ch.row(h_low)[w_high] : 0.0F;
                                const float v3 = v3_cond ? ch.row(h_high)[w_low] : 0.0F;
                                const float v4 = v4_cond ? ch.row(h_high)[w_high] : 0.0F;
                                val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
                            }
                            sum += val * mask_v *
                                   weight_ptr[((oc * in_c + c_im) * this->kernel_h + i) * this->kernel_w + j];
                        }
                    }
                }
                const float v = this->activation_type == 0
                                    ? sum
                                    : Activation(sum, this->activation_type, this->activation_params);
                output.channel(oc).row(h_col)[w_col] = v;
            }
        }
    }
    static_cast<void>(maxk);
    return 0;
}

}  // namespace mlayer

#endif  // MLAYER_DEFORMABLECONV2D_HPP
