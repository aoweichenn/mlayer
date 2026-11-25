// Copyright 2024 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_QUANTIZE_LAYER_HPP
#define MLAYER_QUANTIZE_LAYER_HPP

#include <cmath>
#include <cstdint>

namespace mlayer {

// fp32 -> int8 with per-blob or per-channel scales (ncnn Quantize).
template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, typename ModelBinT>
class Quantize : public LayerT {
public:
    Quantize();

    std::int32_t load_param(const ParamDictT& pd) override;
    std::int32_t load_model(const ModelBinT& mb) override;
    std::int32_t forward(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const override;

public:
    std::int32_t scale_data_size;
    MatT scale_data;
};

#define MLAYER_QUANTIZE_TEMPLATE \
    template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, typename ModelBinT>

MLAYER_QUANTIZE_TEMPLATE
Quantize<LayerT, MatT, OptionT, ParamDictT, ModelBinT>::Quantize() {
    this->one_blob_only = true;
    this->support_inplace = false;
    this->scale_data_size = 1;
}

MLAYER_QUANTIZE_TEMPLATE
std::int32_t Quantize<LayerT, MatT, OptionT, ParamDictT, ModelBinT>::load_param(const ParamDictT& pd) {
    this->scale_data_size = pd.get(0, 1);
    return 0;
}

MLAYER_QUANTIZE_TEMPLATE
std::int32_t Quantize<LayerT, MatT, OptionT, ParamDictT, ModelBinT>::load_model(const ModelBinT& mb) {
    this->scale_data = mb.load(this->scale_data_size, 1);
    return this->scale_data.empty() ? -100 : 0;
}

MLAYER_QUANTIZE_TEMPLATE
std::int32_t Quantize<LayerT, MatT, OptionT, ParamDictT, ModelBinT>::forward(const MatT& bottom_blob,
                                                                           MatT& top_blob,
                                                                           const OptionT& opt) const {
    const std::int32_t dims = bottom_blob.dims;
    const std::int32_t w = bottom_blob.w;
    const std::int32_t h = bottom_blob.h;
    const std::int32_t d = bottom_blob.d;
    const std::int32_t channels = bottom_blob.c;
    const std::int32_t count = w * h * d * channels;

    if (dims == 1) {
        top_blob.create(w, static_cast<std::size_t>(1u), opt.blob_allocator);
    } else if (dims == 2) {
        top_blob.create(w, h, static_cast<std::size_t>(1u), opt.blob_allocator);
    } else if (dims == 3) {
        top_blob.create(w, h, channels, static_cast<std::size_t>(1u), opt.blob_allocator);
    } else {
        top_blob.create(w, h, d, channels, static_cast<std::size_t>(1u), opt.blob_allocator);
    }
    if (top_blob.empty()) {
        return -100;
    }

    const float* src = bottom_blob;
    auto* dst = static_cast<std::int8_t*>(top_blob.data);
    for (std::int32_t i = 0; i < count; ++i) {
        const std::int32_t channel = dims >= 3 ? i / (w * h * d) : 0;
        const float scale = this->scale_data_size == 1 ? this->scale_data[0] : this->scale_data[channel];
        const auto v = static_cast<std::int32_t>(std::round(src[i] * scale));
        dst[i] = static_cast<std::int8_t>(v > 127 ? 127 : (v < -127 ? -127 : v));
    }
    return 0;
}

// int32 -> fp32 with per-blob or per-channel scales and optional bias.
template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, typename ModelBinT>
class Dequantize : public LayerT {
public:
    Dequantize();

    std::int32_t load_param(const ParamDictT& pd) override;
    std::int32_t load_model(const ModelBinT& mb) override;
    std::int32_t forward(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const override;

public:
    std::int32_t scale_data_size;
    std::int32_t bias_data_size;
    MatT scale_data;
    MatT bias_data;
};

MLAYER_QUANTIZE_TEMPLATE
Dequantize<LayerT, MatT, OptionT, ParamDictT, ModelBinT>::Dequantize() {
    this->one_blob_only = true;
    this->support_inplace = false;
    this->scale_data_size = 1;
    this->bias_data_size = 0;
}

MLAYER_QUANTIZE_TEMPLATE
std::int32_t Dequantize<LayerT, MatT, OptionT, ParamDictT, ModelBinT>::load_param(const ParamDictT& pd) {
    this->scale_data_size = pd.get(0, 1);
    this->bias_data_size = pd.get(1, 0);
    return 0;
}

MLAYER_QUANTIZE_TEMPLATE
std::int32_t Dequantize<LayerT, MatT, OptionT, ParamDictT, ModelBinT>::load_model(const ModelBinT& mb) {
    this->scale_data = mb.load(this->scale_data_size, 1);
    if (this->scale_data.empty()) {
        return -100;
    }
    if (this->bias_data_size > 0) {
        this->bias_data = mb.load(this->bias_data_size, 1);
        if (this->bias_data.empty()) {
            return -100;
        }
    }
    return 0;
}

MLAYER_QUANTIZE_TEMPLATE
std::int32_t Dequantize<LayerT, MatT, OptionT, ParamDictT, ModelBinT>::forward(const MatT& bottom_blob,
                                                                             MatT& top_blob,
                                                                             const OptionT& opt) const {
    const std::int32_t dims = bottom_blob.dims;
    const std::int32_t w = bottom_blob.w;
    const std::int32_t h = bottom_blob.h;
    const std::int32_t d = bottom_blob.d;
    const std::int32_t channels = bottom_blob.c;
    const std::int32_t count = w * h * d * channels;

    if (dims == 1) {
        top_blob.create(w, 4u, opt.blob_allocator);
    } else if (dims == 2) {
        top_blob.create(w, h, 4u, opt.blob_allocator);
    } else if (dims == 3) {
        top_blob.create(w, h, channels, 4u, opt.blob_allocator);
    } else {
        top_blob.create(w, h, d, channels, 4u, opt.blob_allocator);
    }
    if (top_blob.empty()) {
        return -100;
    }

    const auto* src = static_cast<const std::int32_t*>(bottom_blob.data);
    float* dst = top_blob;
    for (std::int32_t i = 0; i < count; ++i) {
        const std::int32_t channel = dims >= 3 ? i / (w * h * d) : 0;
        const float scale = this->scale_data_size == 1 ? this->scale_data[0] : this->scale_data[channel];
        const float bias = this->bias_data_size == 0
                               ? 0.0F
                               : (this->bias_data_size == 1 ? this->bias_data[0] : this->bias_data[channel]);
        dst[i] = static_cast<float>(src[i]) * scale + bias;
    }
    return 0;
}

// int32 -> int8 requantization with activation.
template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, typename ModelBinT,
         float (*Activation)(float, std::int32_t, const MatT&)>
class Requantize : public LayerT {
public:
    Requantize();

    std::int32_t load_param(const ParamDictT& pd) override;
    std::int32_t load_model(const ModelBinT& mb) override;
    std::int32_t forward(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const override;

public:
    std::int32_t scale_in_data_size;
    std::int32_t scale_out_data_size;
    std::int32_t bias_data_size;
    std::int32_t activation_type;
    MatT activation_params;
    MatT scale_in_data;
    MatT scale_out_data;
    MatT bias_data;
};

#define MLAYER_REQUANTIZE_TEMPLATE                                                    \
    template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, \
             typename ModelBinT, float (*Activation)(float, std::int32_t, const MatT&)>

MLAYER_REQUANTIZE_TEMPLATE
Requantize<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Activation>::Requantize() {
    this->one_blob_only = true;
    this->support_inplace = false;
    this->scale_in_data_size = 1;
    this->scale_out_data_size = 1;
    this->bias_data_size = 0;
    this->activation_type = 0;
}

MLAYER_REQUANTIZE_TEMPLATE
std::int32_t Requantize<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Activation>::load_param(const ParamDictT& pd) {
    this->scale_in_data_size = pd.get(0, 1);
    this->scale_out_data_size = pd.get(1, 1);
    this->bias_data_size = pd.get(2, 0);
    this->activation_type = pd.get(3, 0);
    this->activation_params = pd.get(4, MatT());
    return 0;
}

MLAYER_REQUANTIZE_TEMPLATE
std::int32_t Requantize<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Activation>::load_model(const ModelBinT& mb) {
    this->scale_in_data = mb.load(this->scale_in_data_size, 1);
    if (this->scale_in_data.empty()) {
        return -100;
    }
    this->scale_out_data = mb.load(this->scale_out_data_size, 1);
    if (this->scale_out_data.empty()) {
        return -100;
    }
    if (this->bias_data_size > 0) {
        this->bias_data = mb.load(this->bias_data_size, 1);
        if (this->bias_data.empty()) {
            return -100;
        }
    }
    return 0;
}

MLAYER_REQUANTIZE_TEMPLATE
std::int32_t Requantize<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Activation>::forward(const MatT& bottom_blob,
                                                                                          MatT& top_blob,
                                                                                          const OptionT& opt) const {
    const std::int32_t dims = bottom_blob.dims;
    const std::int32_t w = bottom_blob.w;
    const std::int32_t h = bottom_blob.h;
    const std::int32_t d = bottom_blob.d;
    const std::int32_t channels = bottom_blob.c;
    const std::int32_t count = w * h * d * channels;

    if (dims == 1) {
        top_blob.create(w, static_cast<std::size_t>(1u), opt.blob_allocator);
    } else if (dims == 2) {
        top_blob.create(w, h, static_cast<std::size_t>(1u), opt.blob_allocator);
    } else if (dims == 3) {
        top_blob.create(w, h, channels, static_cast<std::size_t>(1u), opt.blob_allocator);
    } else {
        top_blob.create(w, h, d, channels, static_cast<std::size_t>(1u), opt.blob_allocator);
    }
    if (top_blob.empty()) {
        return -100;
    }

    const auto* src = static_cast<const std::int32_t*>(bottom_blob.data);
    auto* dst = static_cast<std::int8_t*>(top_blob.data);
    for (std::int32_t i = 0; i < count; ++i) {
        const std::int32_t channel = dims >= 3 ? i / (w * h * d) : 0;
        const float s_in = this->scale_in_data_size == 1 ? this->scale_in_data[0] : this->scale_in_data[channel];
        const float s_out = this->scale_out_data_size == 1 ? this->scale_out_data[0] : this->scale_out_data[channel];
        const float bias = this->bias_data_size == 0
                               ? 0.0F
                               : (this->bias_data_size == 1 ? this->bias_data[0] : this->bias_data[channel]);
        float v = static_cast<float>(src[i]) * s_in + bias;
        if (this->activation_type != 0) {
            v = Activation(v, this->activation_type, this->activation_params);
        }
        const auto q = static_cast<std::int32_t>(std::round(v * s_out));
        dst[i] = static_cast<std::int8_t>(q > 127 ? 127 : (q < -127 ? -127 : q));
    }
    return 0;
}

}  // namespace mlayer

#endif  // MLAYER_QUANTIZE_LAYER_HPP
