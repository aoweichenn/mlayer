// Copyright 2017 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_RNN_HPP
#define MLAYER_RNN_HPP

#include <cmath>
#include <cstdint>

#include <mlayer/int8_quantize.hpp>

namespace mlayer {

// Vanilla (Elman) RNN: h_t = tanh(W_xc x_t + W_hc h_{t-1} + b), optional
// bidirectional pass. int8 weights are dynamically dequantized per step, as in
// ncnn's rnn_int8 (per-row weight scales, dynamic activation/hidden scales).
template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, typename ModelBinT,
         bool Validate = false>
class RNN : public LayerT {
public:
    RNN();

    std::int32_t load_param(const ParamDictT& pd) override;
    std::int32_t load_model(const ModelBinT& mb) override;
    std::int32_t forward(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const override;

private:
    std::int32_t run(const MatT& bottom_blob, MatT& top_blob, std::int32_t reverse, const MatT& weight_xc,
                     const MatT& bias_c, const MatT& weight_hc, MatT& hidden, const OptionT& opt) const;
    std::int32_t run_int8(const MatT& bottom_blob, MatT& top_blob, std::int32_t reverse,
                          const MatT& weight_xc_int8, const float* weight_xc_scales, const MatT& bias_c,
                          const MatT& weight_hc_int8, const float* weight_hc_scales, MatT& hidden,
                          const OptionT& opt) const;

public:
    std::int32_t num_output;
    std::int32_t weight_data_size;
    std::int32_t direction;
    std::int32_t int8_scale_term;
    MatT weight_xc_data;
    MatT bias_c_data;
    MatT weight_hc_data;
    MatT weight_xc_data_int8_scales;
    MatT weight_hc_data_int8_scales;
};

#define MLAYER_RNN_TEMPLATE \
    template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, typename ModelBinT, bool Validate>

MLAYER_RNN_TEMPLATE
RNN<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate>::RNN() {
    this->one_blob_only = true;
    this->support_inplace = false;
    this->num_output = 0;
    this->weight_data_size = 0;
    this->direction = 0;
    this->int8_scale_term = 0;
}

MLAYER_RNN_TEMPLATE
std::int32_t RNN<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate>::load_param(const ParamDictT& pd) {
    this->num_output = pd.get(0, 0);
    this->weight_data_size = pd.get(1, 0);
    this->direction = pd.get(2, 0);
    this->int8_scale_term = pd.get(8, 0);
    if (this->num_output <= 0 || this->direction < 0 || this->direction > 2) {
        return -1;
    }
    return 0;
}

MLAYER_RNN_TEMPLATE
std::int32_t RNN<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate>::load_model(const ModelBinT& mb) {
    const std::int32_t num_directions = this->direction == 2 ? 2 : 1;
    const std::int32_t size = this->weight_data_size / num_directions / this->num_output;
    this->weight_xc_data = mb.load(size, this->num_output, num_directions, 0);
    this->bias_c_data = mb.load(this->num_output, 1, num_directions, 0);
    this->weight_hc_data = mb.load(this->num_output, this->num_output, num_directions, 0);
    if (this->weight_xc_data.empty() || this->bias_c_data.empty() || this->weight_hc_data.empty()) {
        return -100;
    }
    if (this->int8_scale_term != 0) {
        this->weight_xc_data_int8_scales = mb.load(this->num_output, num_directions, 1);
        this->weight_hc_data_int8_scales = mb.load(this->num_output, num_directions, 1);
        if (this->weight_xc_data_int8_scales.empty() || this->weight_hc_data_int8_scales.empty()) {
            return -100;
        }
    }
    return 0;
}

MLAYER_RNN_TEMPLATE
std::int32_t RNN<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate>::run(
    const MatT& bottom_blob, MatT& top_blob, std::int32_t reverse, const MatT& weight_xc, const MatT& bias_c,
    const MatT& weight_hc, MatT& hidden_state, const OptionT& opt) const {
    const std::int32_t size = bottom_blob.w;
    const std::int32_t T = bottom_blob.h;
    const std::int32_t num_output = top_blob.w;

    MatT gates;
    gates.create(num_output, 4u, opt.workspace_allocator);
    if (gates.empty()) {
        return -100;
    }

    for (std::int32_t t = 0; t < T; ++t) {
        const std::int32_t ti = reverse != 0 ? T - 1 - t : t;
        const float* x = bottom_blob.row(ti);
#pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t q = 0; q < num_output; ++q) {
            const float* wxc = weight_xc.row(q);
            const float* whc = weight_hc.row(q);
            float H = bias_c[q];
            for (std::int32_t i = 0; i < size; ++i) {
                H += wxc[i] * x[i];
            }
            for (std::int32_t i = 0; i < num_output; ++i) {
                H += whc[i] * hidden_state[i];
            }
            gates[q] = std::tanh(H);
        }
        float* output_data = top_blob.row(ti);
#pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t q = 0; q < num_output; ++q) {
            hidden_state[q] = gates[q];
            output_data[q] = gates[q];
        }
    }
    return 0;
}

MLAYER_RNN_TEMPLATE
std::int32_t RNN<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate>::run_int8(
    const MatT& bottom_blob, MatT& top_blob, std::int32_t reverse, const MatT& weight_xc_int8,
    const float* weight_xc_scales, const MatT& bias_c, const MatT& weight_hc_int8,
    const float* weight_hc_scales, MatT& hidden_state, const OptionT& opt) const {
    const std::int32_t size = bottom_blob.w;
    const std::int32_t T = bottom_blob.h;
    const std::int32_t num_output = top_blob.w;

    MatT gates;
    gates.create(num_output, 4u, opt.workspace_allocator);
    MatT bottom_blob_int8;
    bottom_blob_int8.create(size, T, 1u, opt.workspace_allocator);
    MatT bottom_blob_int8_scales;
    bottom_blob_int8_scales.create(T, 4u, opt.workspace_allocator);
    MatT hidden_state_int8;
    hidden_state_int8.create(num_output, 1u, opt.workspace_allocator);
    MatT hidden_state_int8_scales;
    hidden_state_int8_scales.create(1, 4u, opt.workspace_allocator);
    if (gates.empty() || bottom_blob_int8.empty() || bottom_blob_int8_scales.empty() || hidden_state_int8.empty() ||
        hidden_state_int8_scales.empty()) {
        return -100;
    }

    if (bottom_blob.dims <= 2) {
        // 1D/2D: ncnn's Quantize applies one scale per row.
        for (std::int32_t t = 0; t < T; ++t) {
            const float* x = bottom_blob.row(t);
            float absmax = 0.0F;
            for (std::int32_t i = 0; i < size; ++i) {
                absmax = std::max(absmax, std::fabs(x[i]));
            }
            const float scale = 127.0F / absmax;
            bottom_blob_int8_scales[t] = scale;
            detail::quantize_row_int8(x, bottom_blob_int8.template row<std::int8_t>(t), size, scale);
        }
    } else {
        // 3D/4D: ncnn's Quantize applies one scale per channel. RNN/GRU still
        // dequantize row t with scales[t], which is the reference behaviour.
        for (std::int32_t t = 0; t < T; ++t) {
            const float* x = bottom_blob.row(t);
            float absmax = 0.0F;
            for (std::int32_t i = 0; i < size; ++i) {
                absmax = std::max(absmax, std::fabs(x[i]));
            }
            bottom_blob_int8_scales[t] = 127.0F / absmax;
        }
        const float scale = bottom_blob_int8_scales[0];
        for (std::int32_t t = 0; t < T; ++t) {
            detail::quantize_row_int8(bottom_blob.row(t), bottom_blob_int8.template row<std::int8_t>(t), size,
                                      scale);
        }
    }

    auto* hidden_state_int8_ptr = static_cast<std::int8_t*>(hidden_state_int8.data);
    for (std::int32_t t = 0; t < T; ++t) {
        const std::int32_t ti = reverse != 0 ? T - 1 - t : t;
        {
            float absmax = 0.0F;
            for (std::int32_t i = 0; i < num_output; ++i) {
                absmax = std::max(absmax, std::fabs(hidden_state[i]));
            }
            if (absmax == 0.0F) {
                hidden_state_int8_scales[0] = 1.0F;
                detail::fill_int8(hidden_state_int8_ptr, num_output, 0);
            } else {
                const float scale = 127.0F / absmax;
                hidden_state_int8_scales[0] = scale;
                detail::quantize_row_int8(static_cast<const float*>(hidden_state.data), hidden_state_int8_ptr,
                                          num_output, scale);
            }
        }

        const std::int8_t* x = bottom_blob_int8.template row<std::int8_t>(ti);
        const std::int8_t* hs = hidden_state_int8_ptr;
        const float descale_x = 1.0F / bottom_blob_int8_scales[ti];
        const float descale_h = 1.0F / hidden_state_int8_scales[0];
#pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t q = 0; q < num_output; ++q) {
            const std::int8_t* wxc = weight_xc_int8.template row<std::int8_t>(q);
            const std::int8_t* whc = weight_hc_int8.template row<std::int8_t>(q);
            const float descale_xc = 1.0F / weight_xc_scales[q];
            const float descale_hc = 1.0F / weight_hc_scales[q];
            int Hx = 0;
            for (std::int32_t i = 0; i < size; ++i) {
                Hx += static_cast<int>(wxc[i]) * static_cast<int>(x[i]);
            }
            int Hh = 0;
            for (std::int32_t i = 0; i < num_output; ++i) {
                Hh += static_cast<int>(whc[i]) * static_cast<int>(hs[i]);
            }
            const float H = bias_c[q] + static_cast<float>(Hx) * (descale_x * descale_xc) +
                            static_cast<float>(Hh) * (descale_h * descale_hc);
            gates[q] = std::tanh(H);
        }

        float* output_data = top_blob.row(ti);
#pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t q = 0; q < num_output; ++q) {
            hidden_state[q] = gates[q];
            output_data[q] = gates[q];
        }
    }
    return 0;
}

MLAYER_RNN_TEMPLATE
std::int32_t RNN<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate>::forward(const MatT& bottom_blob,
                                                                                 MatT& top_blob,
                                                                                 const OptionT& opt) const {
    const std::int32_t T = bottom_blob.h;
    const std::int32_t num_directions = this->direction == 2 ? 2 : 1;

    MatT hidden;
    hidden.create(this->num_output, 4u, opt.workspace_allocator);
    if (hidden.empty()) {
        return -100;
    }
    hidden.fill(0.0F);

    top_blob.create(this->num_output * num_directions, T, 4u, opt.blob_allocator);
    if (top_blob.empty()) {
        return -100;
    }

    if (this->direction == 0 || this->direction == 1) {
        const std::int32_t reverse = this->direction == 1 ? 1 : 0;
        if (this->int8_scale_term != 0) {
            return this->run_int8(bottom_blob, top_blob, reverse, this->weight_xc_data.channel(0),
                                  this->weight_xc_data_int8_scales.row(0), this->bias_c_data.channel(0),
                                  this->weight_hc_data.channel(0), this->weight_hc_data_int8_scales.row(0), hidden,
                                  opt);
        }
        return this->run(bottom_blob, top_blob, reverse, this->weight_xc_data.channel(0),
                         this->bias_c_data.channel(0), this->weight_hc_data.channel(0), hidden, opt);
    }

    MatT forward_out;
    MatT reverse_out;
    forward_out.create(this->num_output, T, 4u, opt.workspace_allocator);
    reverse_out.create(this->num_output, T, 4u, opt.workspace_allocator);
    if (forward_out.empty() || reverse_out.empty()) {
        return -100;
    }
    std::int32_t ret = 0;
    if (this->int8_scale_term != 0) {
        ret = this->run_int8(bottom_blob, forward_out, 0, this->weight_xc_data.channel(0),
                             this->weight_xc_data_int8_scales.row(0), this->bias_c_data.channel(0),
                             this->weight_hc_data.channel(0), this->weight_hc_data_int8_scales.row(0), hidden, opt);
    } else {
        ret = this->run(bottom_blob, forward_out, 0, this->weight_xc_data.channel(0), this->bias_c_data.channel(0),
                        this->weight_hc_data.channel(0), hidden, opt);
    }
    if (ret != 0) {
        return ret;
    }
    hidden.fill(0.0F);
    if (this->int8_scale_term != 0) {
        ret = this->run_int8(bottom_blob, reverse_out, 1, this->weight_xc_data.channel(1),
                             this->weight_xc_data_int8_scales.row(1), this->bias_c_data.channel(1),
                             this->weight_hc_data.channel(1), this->weight_hc_data_int8_scales.row(1), hidden, opt);
    } else {
        ret = this->run(bottom_blob, reverse_out, 1, this->weight_xc_data.channel(1), this->bias_c_data.channel(1),
                        this->weight_hc_data.channel(1), hidden, opt);
    }
    if (ret != 0) {
        return ret;
    }
    for (std::int32_t i = 0; i < T; ++i) {
        const float* pf = forward_out.row(i);
        const float* pr = reverse_out.row(i);
        float* ptr = top_blob.row(i);
        for (std::int32_t q = 0; q < this->num_output; ++q) {
            ptr[q] = pf[q];
        }
        for (std::int32_t q = 0; q < this->num_output; ++q) {
            ptr[this->num_output + q] = pr[q];
        }
    }
    return 0;
}

}  // namespace mlayer

#endif  // MLAYER_RNN_HPP
