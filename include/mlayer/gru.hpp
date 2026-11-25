// Copyright 2017 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_GRU_HPP
#define MLAYER_GRU_HPP

#include <cmath>
#include <cstddef>
#include <cstdint>

#include <mlayer/int8_quantize.hpp>

namespace mlayer {

// GRU over a (size x T) input sequence with gate order R, U, N and an optional
// bidirectional pass. int8 weights use ncnn's per-row weight scales with
// dynamically quantized input/hidden states.
template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, typename ModelBinT,
         bool Validate = false>
class GRU : public LayerT {
public:
    GRU();

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

#define MLAYER_GRU_TEMPLATE \
    template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, typename ModelBinT, bool Validate>

MLAYER_GRU_TEMPLATE
GRU<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate>::GRU() {
    this->one_blob_only = true;
    this->support_inplace = false;
    this->num_output = 0;
    this->weight_data_size = 0;
    this->direction = 0;
    this->int8_scale_term = 0;
}

MLAYER_GRU_TEMPLATE
std::int32_t GRU<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate>::load_param(const ParamDictT& pd) {
    this->num_output = pd.get(0, 0);
    this->weight_data_size = pd.get(1, 0);
    this->direction = pd.get(2, 0);
    this->int8_scale_term = pd.get(8, 0);
    if (this->num_output <= 0 || this->direction < 0 || this->direction > 2) {
        return -1;
    }
    return 0;
}

MLAYER_GRU_TEMPLATE
std::int32_t GRU<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate>::load_model(const ModelBinT& mb) {
    const std::int32_t num_directions = this->direction == 2 ? 2 : 1;
    const std::int32_t size = this->weight_data_size / num_directions / this->num_output / 3;

    this->weight_xc_data = mb.load(size, this->num_output * 3, num_directions, 0);
    this->bias_c_data = mb.load(this->num_output, 4, num_directions, 0);
    this->weight_hc_data = mb.load(this->num_output, this->num_output * 3, num_directions, 0);
    if (this->weight_xc_data.empty() || this->bias_c_data.empty() || this->weight_hc_data.empty()) {
        return -100;
    }
    if (this->int8_scale_term != 0) {
        this->weight_xc_data_int8_scales = mb.load(this->num_output * 3, num_directions, 1);
        this->weight_hc_data_int8_scales = mb.load(this->num_output * 3, num_directions, 1);
        if (this->weight_xc_data_int8_scales.empty() || this->weight_hc_data_int8_scales.empty()) {
            return -100;
        }
    }
    return 0;
}

MLAYER_GRU_TEMPLATE
std::int32_t GRU<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate>::run(
    const MatT& bottom_blob, MatT& top_blob, std::int32_t reverse, const MatT& weight_xc, const MatT& bias_c,
    const MatT& weight_hc, MatT& hidden_state, const OptionT& opt) const {
    const std::int32_t size = bottom_blob.w;
    const std::int32_t T = bottom_blob.h;
    const std::int32_t num_output = top_blob.w;

    MatT gates;
    gates.create(2, num_output, 4u, opt.workspace_allocator);
    if (gates.empty()) {
        return -100;
    }

    for (std::int32_t t = 0; t < T; ++t) {
        const std::int32_t ti = reverse != 0 ? T - 1 - t : t;
        const float* x = bottom_blob.row(ti);

#pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t q = 0; q < num_output; ++q) {
            float* gates_data = gates.row(q);
            const float* bias_R = bias_c.row(0);
            const float* bias_U = bias_c.row(1);
            const float* wxc_R = weight_xc.row(num_output * 0 + q);
            const float* wxc_U = weight_xc.row(num_output * 1 + q);
            const float* whc_R = weight_hc.row(num_output * 0 + q);
            const float* whc_U = weight_hc.row(num_output * 1 + q);

            float R = bias_R[q];
            float U = bias_U[q];
            for (std::int32_t i = 0; i < size; ++i) {
                const float xi = x[i];
                R += wxc_R[i] * xi;
                U += wxc_U[i] * xi;
            }
            for (std::int32_t i = 0; i < num_output; ++i) {
                const float h_cont = hidden_state[i];
                R += whc_R[i] * h_cont;
                U += whc_U[i] * h_cont;
            }
            R = 1.0F / (1.0F + std::exp(-R));
            U = 1.0F / (1.0F + std::exp(-U));

            const float* bias_WN = bias_c.row(2);
            const float* bias_BN = bias_c.row(3);
            const float* wxc_N = weight_xc.row(num_output * 2 + q);
            const float* whc_N = weight_hc.row(num_output * 2 + q);

            float N = bias_BN[q];
            for (std::int32_t i = 0; i < num_output; ++i) {
                N += whc_N[i] * hidden_state[i];
            }
            N = bias_WN[q] + R * N;
            for (std::int32_t i = 0; i < size; ++i) {
                N += wxc_N[i] * x[i];
            }
            N = std::tanh(N);

            gates_data[0] = U;
            gates_data[1] = N;
        }

        float* output_data = top_blob.row(ti);
#pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t q = 0; q < num_output; ++q) {
            const float* gates_data = gates.row(q);
            const float U = gates_data[0];
            const float N = gates_data[1];
            const float H = (1.0F - U) * N + U * hidden_state[q];
            hidden_state[q] = H;
            output_data[q] = H;
        }
    }
    return 0;
}

MLAYER_GRU_TEMPLATE
std::int32_t GRU<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate>::run_int8(
    const MatT& bottom_blob, MatT& top_blob, std::int32_t reverse, const MatT& weight_xc_int8,
    const float* weight_xc_scales, const MatT& bias_c, const MatT& weight_hc_int8,
    const float* weight_hc_scales, MatT& hidden_state, const OptionT& opt) const {
    const std::int32_t size = bottom_blob.w;
    const std::int32_t T = bottom_blob.h;
    const std::int32_t num_output = top_blob.w;

    MatT gates;
    gates.create(2, num_output, 4u, opt.workspace_allocator);
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
            float* gates_data = gates.row(q);
            const float* bias_R = bias_c.row(0);
            const float* bias_U = bias_c.row(1);
            const std::int8_t* wxc_R = weight_xc_int8.template row<std::int8_t>(num_output * 0 + q);
            const std::int8_t* wxc_U = weight_xc_int8.template row<std::int8_t>(num_output * 1 + q);
            const std::int8_t* whc_R = weight_hc_int8.template row<std::int8_t>(num_output * 0 + q);
            const std::int8_t* whc_U = weight_hc_int8.template row<std::int8_t>(num_output * 1 + q);
            const float descale_xc_R = 1.0F / weight_xc_scales[num_output * 0 + q];
            const float descale_xc_U = 1.0F / weight_xc_scales[num_output * 1 + q];
            const float descale_hc_R = 1.0F / weight_hc_scales[num_output * 0 + q];
            const float descale_hc_U = 1.0F / weight_hc_scales[num_output * 1 + q];

            int Rx = 0;
            int Ux = 0;
            for (std::int32_t i = 0; i < size; ++i) {
                const int xi = x[i];
                Rx += static_cast<int>(wxc_R[i]) * xi;
                Ux += static_cast<int>(wxc_U[i]) * xi;
            }
            int Rh = 0;
            int Uh = 0;
            for (std::int32_t i = 0; i < num_output; ++i) {
                const int h_cont = hs[i];
                Rh += static_cast<int>(whc_R[i]) * h_cont;
                Uh += static_cast<int>(whc_U[i]) * h_cont;
            }
            float R = bias_R[q] + static_cast<float>(Rx) * (descale_x * descale_xc_R) +
                      static_cast<float>(Rh) * (descale_h * descale_hc_R);
            float U = bias_U[q] + static_cast<float>(Ux) * (descale_x * descale_xc_U) +
                      static_cast<float>(Uh) * (descale_h * descale_hc_U);
            R = 1.0F / (1.0F + std::exp(-R));
            U = 1.0F / (1.0F + std::exp(-U));

            const float* bias_WN = bias_c.row(2);
            const float* bias_BN = bias_c.row(3);
            const std::int8_t* wxc_N = weight_xc_int8.template row<std::int8_t>(num_output * 2 + q);
            const std::int8_t* whc_N = weight_hc_int8.template row<std::int8_t>(num_output * 2 + q);
            const float descale_xc_N = 1.0F / weight_xc_scales[num_output * 2 + q];
            const float descale_hc_N = 1.0F / weight_hc_scales[num_output * 2 + q];

            int Nh = 0;
            for (std::int32_t i = 0; i < num_output; ++i) {
                Nh += static_cast<int>(whc_N[i]) * static_cast<int>(hs[i]);
            }
            int Nx = 0;
            for (std::int32_t i = 0; i < size; ++i) {
                Nx += static_cast<int>(wxc_N[i]) * static_cast<int>(x[i]);
            }
            float N = bias_BN[q] + static_cast<float>(Nh) * (descale_h * descale_hc_N);
            N = bias_WN[q] + R * N + static_cast<float>(Nx) * (descale_x * descale_xc_N);
            N = std::tanh(N);

            gates_data[0] = U;
            gates_data[1] = N;
        }

        float* output_data = top_blob.row(ti);
#pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t q = 0; q < num_output; ++q) {
            const float* gates_data = gates.row(q);
            const float U = gates_data[0];
            const float N = gates_data[1];
            const float H = (1.0F - U) * N + U * hidden_state[q];
            hidden_state[q] = H;
            output_data[q] = H;
        }
    }
    return 0;
}

MLAYER_GRU_TEMPLATE
std::int32_t GRU<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate>::forward(const MatT& bottom_blob,
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

#endif  // MLAYER_GRU_HPP
