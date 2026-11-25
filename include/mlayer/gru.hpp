// Copyright 2017 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_GRU_HPP
#define MLAYER_GRU_HPP

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace mlayer {

// GRU over a (size x T) input sequence with gate order R, U, N and an optional
// bidirectional pass. int8 is not ported.
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

public:
    std::int32_t num_output;
    std::int32_t weight_data_size;
    std::int32_t direction;
    std::int32_t int8_scale_term;

    MatT weight_xc_data;
    MatT bias_c_data;
    MatT weight_hc_data;
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
    if (this->int8_scale_term != 0) {
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
        return this->run(bottom_blob, top_blob, this->direction == 1 ? 1 : 0, this->weight_xc_data.channel(0),
                         this->bias_c_data.channel(0), this->weight_hc_data.channel(0), hidden, opt);
    }

    MatT forward_out;
    MatT reverse_out;
    forward_out.create(this->num_output, T, 4u, opt.workspace_allocator);
    reverse_out.create(this->num_output, T, 4u, opt.workspace_allocator);
    if (forward_out.empty() || reverse_out.empty()) {
        return -100;
    }

    std::int32_t ret = this->run(bottom_blob, forward_out, 0, this->weight_xc_data.channel(0),
                                 this->bias_c_data.channel(0), this->weight_hc_data.channel(0), hidden, opt);
    if (ret != 0) {
        return ret;
    }
    hidden.fill(0.0F);
    ret = this->run(bottom_blob, reverse_out, 1, this->weight_xc_data.channel(1), this->bias_c_data.channel(1),
                    this->weight_hc_data.channel(1), hidden, opt);
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
