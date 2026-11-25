// Copyright 2017 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_LSTM_HPP
#define MLAYER_LSTM_HPP

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace mlayer {

// LSTM over a (size x T) input sequence: gate order I, F, O, G with
// weight_xc (size x 4*hidden), weight_hc (num_output x 4*hidden), an optional
// hidden-to-output projection and bidirectional support. int8 is not ported.
template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, typename ModelBinT,
         bool Validate = false>
class LSTM : public LayerT {
public:
    LSTM();

    std::int32_t load_param(const ParamDictT& pd) override;
    std::int32_t load_model(const ModelBinT& mb) override;
    std::int32_t forward(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const override;

private:
    std::int32_t run(const MatT& bottom_blob, MatT& top_blob, std::int32_t reverse, const MatT& weight_xc,
                     const MatT& bias_c, const MatT& weight_hc, const MatT& weight_hr, MatT& hidden,
                     MatT& cell, const OptionT& opt) const;

public:
    std::int32_t num_output;
    std::int32_t weight_data_size;
    std::int32_t direction;
    std::int32_t hidden_size;
    std::int32_t int8_scale_term;

    MatT weight_xc_data;
    MatT bias_c_data;
    MatT weight_hc_data;
    MatT weight_hr_data;
};

#define MLAYER_LSTM_TEMPLATE \
    template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, typename ModelBinT, bool Validate>

MLAYER_LSTM_TEMPLATE
LSTM<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate>::LSTM() {
    this->one_blob_only = true;
    this->support_inplace = false;
    this->num_output = 0;
    this->weight_data_size = 0;
    this->direction = 0;
    this->hidden_size = 0;
    this->int8_scale_term = 0;
}

MLAYER_LSTM_TEMPLATE
std::int32_t LSTM<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate>::load_param(const ParamDictT& pd) {
    this->num_output = pd.get(0, 0);
    this->weight_data_size = pd.get(1, 0);
    this->direction = pd.get(2, 0);
    this->hidden_size = pd.get(3, this->num_output);
    this->int8_scale_term = pd.get(8, 0);
    if (this->num_output <= 0 || this->hidden_size <= 0 || this->direction < 0 || this->direction > 2) {
        return -1;
    }
    if (this->int8_scale_term != 0) {
        // int8 LSTM is not ported; fail loudly instead of misreading weights.
        return -1;
    }
    return 0;
}

MLAYER_LSTM_TEMPLATE
std::int32_t LSTM<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate>::load_model(const ModelBinT& mb) {
    const std::int32_t num_directions = this->direction == 2 ? 2 : 1;
    const std::int32_t size = this->weight_data_size / num_directions / this->hidden_size / 4;

    this->weight_xc_data = mb.load(size, this->hidden_size * 4, num_directions, 0);
    this->bias_c_data = mb.load(this->hidden_size, 4, num_directions, 0);
    this->weight_hc_data = mb.load(this->num_output, this->hidden_size * 4, num_directions, 0);
    if (this->num_output != this->hidden_size) {
        this->weight_hr_data = mb.load(this->hidden_size, this->num_output, num_directions, 0);
    }
    if (this->weight_xc_data.empty() || this->bias_c_data.empty() || this->weight_hc_data.empty() ||
        (this->num_output != this->hidden_size && this->weight_hr_data.empty())) {
        return -100;
    }
    return 0;
}

MLAYER_LSTM_TEMPLATE
std::int32_t LSTM<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate>::run(
    const MatT& bottom_blob, MatT& top_blob, std::int32_t reverse, const MatT& weight_xc, const MatT& bias_c,
    const MatT& weight_hc, const MatT& weight_hr, MatT& hidden_state, MatT& cell_state, const OptionT& opt) const {
    const std::int32_t size = bottom_blob.w;
    const std::int32_t T = bottom_blob.h;
    const std::int32_t num_output = top_blob.w;
    const std::int32_t hidden_size = cell_state.w;

    MatT gates;
    gates.create(4, hidden_size, 4u, opt.workspace_allocator);
    MatT tmp_hidden_state;
    if (num_output != hidden_size) {
        tmp_hidden_state.create(hidden_size, 4u, opt.workspace_allocator);
    }
    if (gates.empty() || (num_output != hidden_size && tmp_hidden_state.empty())) {
        return -100;
    }

    for (std::int32_t t = 0; t < T; ++t) {
        const std::int32_t ti = reverse != 0 ? T - 1 - t : t;
        const float* x = bottom_blob.row(ti);

#pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t q = 0; q < hidden_size; ++q) {
            const float* bias_I = bias_c.row(0);
            const float* bias_F = bias_c.row(1);
            const float* bias_O = bias_c.row(2);
            const float* bias_G = bias_c.row(3);
            float* gates_data = gates.row(q);

            const float* wxc_I = weight_xc.row(hidden_size * 0 + q);
            const float* wxc_F = weight_xc.row(hidden_size * 1 + q);
            const float* wxc_O = weight_xc.row(hidden_size * 2 + q);
            const float* wxc_G = weight_xc.row(hidden_size * 3 + q);
            const float* whc_I = weight_hc.row(hidden_size * 0 + q);
            const float* whc_F = weight_hc.row(hidden_size * 1 + q);
            const float* whc_O = weight_hc.row(hidden_size * 2 + q);
            const float* whc_G = weight_hc.row(hidden_size * 3 + q);

            float I = bias_I[q];
            float F = bias_F[q];
            float O = bias_O[q];
            float G = bias_G[q];
            for (std::int32_t i = 0; i < size; ++i) {
                const float xi = x[i];
                I += wxc_I[i] * xi;
                F += wxc_F[i] * xi;
                O += wxc_O[i] * xi;
                G += wxc_G[i] * xi;
            }
            for (std::int32_t i = 0; i < num_output; ++i) {
                const float h_cont = hidden_state[i];
                I += whc_I[i] * h_cont;
                F += whc_F[i] * h_cont;
                O += whc_O[i] * h_cont;
                G += whc_G[i] * h_cont;
            }
            gates_data[0] = I;
            gates_data[1] = F;
            gates_data[2] = O;
            gates_data[3] = G;
        }

        float* output_data = top_blob.row(ti);
#pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t q = 0; q < hidden_size; ++q) {
            const float* gates_data = gates.row(q);
            float I = gates_data[0];
            float F = gates_data[1];
            float O = gates_data[2];
            float G = gates_data[3];
            I = 1.0F / (1.0F + std::exp(-I));
            F = 1.0F / (1.0F + std::exp(-F));
            O = 1.0F / (1.0F + std::exp(-O));
            G = std::tanh(G);
            const float cell2 = F * cell_state[q] + I * G;
            const float H = O * std::tanh(cell2);
            cell_state[q] = cell2;
            if (num_output == hidden_size) {
                hidden_state[q] = H;
                output_data[q] = H;
            } else {
                tmp_hidden_state[q] = H;
            }
        }

        if (num_output != hidden_size) {
#pragma omp parallel for num_threads(opt.num_threads)
            for (std::int32_t q = 0; q < num_output; ++q) {
                const float* hr = weight_hr.row(q);
                float H = 0.0F;
                for (std::int32_t i = 0; i < hidden_size; ++i) {
                    H += tmp_hidden_state[i] * hr[i];
                }
                hidden_state[q] = H;
                output_data[q] = H;
            }
        }
    }
    return 0;
}

MLAYER_LSTM_TEMPLATE
std::int32_t LSTM<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate>::forward(const MatT& bottom_blob,
                                                                                  MatT& top_blob,
                                                                                  const OptionT& opt) const {
    const std::int32_t T = bottom_blob.h;
    const std::int32_t num_directions = this->direction == 2 ? 2 : 1;

    MatT hidden;
    hidden.create(this->num_output, 4u, opt.workspace_allocator);
    MatT cell;
    cell.create(this->hidden_size, 4u, opt.workspace_allocator);
    if (hidden.empty() || cell.empty()) {
        return -100;
    }
    hidden.fill(0.0F);
    cell.fill(0.0F);

    top_blob.create(this->num_output * num_directions, T, 4u, opt.blob_allocator);
    if (top_blob.empty()) {
        return -100;
    }

    if (this->direction == 0 || this->direction == 1) {
        return this->run(bottom_blob, top_blob, this->direction == 1 ? 1 : 0, this->weight_xc_data.channel(0),
                         this->bias_c_data.channel(0), this->weight_hc_data.channel(0),
                         this->num_output == this->hidden_size ? MatT() : this->weight_hr_data.channel(0), hidden,
                         cell, opt);
    }

    MatT forward_out;
    MatT reverse_out;
    forward_out.create(this->num_output, T, 4u, opt.workspace_allocator);
    reverse_out.create(this->num_output, T, 4u, opt.workspace_allocator);
    if (forward_out.empty() || reverse_out.empty()) {
        return -100;
    }

    std::int32_t ret = this->run(bottom_blob, forward_out, 0, this->weight_xc_data.channel(0),
                                 this->bias_c_data.channel(0), this->weight_hc_data.channel(0),
                                 this->num_output == this->hidden_size ? MatT() : this->weight_hr_data.channel(0),
                                 hidden, cell, opt);
    if (ret != 0) {
        return ret;
    }
    hidden.fill(0.0F);
    cell.fill(0.0F);
    ret = this->run(bottom_blob, reverse_out, 1, this->weight_xc_data.channel(1), this->bias_c_data.channel(1),
                    this->weight_hc_data.channel(1),
                    this->num_output == this->hidden_size ? MatT() : this->weight_hr_data.channel(1), hidden, cell,
                    opt);
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

#endif  // MLAYER_LSTM_HPP
