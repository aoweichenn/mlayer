// Copyright 2026 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_SHORTCONV_HPP
#define MLAYER_SHORTCONV_HPP

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace mlayer {

// Causal depthwise short convolution with a rolling state, as used by the
// Qwen3.5 gated-delta-rule blocks.
template<typename LayerT, typename MatT, typename OptionT>
class ShortConv : public LayerT {
public:
    ShortConv() {
        this->one_blob_only = false;
        this->support_inplace = false;
    }

    std::int32_t forward(const std::vector<MatT>& bottom_blobs, std::vector<MatT>& top_blobs,
                         const OptionT& opt) const override {
        if (bottom_blobs.size() < 2 || top_blobs.empty()) {
            return -100;
        }

        const MatT& weight_mat = bottom_blobs[0];
        const MatT& mixed_qkv = bottom_blobs[1];
        const MatT& conv_state = bottom_blobs.size() > 2 ? bottom_blobs[2] : MatT();

        const std::int32_t groups = weight_mat.c;
        const std::int32_t kernel_size = weight_mat.w;
        const bool channel_major =
            mixed_qkv.dims != 1 && mixed_qkv.h == groups && mixed_qkv.w != groups;
        const std::int32_t seq_len =
            mixed_qkv.dims == 1 ? 1 : channel_major ? mixed_qkv.w : mixed_qkv.h;

        if (kernel_size <= 0 || groups <= 0 || seq_len <= 0 || weight_mat.dims != 3 || weight_mat.h != 1 ||
            (mixed_qkv.dims != 1 && mixed_qkv.dims != 2 && mixed_qkv.dims != 3) ||
            (mixed_qkv.dims == 1 && mixed_qkv.w != groups) ||
            (mixed_qkv.dims != 1 && mixed_qkv.w != groups &&
             (mixed_qkv.h != groups || mixed_qkv.w <= 0)) ||
            (mixed_qkv.dims == 3 && mixed_qkv.c != 1)) {
            return -100;
        }

        const std::int32_t prefix_len = kernel_size - 1;
        const std::int32_t total_len = prefix_len + seq_len;

        bool valid_state = conv_state.empty();
        if (!valid_state) {
            if (conv_state.dims == 2 || (conv_state.dims == 3 && conv_state.c == 1)) {
                if (conv_state.w == kernel_size && conv_state.h == groups) {
                    valid_state = true;
                } else if (conv_state.w == prefix_len && conv_state.h == groups) {
                    valid_state = true;
                } else if (conv_state.w == groups &&
                           (conv_state.h == kernel_size || conv_state.h == prefix_len)) {
                    valid_state = true;
                }
            }
        }
        if (!valid_state) {
            return -100;
        }

        MatT stated_mixed_qkv;
        stated_mixed_qkv.create(groups, total_len, 4u, opt.workspace_allocator);
        if (stated_mixed_qkv.empty()) {
            return -100;
        }

        if (conv_state.empty()) {
            std::memset(stated_mixed_qkv.data, 0,
                        static_cast<std::size_t>(prefix_len) * static_cast<std::size_t>(groups) * sizeof(float));
        } else if (conv_state.h == groups) {
            const std::int32_t offset = conv_state.w - prefix_len;
            for (std::int32_t t = 0; t < prefix_len; ++t) {
                float* row = stated_mixed_qkv.row(t);
                for (std::int32_t g = 0; g < groups; ++g) {
                    row[g] = conv_state.row(g)[offset + t];
                }
            }
        } else {
            const std::int32_t offset = conv_state.h - prefix_len;
            for (std::int32_t t = 0; t < prefix_len; ++t) {
                float* row = stated_mixed_qkv.row(t);
                const float* src_row = conv_state.row(offset + t);
                std::memcpy(row, src_row, static_cast<std::size_t>(groups) * sizeof(float));
            }
        }

        if (mixed_qkv.dims == 1) {
            std::memcpy(stated_mixed_qkv.row(prefix_len), mixed_qkv.data,
                        static_cast<std::size_t>(groups) * sizeof(float));
        } else if (channel_major) {
            const auto* mixed_data = static_cast<const float*>(mixed_qkv.data);
            for (std::int32_t g = 0; g < groups; ++g) {
                for (std::int32_t t = 0; t < seq_len; ++t) {
                    stated_mixed_qkv.row(prefix_len + t)[g] = mixed_data[g * seq_len + t];
                }
            }
        } else {
            std::memcpy(stated_mixed_qkv.row(prefix_len), mixed_qkv.data,
                        static_cast<std::size_t>(seq_len) * static_cast<std::size_t>(groups) * sizeof(float));
        }

        if (top_blobs.size() > 1) {
            MatT& last_conv_state = top_blobs[1];
            last_conv_state.create(kernel_size, groups, 4u, opt.blob_allocator);
            if (last_conv_state.empty()) {
                return -100;
            }
            for (std::int32_t g = 0; g < groups; ++g) {
                float* state_row = last_conv_state.row(g);
                for (std::int32_t k = 0; k < kernel_size; ++k) {
                    state_row[k] = stated_mixed_qkv.row(total_len - kernel_size + k)[g];
                }
            }
        }

        MatT& top_blob = top_blobs[0];
        top_blob.create(channel_major ? seq_len : groups, channel_major ? groups : seq_len, 4u,
                        opt.blob_allocator);
        if (top_blob.empty()) {
            return -100;
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t g = 0; g < groups; ++g) {
            const float* weight = weight_mat.channel(g);
            for (std::int32_t t = 0; t < seq_len; ++t) {
                const std::int32_t base = prefix_len + t;
                float sum = 0.0F;
                for (std::int32_t k = 0; k < kernel_size; ++k) {
                    sum += stated_mixed_qkv.row(base - (kernel_size - 1) + k)[g] * weight[k];
                }
                const float activated = sum * (1.0F / (1.0F + std::exp(-sum)));
                if (channel_major) {
                    top_blob.row(g)[t] = activated;
                } else {
                    top_blob.row(t)[g] = activated;
                }
            }
        }

        return 0;
    }
};

}  // namespace mlayer

#endif  // MLAYER_SHORTCONV_HPP
