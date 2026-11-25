// Copyright 2025 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_ROTARYEMBED_HPP
#define MLAYER_ROTARYEMBED_HPP

#include <cstdint>
#include <vector>

namespace mlayer {

template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT>
class RotaryEmbed : public LayerT {
public:
    RotaryEmbed() { this->interleaved = 0; }

    std::int32_t load_param(const ParamDictT& pd) override {
        this->interleaved = pd.get(0, 0);
        return 0;
    }

    std::int32_t forward(const std::vector<MatT>& bottom_blobs, std::vector<MatT>& top_blobs,
                         const OptionT& opt) const override {
        const MatT& bottom_blob = bottom_blobs[0];
        const MatT& cos_cache = bottom_blobs[1];
        const MatT& sin_cache = bottom_blobs[2];

        const std::int32_t embed_dim = bottom_blob.w;
        const std::int32_t seqlen = bottom_blob.h;
        const std::int32_t num_heads = bottom_blob.c;

        MatT& top_blob = top_blobs[0];
        top_blob.create_like(bottom_blob, opt.blob_allocator);
        if (top_blob.empty()) {
            return -100;
        }

#pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t q = 0; q < num_heads; ++q) {
            const MatT head = bottom_blob.channel(q);
            MatT out_head = top_blob.channel(q);

            for (std::int32_t i = 0; i < seqlen; ++i) {
                if (this->interleaved != 0) {
                    const float* ptr = head.row(i);
                    const float* cos_ptr = cos_cache.row(i);
                    const float* sin_ptr = sin_cache.row(i);
                    float* outptr = out_head.row(i);

                    for (std::int32_t j = 0; j < embed_dim / 2; ++j) {
                        const float x0 = ptr[0];
                        const float x1 = ptr[1];
                        const float cos_val = *cos_ptr++;
                        const float sin_val = *sin_ptr++;
                        outptr[0] = x0 * cos_val - x1 * sin_val;
                        outptr[1] = x0 * sin_val + x1 * cos_val;
                        ptr += 2;
                        outptr += 2;
                    }
                } else {
                    const std::int32_t half = embed_dim / 2;
                    const std::int32_t cw = cos_cache.w == embed_dim ? half : 0;
                    const float* ptr0 = head.row(i);
                    const float* ptr1 = ptr0 + half;
                    const float* cos_ptr0 = cos_cache.row(i);
                    const float* sin_ptr0 = sin_cache.row(i);
                    const float* cos_ptr1 = cos_ptr0 + cw;
                    const float* sin_ptr1 = sin_ptr0 + cw;
                    float* outptr0 = out_head.row(i);
                    float* outptr1 = outptr0 + half;

                    for (std::int32_t j = 0; j < half; ++j) {
                        const float x0 = ptr0[j];
                        const float x1 = ptr1[j];
                        outptr0[j] = x0 * cos_ptr0[j] - x1 * sin_ptr0[j];
                        outptr1[j] = x0 * sin_ptr1[j] + x1 * cos_ptr1[j];
                    }
                }
            }
        }

        return 0;
    }

public:
    std::int32_t interleaved;
};

}  // namespace mlayer

#endif  // MLAYER_ROTARYEMBED_HPP
