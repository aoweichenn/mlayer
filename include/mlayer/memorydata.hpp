// Copyright 2017 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_MEMORYDATA_HPP
#define MLAYER_MEMORYDATA_HPP

#include <cstdint>
#include <vector>

namespace mlayer {

template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, typename ModelBinT>
class MemoryData : public LayerT {
public:
    MemoryData() {
        this->one_blob_only = false;
        this->support_inplace = false;
        this->w = 0;
        this->h = 0;
        this->d = 0;
        this->c = 0;
        this->load_type = 1;
    }

    std::int32_t load_param(const ParamDictT& pd) override {
        this->w = pd.get(0, 0);
        this->h = pd.get(1, 0);
        this->d = pd.get(11, 0);
        this->c = pd.get(2, 0);
        this->load_type = pd.get(21, 1);
        return 0;
    }

    std::int32_t load_model(const ModelBinT& mb) override {
        if (this->d != 0) {
            this->data = mb.load(this->w, this->h, this->d, this->c, this->load_type);
        } else if (this->c != 0) {
            this->data = mb.load(this->w, this->h, this->c, this->load_type);
        } else if (this->h != 0) {
            this->data = mb.load(this->w, this->h, this->load_type);
        } else if (this->w != 0) {
            this->data = mb.load(this->w, this->load_type);
        } else {
            this->data.create(1);
        }
        return this->data.empty() ? -100 : 0;
    }

    std::int32_t forward(const std::vector<MatT>&, std::vector<MatT>& top_blobs,
                         const OptionT& opt) const override {
        MatT& top = top_blobs[0];
        top = this->data.clone(opt.blob_allocator);
        return top.empty() ? -100 : 0;
    }

public:
    std::int32_t w;
    std::int32_t h;
    std::int32_t d;
    std::int32_t c;
    std::int32_t load_type;
    MatT data;
};

}  // namespace mlayer

#endif  // MLAYER_MEMORYDATA_HPP
