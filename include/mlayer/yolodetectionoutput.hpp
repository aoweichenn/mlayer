// Copyright 2018 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_YOLODETECTIONOUTPUT_HPP
#define MLAYER_YOLODETECTIONOUTPUT_HPP

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace mlayer {

template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, bool Validate = false>
class YoloDetectionOutput : public LayerT {
public:
    struct BBoxRect {
        float xmin;
        float ymin;
        float xmax;
        float ymax;
        std::int32_t label;
    };

    YoloDetectionOutput() {
        this->one_blob_only = false;
        this->support_inplace = false;
    }

    std::int32_t load_param(const ParamDictT& pd) override {
        this->num_class = pd.get(0, 20);
        this->num_box = pd.get(1, 5);
        this->confidence_threshold = pd.get(2, 0.01F);
        this->nms_threshold = pd.get(3, 0.45F);
        this->biases = pd.get(4, MatT());

        if constexpr (Validate) {
            if (this->num_class <= 0 || this->num_box <= 0 || this->num_box > this->biases.w / 2) {
                return -1;
            }
        }

        if (pd.type(4) == 5 && !this->biases.empty()) {
            MatT converted(this->biases.w);
            if (converted.empty()) {
                return -100;
            }
            const auto* p = static_cast<const std::int32_t*>(this->biases);
            for (std::int32_t i = 0; i < this->biases.w; ++i) {
                converted[static_cast<std::size_t>(i)] = static_cast<float>(p[i]);
            }
            this->biases = converted;
        }
        return 0;
    }

    std::int32_t forward(const std::vector<MatT>& bottom_blobs, std::vector<MatT>& top_blobs,
                         const OptionT& opt) const override {
        std::vector<BBoxRect> all_bbox_rects;
        std::vector<float> all_bbox_scores;

        for (std::size_t b = 0; b < bottom_blobs.size(); ++b) {
            const MatT& blob = bottom_blobs[b];

            const std::int32_t w = blob.w;
            const std::int32_t h = blob.h;
            const std::int32_t channels = blob.c;
            const std::int32_t channels_per_box = channels / this->num_box;
            if (channels_per_box != 4 + 1 + this->num_class) {
                return -1;
            }

            std::vector<std::vector<BBoxRect>> all_box_bbox_rects(
                static_cast<std::size_t>(this->num_box));
            std::vector<std::vector<float>> all_box_bbox_scores(static_cast<std::size_t>(this->num_box));

            #pragma omp parallel for num_threads(opt.num_threads)
            for (std::int32_t pp = 0; pp < this->num_box; ++pp) {
                const std::int32_t p = pp * channels_per_box;
                const float bias_w = this->biases[static_cast<std::size_t>(pp) * 2];
                const float bias_h = this->biases[static_cast<std::size_t>(pp) * 2 + 1];

                const float* xptr = blob.channel(p);
                const float* yptr = blob.channel(p + 1);
                const float* wptr = blob.channel(p + 2);
                const float* hptr = blob.channel(p + 3);
                const float* box_score_ptr = blob.channel(p + 4);
                const MatT scores = blob.channel_range(p + 5, this->num_class);

                std::vector<float> probabilities(static_cast<std::size_t>(this->num_class));
                for (std::int32_t i = 0; i < h; ++i) {
                    for (std::int32_t j = 0; j < w; ++j) {
                        // Softmax over the class channels at this pixel.
                        float max_score = -FLT_MAX;
                        for (std::int32_t q = 0; q < this->num_class; ++q) {
                            max_score = std::max(max_score, scores.channel(q).row(i)[j]);
                        }
                        float sum = 0.0F;
                        for (std::int32_t q = 0; q < this->num_class; ++q) {
                            const float e = std::exp(scores.channel(q).row(i)[j] - max_score);
                            probabilities[static_cast<std::size_t>(q)] = e;
                            sum += e;
                        }

                        const float bbox_cx =
                            (static_cast<float>(j) + sigmoid(xptr[0])) / static_cast<float>(w);
                        const float bbox_cy =
                            (static_cast<float>(i) + sigmoid(yptr[0])) / static_cast<float>(h);
                        const float bbox_w = std::exp(wptr[0]) * bias_w / static_cast<float>(w);
                        const float bbox_h = std::exp(hptr[0]) * bias_h / static_cast<float>(h);

                        const float box_score = sigmoid(box_score_ptr[0]);

                        std::int32_t class_index = 0;
                        float class_score = 0.0F;
                        for (std::int32_t q = 0; q < this->num_class; ++q) {
                            const float score = probabilities[static_cast<std::size_t>(q)] / sum;
                            if (score > class_score) {
                                class_index = q;
                                class_score = score;
                            }
                        }

                        const float confidence = box_score * class_score;
                        if (confidence >= this->confidence_threshold) {
                            BBoxRect c = {bbox_cx - bbox_w * 0.5F, bbox_cy - bbox_h * 0.5F,
                                          bbox_cx + bbox_w * 0.5F, bbox_cy + bbox_h * 0.5F, class_index};
                            all_box_bbox_rects[static_cast<std::size_t>(pp)].push_back(c);
                            all_box_bbox_scores[static_cast<std::size_t>(pp)].push_back(confidence);
                        }

                        ++xptr;
                        ++yptr;
                        ++wptr;
                        ++hptr;
                        ++box_score_ptr;
                    }
                }
            }

            for (std::int32_t i = 0; i < this->num_box; ++i) {
                const auto& box_bbox_rects = all_box_bbox_rects[static_cast<std::size_t>(i)];
                const auto& box_bbox_scores = all_box_bbox_scores[static_cast<std::size_t>(i)];
                all_bbox_rects.insert(all_bbox_rects.end(), box_bbox_rects.begin(), box_bbox_rects.end());
                all_bbox_scores.insert(all_bbox_scores.end(), box_bbox_scores.begin(), box_bbox_scores.end());
            }
        }

        qsort_descent_inplace(all_bbox_rects, all_bbox_scores);

        std::vector<std::size_t> picked;
        nms_sorted_bboxes(all_bbox_rects, picked, this->nms_threshold);

        std::vector<BBoxRect> bbox_rects;
        std::vector<float> bbox_scores;
        bbox_rects.reserve(picked.size());
        bbox_scores.reserve(picked.size());
        for (const std::size_t index : picked) {
            bbox_rects.push_back(all_bbox_rects[index]);
            bbox_scores.push_back(all_bbox_scores[index]);
        }

        const std::int32_t num_detected = static_cast<std::int32_t>(bbox_rects.size());
        if (num_detected == 0) {
            return 0;
        }

        MatT& top_blob = top_blobs[0];
        top_blob.create(6, num_detected, 4u, opt.blob_allocator);
        if (top_blob.empty()) {
            return -100;
        }

        for (std::int32_t i = 0; i < num_detected; ++i) {
            const BBoxRect& r = bbox_rects[static_cast<std::size_t>(i)];
            float* outptr = top_blob.row(i);
            outptr[0] = static_cast<float>(r.label) + 1.0F;
            outptr[1] = bbox_scores[static_cast<std::size_t>(i)];
            outptr[2] = r.xmin;
            outptr[3] = r.ymin;
            outptr[4] = r.xmax;
            outptr[5] = r.ymax;
        }
        return 0;
    }

public:
    std::int32_t num_class = 20;
    std::int32_t num_box = 5;
    float confidence_threshold = 0.01F;
    float nms_threshold = 0.45F;
    MatT biases;

private:
    static float sigmoid(float x) { return 1.0F / (1.0F + std::exp(-x)); }

    static float intersection_area(const BBoxRect& a, const BBoxRect& b) {
        if (a.xmin > b.xmax || a.xmax < b.xmin || a.ymin > b.ymax || a.ymax < b.ymin) {
            return 0.0F;
        }
        const float inter_width = std::min(a.xmax, b.xmax) - std::max(a.xmin, b.xmin);
        const float inter_height = std::min(a.ymax, b.ymax) - std::max(a.ymin, b.ymin);
        return inter_width * inter_height;
    }

    static void qsort_descent_inplace(std::vector<BBoxRect>& datas, std::vector<float>& scores,
                                      std::int32_t left, std::int32_t right) {
        std::int32_t i = left;
        std::int32_t j = right;
        const float p = scores[static_cast<std::size_t>((left + right) / 2)];

        while (i <= j) {
            while (scores[static_cast<std::size_t>(i)] > p) {
                ++i;
            }
            while (scores[static_cast<std::size_t>(j)] < p) {
                --j;
            }
            if (i <= j) {
                std::swap(datas[static_cast<std::size_t>(i)], datas[static_cast<std::size_t>(j)]);
                std::swap(scores[static_cast<std::size_t>(i)], scores[static_cast<std::size_t>(j)]);
                ++i;
                --j;
            }
        }

        if (left < j) {
            qsort_descent_inplace(datas, scores, left, j);
        }
        if (i < right) {
            qsort_descent_inplace(datas, scores, i, right);
        }
    }

    static void qsort_descent_inplace(std::vector<BBoxRect>& datas, std::vector<float>& scores) {
        if (datas.empty() || scores.empty()) {
            return;
        }
        qsort_descent_inplace(datas, scores, 0, static_cast<std::int32_t>(scores.size() - 1));
    }

    static void nms_sorted_bboxes(const std::vector<BBoxRect>& bboxes, std::vector<std::size_t>& picked,
                                  float nms_threshold) {
        picked.clear();

        const std::size_t n = bboxes.size();
        std::vector<float> areas(n);
        for (std::size_t i = 0; i < n; ++i) {
            const BBoxRect& r = bboxes[i];
            areas[i] = (r.xmax - r.xmin) * (r.ymax - r.ymin);
        }

        for (std::size_t i = 0; i < n; ++i) {
            const BBoxRect& a = bboxes[i];
            bool keep = true;
            for (const std::size_t index : picked) {
                const BBoxRect& b = bboxes[index];
                const float inter_area = intersection_area(a, b);
                const float union_area = areas[i] + areas[index] - inter_area;
                if (inter_area / union_area > nms_threshold) {
                    keep = false;
                }
            }
            if (keep) {
                picked.push_back(i);
            }
        }
    }
};

}  // namespace mlayer

#endif  // MLAYER_YOLODETECTIONOUTPUT_HPP
