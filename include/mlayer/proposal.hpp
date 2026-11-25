// Copyright 2017 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_PROPOSAL_HPP
#define MLAYER_PROPOSAL_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace mlayer {

// Faster-RCNN proposal layer: decode anchors, clip, filter and NMS.
template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, bool Validate = false>
class Proposal : public LayerT {
public:
    Proposal() {
        this->one_blob_only = false;
        this->support_inplace = false;

        this->ratios.create(3);
        this->ratios[0] = 0.5F;
        this->ratios[1] = 1.0F;
        this->ratios[2] = 2.0F;

        this->scales.create(3);
        this->scales[0] = 8.0F;
        this->scales[1] = 16.0F;
        this->scales[2] = 32.0F;
    }

    std::int32_t load_param(const ParamDictT& pd) override {
        this->feat_stride = pd.get(0, 16);
        this->base_size = pd.get(1, 16);
        this->pre_nms_topN = pd.get(2, 6000);
        this->after_nms_topN = pd.get(3, 300);
        this->nms_thresh = pd.get(4, 0.7F);
        this->min_size = pd.get(5, 16);

        this->anchors = generate_anchors(this->base_size, this->ratios, this->scales);
        if (this->anchors.empty()) {
            return -100;
        }
        return 0;
    }

    std::int32_t forward(const std::vector<MatT>& bottom_blobs, std::vector<MatT>& top_blobs,
                         const OptionT& opt) const override {
        if (bottom_blobs.size() < 3 || top_blobs.empty()) {
            return -1;
        }
        const MatT& score_blob = bottom_blobs[0];
        const MatT& bbox_blob = bottom_blobs[1];
        const MatT& im_info_blob = bottom_blobs[2];

        const std::int32_t w = score_blob.w;
        const std::int32_t h = score_blob.h;
        const std::int32_t num_anchors = this->anchors.h;

        MatT proposals;
        proposals.create(4, w * h, num_anchors);
        if (proposals.empty()) {
            return -100;
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t q = 0; q < num_anchors; ++q) {
            const float* bbox_xptr = bbox_blob.channel(q * 4);
            const float* bbox_yptr = bbox_blob.channel(q * 4 + 1);
            const float* bbox_wptr = bbox_blob.channel(q * 4 + 2);
            const float* bbox_hptr = bbox_blob.channel(q * 4 + 3);
            MatT pbs = proposals.channel(q);
            const float* anchor = this->anchors.row(q);

            float anchor_y = anchor[1];
            const float anchor_w = anchor[2] - anchor[0];
            const float anchor_h = anchor[3] - anchor[1];

            for (std::int32_t i = 0; i < h; ++i) {
                float anchor_x = anchor[0];
                for (std::int32_t j = 0; j < w; ++j) {
                    float* pb = pbs.row(i * w + j);
                    const float dx = bbox_xptr[j];
                    const float dy = bbox_yptr[j];
                    const float dw = bbox_wptr[j];
                    const float dh = bbox_hptr[j];

                    const float cx = anchor_x + anchor_w * 0.5F;
                    const float cy = anchor_y + anchor_h * 0.5F;
                    const float pb_cx = cx + anchor_w * dx;
                    const float pb_cy = cy + anchor_h * dy;
                    const float pb_w = anchor_w * std::exp(dw);
                    const float pb_h = anchor_h * std::exp(dh);

                    pb[0] = pb_cx - pb_w * 0.5F;
                    pb[1] = pb_cy - pb_h * 0.5F;
                    pb[2] = pb_cx + pb_w * 0.5F;
                    pb[3] = pb_cy + pb_h * 0.5F;

                    anchor_x += static_cast<float>(this->feat_stride);
                }
                bbox_xptr += w;
                bbox_yptr += w;
                bbox_wptr += w;
                bbox_hptr += w;
                anchor_y += static_cast<float>(this->feat_stride);
            }
        }

        const float im_w = im_info_blob[1];
        const float im_h = im_info_blob[0];

        #pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t q = 0; q < num_anchors; ++q) {
            MatT pbs = proposals.channel(q);
            for (std::int32_t i = 0; i < w * h; ++i) {
                float* pb = pbs.row(i);
                pb[0] = std::max(std::min(pb[0], im_w - 1.0F), 0.0F);
                pb[1] = std::max(std::min(pb[1], im_h - 1.0F), 0.0F);
                pb[2] = std::max(std::min(pb[2], im_w - 1.0F), 0.0F);
                pb[3] = std::max(std::min(pb[3], im_h - 1.0F), 0.0F);
            }
        }

        std::vector<Rect> proposal_boxes;
        std::vector<float> scores;
        const float im_scale = im_info_blob[2];
        const float min_boxsize = static_cast<float>(this->min_size) * im_scale;

        for (std::int32_t q = 0; q < num_anchors; ++q) {
            MatT pbs = proposals.channel(q);
            const float* scoreptr = score_blob.channel(q + num_anchors);
            for (std::int32_t i = 0; i < w * h; ++i) {
                float* pb = pbs.row(i);
                const float pb_w = pb[2] - pb[0] + 1.0F;
                const float pb_h = pb[3] - pb[1] + 1.0F;
                if (pb_w >= min_boxsize && pb_h >= min_boxsize) {
                    proposal_boxes.push_back(Rect{pb[0], pb[1], pb[2], pb[3]});
                    scores.push_back(scoreptr[i]);
                }
            }
        }

        qsort_descent_inplace(proposal_boxes, scores);

        if (this->pre_nms_topN > 0 && this->pre_nms_topN < static_cast<std::int32_t>(proposal_boxes.size())) {
            proposal_boxes.resize(static_cast<std::size_t>(this->pre_nms_topN));
            scores.resize(static_cast<std::size_t>(this->pre_nms_topN));
        }

        std::vector<std::size_t> picked;
        nms_sorted_bboxes(proposal_boxes, picked, this->nms_thresh);

        const std::int32_t picked_count =
            std::min(static_cast<std::int32_t>(picked.size()), this->after_nms_topN);

        MatT& roi_blob = top_blobs[0];
        roi_blob.create(4, 1, picked_count);
        if (roi_blob.empty()) {
            return -100;
        }
        for (std::int32_t i = 0; i < picked_count; ++i) {
            float* outptr = roi_blob.channel(i);
            outptr[0] = proposal_boxes[picked[static_cast<std::size_t>(i)]].x1;
            outptr[1] = proposal_boxes[picked[static_cast<std::size_t>(i)]].y1;
            outptr[2] = proposal_boxes[picked[static_cast<std::size_t>(i)]].x2;
            outptr[3] = proposal_boxes[picked[static_cast<std::size_t>(i)]].y2;
        }

        if (top_blobs.size() > 1) {
            MatT& roi_score_blob = top_blobs[1];
            roi_score_blob.create(1, 1, picked_count);
            if (roi_score_blob.empty()) {
                return -100;
            }
            for (std::int32_t i = 0; i < picked_count; ++i) {
                roi_score_blob.channel(i)[0] = scores[picked[static_cast<std::size_t>(i)]];
            }
        }

        return 0;
    }

public:
    std::int32_t feat_stride = 16;
    std::int32_t base_size = 16;
    std::int32_t pre_nms_topN = 6000;
    std::int32_t after_nms_topN = 300;
    float nms_thresh = 0.7F;
    std::int32_t min_size = 16;
    MatT ratios;
    MatT scales;
    MatT anchors;

private:
    struct Rect {
        float x1;
        float y1;
        float x2;
        float y2;
    };

    static MatT generate_anchors(std::int32_t base_size, const MatT& ratios, const MatT& scales) {
        const std::int32_t num_ratio = ratios.w;
        const std::int32_t num_scale = scales.w;
        MatT anchors;
        anchors.create(4, num_ratio * num_scale);
        if (anchors.empty()) {
            return MatT();
        }

        const float cx = static_cast<float>(base_size) * 0.5F;
        const float cy = static_cast<float>(base_size) * 0.5F;
        for (std::int32_t i = 0; i < num_ratio; ++i) {
            const float ar = ratios[i];
            const auto r_w = static_cast<std::int32_t>(std::round(static_cast<float>(base_size) / std::sqrt(ar)));
            const auto r_h = static_cast<std::int32_t>(std::round(static_cast<float>(r_w) * ar));
            for (std::int32_t j = 0; j < num_scale; ++j) {
                const float scale = scales[j];
                const float rs_w = static_cast<float>(r_w) * scale;
                const float rs_h = static_cast<float>(r_h) * scale;
                float* anchor = anchors.row(i * num_scale + j);
                anchor[0] = cx - rs_w * 0.5F;
                anchor[1] = cy - rs_h * 0.5F;
                anchor[2] = cx + rs_w * 0.5F;
                anchor[3] = cy + rs_h * 0.5F;
            }
        }
        return anchors;
    }

    static float intersection_area(const Rect& a, const Rect& b) {
        if (a.x1 > b.x2 || a.x2 < b.x1 || a.y1 > b.y2 || a.y2 < b.y1) {
            return 0.0F;
        }
        const float inter_width = std::min(a.x2, b.x2) - std::max(a.x1, b.x1);
        const float inter_height = std::min(a.y2, b.y2) - std::max(a.y1, b.y1);
        return inter_width * inter_height;
    }

    static void qsort_descent_inplace(std::vector<Rect>& datas, std::vector<float>& scores,
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

    static void qsort_descent_inplace(std::vector<Rect>& datas, std::vector<float>& scores) {
        if (datas.empty() || scores.empty()) {
            return;
        }
        qsort_descent_inplace(datas, scores, 0, static_cast<std::int32_t>(scores.size() - 1));
    }

    static void nms_sorted_bboxes(const std::vector<Rect>& bboxes, std::vector<std::size_t>& picked,
                                  float nms_threshold) {
        picked.clear();
        const std::size_t n = bboxes.size();
        std::vector<float> areas(n);
        for (std::size_t i = 0; i < n; ++i) {
            const Rect& r = bboxes[i];
            areas[i] = (r.x2 - r.x1) * (r.y2 - r.y1);
        }

        for (std::size_t i = 0; i < n; ++i) {
            const Rect& a = bboxes[i];
            bool keep = true;
            for (const std::size_t index : picked) {
                const Rect& b = bboxes[index];
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

#endif  // MLAYER_PROPOSAL_HPP
