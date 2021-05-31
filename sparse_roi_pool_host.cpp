#include <cassert>
#include <stdio.h>
#include <unordered_map>
#include <tuple>

#include "sparse_roi_pool_device.cuh"

typedef std::tuple<int, int, int, int> pool_key;

void cpuSparseRoiPooling(const int *in_loc, const float *in_feats,
    const int *out_loc, const float *out_feats, int sparse_n, int n, int c,
    int h, int w, std::vector<RoiBox> roi_boxes, int p, int q) {

    /* hashmap of { (img_idx, channel, wIdx, hIdx): value } to keep track of maximum
     * for the corresponding pool section */
    // std::unordered_map<pool_key, float> pool_map;

    for (int i = 0; i < roi_boxes.size(); i++) {
        int roi_width = roi_boxes[i].xmax - roi_boxes[i].xmin + 1,
            roi_height = roi_boxes[i].ymax - roi_boxes[i].ymin + 1;
        for (int j = 0; j < sparse_n; j++) {
            // if the roi box image index is different from the sparse image index
            if (in_loc[4 * j] != roi_boxes[i].img_indx) {
                continue;
            }
            // if the input location is not inside the roi box
            if (in_loc[4 * j + 2] < roi_boxes[i].ymin || in_loc[4 * j + 2] > roi_boxes[i].ymax ||
                    in_loc[4 * j + 3] < roi_boxes[i].xmin || in_loc[4 * j + 3] > roi_boxes[i].xmax) {
                continue;
            }

            // get pool section idx (0 <= poolIdxH < p && 0 <= poolIdxW < q)
            int poolIdxH = -1, hIdx = 0, hRem = roi_height % p;
            for (int k = 0; k < p; k++) {
                if (hRem > 0) {
                    hIdx += roi_height / p + 1;
                    hRem--;
                } else {
                    hIdx += roi_height / p;
                }
                if (in_loc[4 * j + 2] - roi_boxes[i].ymin <= hIdx - 1) {
                    poolIdxH = k;
                    break;
                }
            }
            int poolIdxW = -1, wIdx = 0, wRem = roi_width % q;
            for (int k = 0; k < q; k++) {
                if (wRem > 0) {
                    wIdx += roi_width / q + 1;
                    wRem--;
                } else {
                    wIdx += roi_width / q;
                }
                if (in_loc[4 * j + 3] - roi_boxes[i].xmin <= wIdx - 1) {
                    poolIdxW = k;
                    break;
                }
            }

            for (int k = 0; k < c; k++) {
                if (true) { }
            }
        }
    }
}