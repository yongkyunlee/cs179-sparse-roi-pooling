#include <cassert>
#include <stdio.h>
#include <unordered_map>
#include <tuple>
#include <iostream>

#include "sparse_roi_pool_device.cuh"
#include "sparse_utils.hpp"

void cpuSparseRoiPooling(const int *in_loc, const float *in_feats,
    int *out_loc, float *out_feats, int sparse_n, int n, int c,
    int h, int w, std::vector<RoiBox> roi_boxes, int p, int q) {

    /* hashmap of { (img_idx, channel, wIdx, hIdx): value } to keep track of maximum
     * for the corresponding pool section */
    std::unordered_map<pool_key, float, CustomHash> pool_map;
    // std::cout << "stargin cpuSparseRoiPooling " << std::endl;
    for (unsigned int i = 0; i < roi_boxes.size(); i++) {
        int roi_width = roi_boxes[i].xmax - roi_boxes[i].xmin + 1,
            roi_height = roi_boxes[i].ymax - roi_boxes[i].ymin + 1;
        // std::cout << "processing roi_boxes i: " << i << std::endl;
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

            // std::cout << "roi_height: " << roi_height << " roi_width: " << roi_width << std::endl;

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

            // std::cout << "poolIdxH: " << poolIdxH << " poolIdxW: " << poolIdxW << std::endl;

            pool_key pool_val_key = std::make_tuple(in_loc[4 * j], i, in_loc[4 * j + 1],
                                                    poolIdxH, poolIdxW);
            if (pool_map.find(pool_val_key) == pool_map.end()) {
                // insert pool section value if the pool_val_key does not exist
                pool_map[pool_val_key] = in_feats[j];
            } else {
                // update pool section value if the new value is larger
                if (pool_map[pool_val_key] < in_feats[j]) {
                    pool_map[pool_val_key] = in_feats[j];
                }
            }
        }
    }
    int out_idx = 0;
    pool_key pool_val_key;
    for ( auto it = pool_map.begin(); it != pool_map.end(); ++it ) {
        pool_val_key = it->first;
        out_loc[5 * out_idx] = std::get<0>(pool_val_key);
        out_loc[5 * out_idx + 1] = std::get<1>(pool_val_key);
        out_loc[5 * out_idx + 2] = std::get<2>(pool_val_key);
        out_loc[5 * out_idx + 3] = std::get<3>(pool_val_key);
        out_loc[5 * out_idx + 4] = std::get<4>(pool_val_key);
        out_feats[out_idx] = it->second;
        out_idx++;
    }
}