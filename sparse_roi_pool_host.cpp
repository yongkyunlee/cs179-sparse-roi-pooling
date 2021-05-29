#include <cassert>
#include <stdio.h>
#include "sparse_roi_pool_device.cuh"



void cpuSparseRoiPooling(const int *in_loc, const float *in_feats,
    const int *out_loc, const float *out_feats, int n, int c,
    int h, int w, std::vector<RoiBox> roi_boxes, int p, int q) {

}
