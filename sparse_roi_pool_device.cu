#include <cassert>
#include <stdio.h>
#include <cuda_runtime.h>
#include "sparse_roi_pool_device.cuh"


void cudaSparseRoiPooling(const int *d_in_loc, const float *d_in_feats,
    const int *d_out_loc, const float *d_out_feats, int n, int c,
    int h, int w, std::vector<RoiBox> roi_boxes, int p, int q,
    Implementation type) {

}
