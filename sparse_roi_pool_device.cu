#include <cassert>
#include <stdio.h>
#include <cuda_runtime.h>

#include "sparse_roi_pool_device.cuh"
#include "sparse_utils.hpp"
#include "linearprobing.h"

void cudaSparseRoiPooling(const int *d_in_loc, const float *d_in_feats,
    int *d_out_loc, float *d_out_feats, int sparse_n, int n, int c,
    int h, int w, std::vector<RoiBox> roi_boxes, int p, int q,
    Implementation type) {

    /*
    Iterate through all in_loc for sparse_n steps
    If it's in the right area, find its cell in the output
    If there's an entry, max it. Otherwise, just enter it.

    At the end, one thread reads through all entries and writes it to
    out_loc, out_feats
     */

}
