#ifndef CUDA_TRANSPOSE_CUH
#define CUDA_TRANSPOSE_CUH

#include <vector>

enum Implementation { NAIVE, OPTIMAL };

struct RoiBox {
    int img_indx;
    int xmin;
    int ymin;
    int xmax;
    int ymax;
};

void cudaSparseRoiPooling(
    const int *d_in_loc, const float *d_in_feats,
    const int *d_out_loc, const float *d_out_feats, 
    int n,
    int c,
    int h,
    int w,
    std::vector<RoiBox> roi_boxes,
    int p,
    int q,
    Implementation type);

void cpuSparseRoiPooling(
    const int *in_loc, const float *in_feats,
    const int *out_loc, const float *out_feats, 
    int n,
    int c,
    int h,
    int w,
    std::vector<RoiBox> roi_boxes,
    int p,
    int q);

#endif
