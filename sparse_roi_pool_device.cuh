#ifndef CUDA_TRANSPOSE_CUH
#define CUDA_TRANSPOSE_CUH

#include <vector>

enum Implementation { CPU, GPU };

struct RoiBox {
    int img_indx;
    int xmin;
    int ymin;
    int xmax;
    int ymax;
};
struct SubRoiBox {
    int img_indx;
    int xmin;
    int ymin;
    int xmax;
    int ymax;
};

void cudaSparseRoiPooling(
    int *d_in_loc, float *d_in_feats,
    int *d_out_loc, float *d_out_feats,
    int sparse_n,
    int n,
    int c,
    int h,
    int w,
    RoiBox *d_roi_boxes,
    int b,
    int p,
    int q,
    Implementation type);

void cpuSparseRoiPooling(
    const int *in_loc, const float *in_feats,
    int *out_loc, float *out_feats, 
    int sparse_n,
    int n,
    int c,
    int h,
    int w,
    std::vector<RoiBox> roi_boxes,
    int p,
    int q);

#endif
