#include <unordered_map>
#include "sparse_roi_pool_device.cuh"

typedef struct {
    int n_images;
    int n_channels;
    int height;
    int width;
    int n_elems;
} DataInfo;

typedef struct {
    int n_images;
    int n_channels;
    int n_boxes;
    int p;
    int q;
    int out_size;
} PoolInfo;

void run_mini_test1(Implementation mode);
void run_mini_test2(Implementation mode);