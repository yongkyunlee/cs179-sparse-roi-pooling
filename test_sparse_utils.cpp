#include <iostream>
#include <unordered_map>
#include "sparse_utils.hpp"

using namespace std;

// code to test sparse util functions

int main() {
    int n_images = 1, n_channels = 1, height = 8, width = 8;
    float data[n_images][n_channels][height][width] =
        {{{{0.88, 0.44,    0, 0.16, 0.37,    0, 0.96, 0.27},
           {   0, 0.45, 0.57, 0.16, 0.63, 0.29,    0,    0},
           {0.66,    0, 0.82, 0.64, 0.54,    0, 0.59, 0.26},
           {0.85, 0.34, 0.76, 0.84, 0.29, 0.75, 0.62, 0.25},
           {   0, 0.74, 0.21, 0.39, 0.34, 0.03, 0.33, 0.48},
           {0.20, 0.14, 0.16, 0.13,    0, 0.65, 0.96,    0},
           {0.19, 0.69, 0.09, 0.86, 0.88, 0.07,    0, 0.48},
           {0.83,    0,  0.97, 0.04,    0, 0.35, 0.50, 0.91}}}};
    float *data_dense = new float[n_images * n_channels * height * width];
    for (int i = 0; i < n_images; i++) {
        for (int c = 0; c < n_channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    int dense_idx = i * n_channels * height * width + c * height * width +\
                                    h * width + w;
                    data_dense[dense_idx] = data[i][c][h][w];
                }
            }
        }
    }

    // should be 51
    int nonzero_cnt = count_dense_nonzero(data_dense, n_images, n_channels, 
                                          height, width);
    cout << "Number of nonzero elements: " << nonzero_cnt << endl;

    // initialize loc and feats
    int *loc = new int[nonzero_cnt * 4];
    float *feats = new float[nonzero_cnt];

    dense_to_sparse(data_dense, loc, feats, 4, n_images, -1,
                    n_channels, height, width);
    // print_sparse(loc, feats, nonzero_cnt, 4);

    float *data_dense2 = new float[n_images * n_channels * height * width];
    sparse_to_dense(loc, feats, data_dense2, 4, nonzero_cnt,
                    n_images, -1, n_channels, height, width);

    bool is_equal = is_dense_equal(data_dense, data_dense2, n_images * n_channels * height * width);
    
    // should be true
    cout << "Are dense arrays equal: " << is_equal << endl;

    delete [] loc;
    delete [] feats;
    delete [] data_dense;
    delete [] data_dense2;


}