#include "test_utils.hpp"
#include "sparse_utils.hpp"
#include <iostream>

using namespace std;

float* init_test1_data(DataInfo *test_data_info);
float* init_test1_ans(PoolInfo *pool_info, std::vector<RoiBox> &roi_boxes);
float* init_test2_ans(PoolInfo *pool_info, std::vector<RoiBox> &roi_boxes);
void clean_up_test(float *data_dense, int *in_loc, float *in_feats,
                   int *ans_loc, float *ans_feats, int *out_loc, float *out_feats);

float* init_test1_data(DataInfo *test_data_info) {
    const int n_images = 1, n_channels = 1, height = 8, width = 8;
    *test_data_info = (DataInfo){.n_images = n_images, .n_channels = n_channels,
                                 .height = height, .width = width,
                                 .n_elems = n_images * n_channels * height * width};

    const int n_elems = n_images * n_channels * height * width;
    const float data[n_images][n_channels][height][width] =
        {{{{0.88, 0.44,    0, 0.16, 0.37,    0, 0.96, 0.27},
           {   0, 0.45, 0.57, 0.16, 0.63, 0.29,    0,    0},
           {0.66,    0, 0.82, 0.64, 0.54,    0, 0.59, 0.26},
           {0.85, 0.34, 0.76, 0.84, 0.29, 0.75, 0.62, 0.25},
           {0.32, 0.74, 0.21, 0.39, 0.34, 0.03, 0.33, 0.48},
           {0.20, 0.14, 0.16, 0.13, 0.73, 0.65, 0.96, 0.32},
           {0.19, 0.69, 0.09, 0.86, 0.88, 0.07, 0.01, 0.48},
           {0.83, 0.24, 0.97, 0.04, 0.24, 0.35, 0.50, 0.91}}}};
    float *data_dense = new float[n_elems];
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
    return data_dense;
}

float* init_test1_ans(PoolInfo *pool_info, vector<RoiBox> &roi_boxes) {
    roi_boxes.push_back({0, 0, 0, 2, 2});
    const int n_images = 1, n_channels = 1, n_boxes = roi_boxes.size(), p = 2, q = 2,
              out_size = n_images * n_channels * n_boxes * p * q;
    *pool_info = (PoolInfo){.n_images = n_images, .n_channels = n_channels,
                            .n_boxes = n_boxes, .p = p, .q = q,
                            .out_size = out_size};
    
    float ans[n_images][n_boxes][n_channels][p][q] = 
        {{{{{0.88, 0.57},
            {0.66, 0.82}}}}};
    float *ans_dense = new float[out_size];
    for (int i = 0; i < n_images; i++) {
        for (unsigned b = 0; b < roi_boxes.size(); b++) {
            for (int c = 0; c < n_channels; c++) {
                for (int h = 0; h < p; h++) {
                    for (int w = 0; w < q; w++) {
                        int dense_idx = i * roi_boxes.size() * n_channels * p * q + \
                                        b * n_channels * p * q + c * p * q + \
                                        h * q + w;
                        ans_dense[dense_idx] = ans[i][b][c][h][w];
                    }
                }
            }
        }
    }
    return ans_dense;
}

void run_mini_test1() {
    // set up data for testing
    DataInfo test_data_info;
    float *data_dense = init_test1_data(&test_data_info);
    const int n_images = test_data_info.n_images, n_channels = test_data_info.n_channels,
              width = test_data_info.width, height = test_data_info.height;

    // convert dense matrix to sparse matrix
    int sparse_n = count_dense_nonzero(data_dense, n_images, n_channels, 
                                       height, width);
    int *in_loc = new int[sparse_n * 4];
    float *in_feats = new float[sparse_n];
    dense_to_sparse(data_dense, in_loc, in_feats, 4, n_images, -1,
                    n_channels, height, width);
    
    // set up pooling and the correct answer
    PoolInfo test_pool_info;
    vector<RoiBox> roi_boxes;
    float *ans_dense = init_test1_ans(&test_pool_info, roi_boxes);
    const int p = test_pool_info.p, q = test_pool_info.q,
                  out_size = test_pool_info.out_size;
    
    int *ans_loc = new int[out_size * 5];
    float *ans_feats = new float[out_size];
    dense_to_sparse(ans_dense, ans_loc, ans_feats, 5, n_images,
                    roi_boxes.size(), n_channels, p, q);
    int *out_loc = new int[out_size * 5];
    float *out_feats = new float[out_size];

    cpuSparseRoiPooling(in_loc, in_feats, out_loc, out_feats, sparse_n,
                n_images, n_channels, height, width, roi_boxes, p, q);
    bool correct = is_sparse_equal(ans_loc, ans_feats, out_size, out_loc, out_feats, out_size);
    cout << "===== Test1 Result =====" << endl;
    cout << "Is answer correct: " << correct << endl;

    clean_up_test(data_dense, in_loc, in_feats, ans_loc, ans_feats, out_loc, out_feats);
}

float* init_test2_ans(PoolInfo *pool_info, vector<RoiBox> &roi_boxes) {
    roi_boxes.push_back({0, 0, 3, 6, 7});
    roi_boxes.push_back({0, 5, 0, 7, 2});
    const int n_images = 1, n_channels = 1, n_boxes = roi_boxes.size(),
              p = 3, q = 3, out_size = n_images * n_channels * n_boxes * p * q;
    *pool_info = (PoolInfo){.n_images = n_images, .n_channels = n_channels,
                            .n_boxes = n_boxes, .p = p, .q = q,
                            .out_size = out_size};
    
    const float ans[n_images][n_boxes][n_channels][p][q] = 
        {{{{{0.85, 0.84, 0.75},
           {0.69, 0.88, 0.96},
           {0.97, 0.24, 0.50}}},
          {{{0, 0.96, 0.27},
           {0.29, 0, 0},
           {0, 0.59, 0.26}}}}};  
    float *ans_dense = new float[out_size];
    for (int i = 0; i < n_images; i++) {
        for (unsigned int b = 0; b < roi_boxes.size(); b++) {
            for (int c = 0; c < n_channels; c++) {
                for (int h = 0; h < p; h++) {
                    for (int w = 0; w < q; w++) {
                        int dense_idx = i * roi_boxes.size() * n_channels * p * q + \
                                        b * n_channels * p * q + c * p * q + \
                                        h * q + w;
                        ans_dense[dense_idx] = ans[i][b][c][h][w];
                    }
                }
            }
        }
    }
    return ans_dense;
}

void run_mini_test2() {
    // set up data for testing
    DataInfo test_data_info;
    float *data_dense = init_test1_data(&test_data_info);
    const int n_images = test_data_info.n_images, n_channels = test_data_info.n_channels,
              width = test_data_info.width, height = test_data_info.height;

    // convert dense matrix to sparse matrix
    int sparse_n = count_dense_nonzero(data_dense, n_images, n_channels, 
                                       height, width);
    int *in_loc = new int[sparse_n * 4];
    float *in_feats = new float[sparse_n];
    dense_to_sparse(data_dense, in_loc, in_feats, 4, n_images, -1,
                    n_channels, height, width);
    
    // set up pooling and the correct answer
    PoolInfo test_pool_info;
    vector<RoiBox> roi_boxes;
    float *ans_dense = init_test2_ans(&test_pool_info, roi_boxes);
    const int p = test_pool_info.p, q = test_pool_info.q,
              out_size = test_pool_info.out_size;
    
    int *ans_loc = new int[out_size * 5];
    float *ans_feats = new float[out_size];
    dense_to_sparse(ans_dense, ans_loc, ans_feats, 5, n_images,
                    roi_boxes.size(), n_channels, p, q);
    int *out_loc = new int[out_size * 5];
    float *out_feats = new float[out_size];
    // print_sparse(ans_loc, ans_feats, out_size, 5);

    cpuSparseRoiPooling(in_loc, in_feats, out_loc, out_feats, sparse_n,
                n_images, n_channels, height, width, roi_boxes, p, q);
    // print_sparse(out_loc, out_feats, out_size, 5);
    bool correct = is_sparse_equal(ans_loc, ans_feats, out_size, out_loc, out_feats, out_size);
    cout << "===== Test2 Result =====" << endl;
    cout << "Is answer correct: " << correct << endl;

    clean_up_test(data_dense, in_loc, in_feats, ans_loc, ans_feats, out_loc, out_feats);
}

void clean_up_test(float *data_dense, int *in_loc, float *in_feats,
                   int *ans_loc, float *ans_feats, int *out_loc, float *out_feats) {
    delete [] data_dense;
    delete [] in_loc;
    delete [] in_feats;
    delete [] ans_loc;
    delete [] ans_feats;
    delete [] out_loc;
    delete [] out_feats;
}