#include <iostream>

using namespace std;

/* This function counts the number of nonzero elements in a dense array.
 * Dense has size (n_images * n_channels * height * width)
 */
int count_dense_nonzero(float *dense, int n_images, int n_channels,
                        int height, int width) {
    int nonzero_cnt = 0;
    for (int i = 0; i < n_images; i++) {
        for (int c = 0; c < n_channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    if (dense[i * n_channels * height * width + c * height * width + h * width + w] != 0) {
                        nonzero_cnt++;
                    }
                }
            }
        }
    }
    return nonzero_cnt;
}

/* This function converts dense array (in 1d shape) to sparse matrix.
 * Since spare matrix is expressed with loc and feats, fill those two arrays
 * appropriately.
 * We assume loc and feats are initialized to the right size.
 */
void dense_to_sparse(float *dense, int *loc, float *feats,
                     int n_images, int n_channels, int height, int width) {
    int sparse_idx = 0;
    for (int i = 0; i < n_images; i++) {
        for (int c = 0; c < n_channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    float dense_val = dense[i * n_channels * height * width + c * height * width + h * width + w];
                    if (dense_val != 0) {
                        loc[4 * sparse_idx] = i;
                        loc[4 * sparse_idx + 1] = c;
                        loc[4 * sparse_idx + 2] = h;
                        loc[4 * sparse_idx + 3] = w;
                        feats[sparse_idx] = dense_val;
                        sparse_idx++;
                    }
                }
            }
        }
    }
}

/* This function converts a sparse matrix (composed of loc and feats) into
 * a dense array. Assume that dense array is initialized to the right size.
 * dim is either 4 or 5 (4 for in_loc and 5 for out_loc)
 * n_boxes is -1 if dim is 4
 */
void sparse_to_dense(int *loc, float *feats, float *dense, int dim, int sparse_n,
                     int n_images, int n_boxes, int n_channels, int height, int width) {
    for (int i = 0; i < sparse_n; i++) {
        if (dim == 4) {
            int img_idx = loc[4 * i], c = loc[4 * i + 1], h = loc[4 * i + 2],
                w = loc[4 * i + 3];
            int dense_idx = img_idx * n_channels * height * width + c * height * width +\
                            h * width + w;
            dense[dense_idx] = feats[i];
        } else if (dim == 5) {
            int img_idx = loc[4 * i], b = loc[4 * i + 1], c = loc[4 * i + 2],
                p = loc[4 * i + 3], q = loc[4 * i + 4];
            int dense_idx = img_idx * n_boxes * n_channels * height * width + \
                            b * n_channels * height * width + c * height * width + \
                            p * width + q;
            dense[dense_idx] = feats[i];
        }
    }
}

/* This function prints sparse matrix */
void print_sparse(int *loc, float *feats, int sparse_n) {
    cout << "(image index, channel, height, width): value" << endl;
    for (int i = 0; i < sparse_n; i++) {
        cout << "(" << loc[4 * i] << ", ";
        cout << loc[4 * i + 1] << ", ";
        cout << loc[4 * i + 2] << ", ";
        cout << loc[4 * i + 3] << "): " << feats[i] << endl;;
    }
}

/* This function checks whether two dense arrays are equal or not */
bool is_dense_equal(float *dense1, float* dense2, int n_elem) {
    for (int i = 0; i < n_elem; i++) {
        if (dense1[i] != dense2[i]) return false;
    }
    return true;
}

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

    dense_to_sparse(data_dense, loc, feats, n_images,
                    n_channels, height, width);
    // print_sparse(loc, feats, nonzero_cnt);

    float *data_dense2 = new float[n_images * n_channels * height * width];
    sparse_to_dense(loc, feats, data_dense2, 4, nonzero_cnt,
                    n_images, -1, n_channels, height, width);

    bool is_equal = is_dense_equal(data_dense, data_dense2, n_images * n_channels * height * width);
    
    // should be true
    cout << "Are dense arrays equal: " << is_equal << endl;
}
