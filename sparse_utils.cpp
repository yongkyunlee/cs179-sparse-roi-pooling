#include <iostream>
#include <stdio.h>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include "sparse_utils.hpp"

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
void dense_to_sparse(float *dense, int *loc, float *feats, int dim, int n_images,
                     int n_boxes, int n_channels, int height, int width) {
    int sparse_idx = 0;
    if (dim == 4) {
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
    } else if (dim == 5) {
        for (int i = 0; i < n_images; i++) {
            for (int b = 0; b < n_boxes; b++) {
                for (int c = 0; c < n_channels; c++) {
                    for (int h = 0; h < height; h++) {
                        for (int w = 0; w < width; w++) {
                            int dense_idx = i * n_boxes * n_channels * height * width + \
                                            b * n_channels * height * width + \
                                            c * height * width + h * width + w;
                            float dense_val = dense[dense_idx];
                            if (dense_val != 0) {
                                loc[5 * sparse_idx] = i;
                                loc[5 * sparse_idx + 1] = b;
                                loc[5 * sparse_idx + 2] = c;
                                loc[5 * sparse_idx + 3] = h;
                                loc[5 * sparse_idx + 4] = w;
                                feats[sparse_idx] = dense_val;
                                sparse_idx++;
                            }
                        }
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
void print_sparse(int *loc, float *feats, int sparse_n, int dim) {
    if (dim == 4) {
        cout << "(image index, channel, height, width): value" << endl;
        for (int i = 0; i < sparse_n; i++) {
            cout << "(" << loc[4 * i] << ", ";
            cout << loc[4 * i + 1] << ", ";
            cout << loc[4 * i + 2] << ", ";
            cout << loc[4 * i + 3] << "): " << feats[i] << endl;;
        }
    } else if (dim == 5) {
        cout << "(image index, box index, channel, height, width): value" << endl;
        for (int i = 0; i < sparse_n; i++) {
            cout << "(" << loc[5 * i] << ", ";
            cout << loc[5 * i + 1] << ", ";
            cout << loc[5 * i + 2] << ", ";
            cout << loc[5 * i + 3] << ", ";
            cout << loc[5 * i + 4] << "): " << feats[i] << endl;
        }
    }
    
}

/* This function checks whether two dense arrays are equal or not */
bool is_dense_equal(float *dense1, float* dense2, int n_elem) {
    for (int i = 0; i < n_elem; i++) {
        if (dense1[i] != dense2[i]) return false;
    }
    return true;
}

/* This function checks whether two sparse matrices (dim 5) are
 * equal or not. Assume it is used for the output of the roi pooling.
 */
bool is_sparse_equal(int *loc1, float *feats1, int size1,
                     int *loc2, float *feats2, int size2) {
    unordered_map<pool_key, float, PoolKeyHash> pool_map1, pool_map2;
    pool_key pool_val_key;
    float pool_val;
    bool is_equal = true;
    for (int i = 0; i < size1; i++) {
        pool_val = feats1[i];
        if (pool_val == 0) continue;
        pool_val_key = make_tuple(loc1[5 * i], loc1[5 * i + 1], loc1[5 * i + 2],
                                  loc1[5 * i + 3], loc1[5 * i + 4]);
        pool_map1[pool_val_key] = pool_val;
    }
    for (int i = 0; i < size2; i++) {
        pool_val = feats2[i];
        if (pool_val == 0) continue;
        pool_val_key = make_tuple(loc2[5 * i], loc2[5 * i + 1], loc2[5 * i + 2],
                                  loc2[5 * i + 3], loc2[5 * i + 4]);
        pool_map2[pool_val_key] = pool_val;
    }

    if (pool_map1.size() != pool_map2.size()) {
        printf("Pool sizes not equal: %d != %d\n",
            pool_map1.size(), pool_map2.size());
        is_equal = false;
    }
    for ( auto it = pool_map1.begin(); it != pool_map1.end(); ++it ) {
        pool_val_key = it->first;
        if (pool_map2.find(pool_val_key) == pool_map2.end()) {
            printf("Key in 1 not found in 2: (%d, %d, %d, %d, %d): %f\n",
                get<0>(pool_val_key),
                get<1>(pool_val_key),
                get<2>(pool_val_key),
                get<3>(pool_val_key),
                get<4>(pool_val_key),
                pool_map1[pool_val_key]);
            is_equal = false;
        }
        else if (pool_map1[pool_val_key] != pool_map2[pool_val_key]) {
            printf("Mismatched values for key: (%d, %d, %d, %d, %d): %f != %f\n",
                get<0>(pool_val_key),
                get<1>(pool_val_key),
                get<2>(pool_val_key),
                get<3>(pool_val_key),
                get<4>(pool_val_key),
                pool_map1[pool_val_key],
                pool_map2[pool_val_key]);
            is_equal = false;
        }
    }
    for ( auto it = pool_map2.begin(); it != pool_map2.end(); ++it ) {
        pool_val_key = it->first;
        if (pool_map1.find(pool_val_key) == pool_map1.end()) {
            printf("Key in 2 not found in 1: (%d, %d, %d, %d, %d): %f\n",
                get<0>(pool_val_key),
                get<1>(pool_val_key),
                get<2>(pool_val_key),
                get<3>(pool_val_key),
                get<4>(pool_val_key),
                pool_map2[pool_val_key]);
            is_equal = false;
        }
    }
    return is_equal;
}

/* This function generates a random sparse matrix for in_loc and in_feats
 * given the input shape (n_images, n_channels, height, width).
 * The values of the data are between min_val and max_val
 */
void generate_random_sparse(int *loc, float *feats, float min_val, float max_val, int sparse_n,
                            int n_images, int n_channels, int height, int width) {
    int sparse_idx = 0;
    unordered_set<sparse_key, SparseKeyHash> sparse_key_set;
    sparse_key sparse_rand_key;
    int img_idx, c, h, w;

    // random number generator for the data
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(min_val, max_val);
    
    while (sparse_idx < sparse_n) {
        img_idx = rand() % n_images;
        c = rand() % n_channels;
        h = rand() % height;
        w = rand() % width;
        sparse_rand_key = make_tuple(img_idx, c, h, w);
        // only add to sparse_key_set if the key does not exist in sparse_key_set
        if (sparse_key_set.find(sparse_rand_key) == sparse_key_set.end()) {
            sparse_key_set.insert(sparse_rand_key);
            loc[4 * sparse_idx] = img_idx;
            loc[4 * sparse_idx + 1] = c;
            loc[4 * sparse_idx + 2] = h;
            loc[4 * sparse_idx + 3] = w;
            feats[sparse_idx] = dis(gen);
            sparse_idx++;
        }
    }
}

/* This function generates num_roi_boxes random RoI boxes per image, given
 * (n_images, height, weight, num_roi_boxes, p, q)
 */
vector<RoiBox> generate_random_roi_boxes(int n_images, int height, int width,
    int num_roi_boxes, int p, int q) {
    vector<RoiBox> roi_boxes;

    for (int img_idx = 0; img_idx < n_images; img_idx++) {
        roi_boxes.push_back({img_idx, 0, 0, width - 1, height - 1});
        for (int i = 0; i < num_roi_boxes; i++) {
            int a = rand() % width, b = rand() % width;
            int c = rand() % height, d = rand() % height;
            int xmin = min(a, b);
            int xmax = max(a, b);
            int ymin = min(c, d);
            int ymax = max(c, d);

            if ((xmax - xmin <= q) || (ymax - ymin <= p)) {
                i--;
            } else {
                // img_idx, xmin, ymin, xmax, ymax
                roi_boxes.push_back({img_idx, xmin, ymin, xmax, ymax});
            }

        }
    }
    return roi_boxes;
}