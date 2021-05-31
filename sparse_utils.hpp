int count_dense_nonzero(float *dense, int n_images, int n_channels,
                        int height, int width);
void dense_to_sparse(float *dense, int *loc, float *feats,
                     int n_images, int n_channels, int height, int width);
void sparse_to_dense(int *loc, float *feats, float *dense, int dim, int sparse_n,
                     int n_images, int n_boxes, int n_channels, int height, int width);
void print_sparse(int *loc, float *feats, int sparse_n);
bool is_dense_equal(float *dense1, float* dense2, int n_elem);
