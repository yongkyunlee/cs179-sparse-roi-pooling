#include <tuple>

static int base_mult = 1000003;
static int base_adder = 82520 + 5;

// img_idx, box_idx, channel, hIdx, wIdx
typedef std::tuple<int, int, int, int, int> pool_key;

class CustomHash
{
    public:
    // Implement a hash function
    std::size_t operator()(const pool_key& k) const
    {
        // Inspired by https://github.com/python/cpython/blob/3.7/Objects/tupleobject.c#L348
        std::size_t h1 = std::hash<int>{}(std::get<0>(k));
        std::size_t h2 = std::hash<int>{}(std::get<1>(k));
        std::size_t h3 = std::hash<int>{}(std::get<2>(k));
        std::size_t h4 = std::hash<int>{}(std::get<3>(k));
        std::size_t h5 = std::hash<int>{}(std::get<4>(k));
        return ((((((((h1 ^ h2) * base_mult)
            ^ h3) * (base_mult + base_adder))
            ^ h4) * (base_mult + base_adder * 2))
            ^ h5) * (base_mult + base_adder * 3));
    }
};

int count_dense_nonzero(float *dense, int n_images, int n_channels,
                        int height, int width);
void dense_to_sparse(float *dense, int *loc, float *feats, int dim,
                     int n_images, int n_boxes, int n_channels, int height, int width);
void sparse_to_dense(int *loc, float *feats, float *dense, int dim, int sparse_n,
                     int n_images, int n_boxes, int n_channels, int height, int width);
void print_sparse(int *loc, float *feats, int sparse_n, int dim);
bool is_dense_equal(float *dense1, float* dense2, int n_elem);
bool is_sparse_equal(int *loc1, float *feats1, int size1,
                     int *loc2, float *feats2, int size2);