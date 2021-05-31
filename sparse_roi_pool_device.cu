#include <cassert>
#include <stdio.h>
#include <cuda_runtime.h>

#include "cuda_header.cuh"
#include "sparse_roi_pool_device.cuh"
#include "sparse_utils.hpp"
#include "linearprobing.h"


CUDA_CALLABLE
bool in_roi_box(int j, int *in_loc, RoiBox roi_box) {
    // if the roi box image index is different from the sparse image index
    if (in_loc[4 * j] != roi_box.img_indx) {
        return false;
    }
    // if the input location is not inside the roi box
    if (in_loc[4 * j + 2] < roi_box.ymin || in_loc[4 * j + 2] > roi_box.ymax ||
            in_loc[4 * j + 3] < roi_box.xmin || in_loc[4 * j + 3] > roi_box.xmax) {
        return false;
    }

    return true;
}

__device__ uint32_t my_tuple_hash(d_pool_key k) {
    uint32_t h = ((((((((thrust::get<0>(k) ^ thrust::get<1>(k)) * base_mult)
        ^ thrust::get<2>(k)) * (base_mult + base_adder))
        ^ thrust::get<3>(k)) * (base_mult + base_adder * 2))
        ^ thrust::get<4>(k)) * (base_mult + base_adder * 3));
    return h & (kHashTableCapacity-1);
}

__device__ uint32_t my_alt_tuple_hash(d_pool_key k) {
    uint32_t h = ((((((((thrust::get<1>(k) ^ thrust::get<2>(k)) * base_mult)
        ^ thrust::get<3>(k)) * (base_mult + base_adder))
        ^ thrust::get<4>(k)) * (base_mult + base_adder * 2))
        ^ thrust::get<0>(k)) * (base_mult + base_adder * 3));
    return h & (kHashTableCapacity-1);
}

__device__ void my_lookup_key(KeyValue* hashtable, d_pool_key key, float* value_p) {

    d_pool_key kEmpty = thrust::make_tuple(
        0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff);
    uint32_t slot = my_tuple_hash(key);

    while (true)
    {
        if (hashtable[slot].key == key)
        {
            *value_p = hashtable[slot].value;
            return;
        }
        if (hashtable[slot].key == kEmpty)
        {
            *value_p = vEmpty;
            return;
        }
        slot = (slot + 1) & (kHashTableCapacity - 1);
    }
}

__device__ void my_insert_key_value(KeyValue* hashtable,
    d_pool_key key, float value)
{
    d_pool_key kEmpty = thrust::make_tuple(
        0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff);
    uint32_t slot = my_tuple_hash(key);
    uint32_t rotated_slot = my_alt_tuple_hash(key);
    uint32_t empty_start = thrust::get<0>(kEmpty);

    while (true)
    {
        // Highly unlikely to get two tuples hashing to same thing after
        // rotation

        // atomicCas only takes int 16, 32, 64
        // so we check the first section of tuple instead of the whole
        // tuple
        // uint32_t prev = atomicCAS(
        //     &thrust::get<0>(hashtable[slot].key),
        //     empty_start,
        //     rotated_slot);
        uint32_t prev = empty_start;
        if (prev == empty_start || prev == rotated_slot)
        {
            // hashtable[slot].key = key;
            // hashtable[slot].value = value;
            return;
        }

        slot = (slot + 1) & (kHashTableCapacity-1);
    }
}


#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
    bool abort = true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n",
            cudaGetErrorString(code), file, line);
        exit(code);
    }
}
__global__
void naiveSparseRoiPoolingKernel(int *in_loc, float *in_feats,
    int *out_loc, float *out_feats, int sparse_n, int n, int c,
    int h, int w, RoiBox roi_box, int p, int q, int image_idx,
    KeyValue* pHashTable) {

    int roi_width = roi_box.xmax - roi_box.xmin + 1,
        roi_height = roi_box.ymax - roi_box.ymin + 1;

    uint thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    while (thread_index < sparse_n) {
        int j = thread_index;

        if (!(in_roi_box(j, in_loc, roi_box))) {
            thread_index += blockDim.x * gridDim.x;
            continue;
        }

        // get pool section idx (0 <= poolIdxH < p && 0 <= poolIdxW < q)
        int poolIdxH = -1, hIdx = 0, hRem = roi_height % p;
        for (int k = 0; k < p; k++) {
            if (hRem > 0) {
                hIdx += roi_height / p + 1;
                hRem--;
            } else {
                hIdx += roi_height / p;
            }
            if (in_loc[4 * j + 2] - roi_box.ymin <= hIdx - 1) {
                poolIdxH = k;
                break;
            }
        }
        int poolIdxW = -1, wIdx = 0, wRem = roi_width % q;
        for (int k = 0; k < q; k++) {
            if (wRem > 0) {
                wIdx += roi_width / q + 1;
                wRem--;
            } else {
                wIdx += roi_width / q;
            }
            if (in_loc[4 * j + 3] - roi_box.xmin <= wIdx - 1) {
                poolIdxW = k;
                break;
            }
        }
        d_pool_key pool_val_key = thrust::make_tuple(
            in_loc[4 * j], image_idx, in_loc[4 * j + 1], poolIdxH, poolIdxW);

        float value;
        my_lookup_key(pHashTable, pool_val_key, &value);
        if ((value == vEmpty)  || (value < in_feats[j])) {
            my_insert_key_value(pHashTable, pool_val_key, in_feats[j]);
        }
        thread_index += blockDim.x * gridDim.x;
    }
}


void cudaSparseRoiPooling(const int *in_loc, const float *in_feats,
    int *out_loc, float *out_feats, int sparse_n, int n, int c,
    int h, int w, std::vector<RoiBox> roi_boxes, int p, int q,
    Implementation type) {

    int blocks = 512;
    int threadsPerBlock = 128;

    // Allocate device memory
    float *d_in_feats;
    float *d_out_feats;
    int *d_in_loc;
    int *d_out_loc;
    gpuErrChk(cudaMalloc(&d_in_feats, sparse_n * sizeof(float)));
    gpuErrChk(cudaMalloc(&d_out_feats, sparse_n * sizeof(float)));
    gpuErrChk(cudaMalloc(&d_in_loc, sparse_n * 4 * sizeof(int)));
    gpuErrChk(cudaMalloc(&d_out_loc, sparse_n * 5 * sizeof(int)));
    // Copy input to GPU
    gpuErrChk(cudaMemcpy(d_in_feats, in_feats, sparse_n * sizeof(float),
        cudaMemcpyHostToDevice));
    gpuErrChk(cudaMemcpy(d_in_loc, in_loc, sparse_n * 4 * sizeof(int),
        cudaMemcpyHostToDevice));


    /*
    Iterate through all in_loc for sparse_n steps
    If it's in the right area, find its cell in the output
    If there's an entry, max it. Otherwise, just enter it.

    At the end, one thread reads through all entries and writes it to
    out_loc, out_feats
     */
    KeyValue* pHashTable = create_hashtable();

    for (unsigned int i = 0; i < roi_boxes.size(); i++) {

        /* Call cudakernel here */
        naiveSparseRoiPoolingKernel<<<blocks, threadsPerBlock>>>(
            d_in_loc, d_in_feats, d_out_loc, d_out_feats, sparse_n, n, c,
            h, w, roi_boxes[i], p, q, i, pHashTable
        );
    }

    // TODO: create a kernel to do this for greater efficiency
    int out_idx = 0;
    d_pool_key pool_val_key;
    std::vector<KeyValue> kvs = iterate_hashtable(pHashTable);
    for (auto it = kvs.begin(); it != kvs.end(); ++it) {
        pool_val_key = it->key;
        d_out_loc[5 * out_idx] = thrust::get<0>(pool_val_key);
        d_out_loc[5 * out_idx + 1] = thrust::get<1>(pool_val_key);
        d_out_loc[5 * out_idx + 2] = thrust::get<2>(pool_val_key);
        d_out_loc[5 * out_idx + 3] = thrust::get<3>(pool_val_key);
        d_out_loc[5 * out_idx + 4] = thrust::get<4>(pool_val_key);
        d_out_feats[out_idx] = it->value;
        out_idx++;
    }

    gpuErrChk(cudaMemcpy(out_feats, d_out_feats,
        sparse_n * sizeof(float),
        cudaMemcpyDeviceToHost));
    gpuErrChk(cudaMemcpy(out_loc, d_out_loc,
        sparse_n * 5 * sizeof(int),
        cudaMemcpyDeviceToHost));

    // // Not necessary
    // // gpuErrChk(cudaMemset(d_out_feats, 0, sparse_n * sizeof(float)));
    // // gpuErrChk(cudaMemset(d_out_loc, 0, sparse_n * 5 * sizeof(int)));


    // Free device memory
    gpuErrChk(cudaFree(d_in_feats));
    gpuErrChk(cudaFree(d_out_feats));
    gpuErrChk(cudaFree(d_in_loc));
    gpuErrChk(cudaFree(d_out_loc));
    destroy_hashtable(pHashTable);
}
