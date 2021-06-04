#include <cassert>
#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>

#include "cuda_header.cuh"
#include "sparse_roi_pool_device.cuh"
#include "sparse_utils.hpp"
#include "linearprobing.h"


CUDA_CALLABLE
bool in_box_bounds(int row, int col, int xmin, int ymin, int xmax, int ymax) {
    return (row < ymax && row >= ymin && col < xmax && col >= xmin);
}

CUDA_CALLABLE
bool in_roi_box(int j, int *in_loc, RoiBox roi_box) {
    // if the roi box image index is different from the sparse image index
    if (in_loc[4 * j] != roi_box.img_indx) {
        return false;
    }
    // if the input location is not inside the roi box
    if (!in_box_bounds(in_loc[4 * j + 2], in_loc[4 * j + 3], roi_box.xmin,
        roi_box.ymin, roi_box.xmax + 1, roi_box.ymax + 1)) {
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
            hashtable[slot].key = key;
            hashtable[slot].value = value;
            printf("Inserted key: (%d %d %d %d %d) with value %f at %d\n",
                thrust::get<0>(key), thrust::get<1>(key), thrust::get<2>(key),
                thrust::get<3>(key), thrust::get<4>(key), value, slot);
            return;
        }

        slot = (slot + 1) & (kHashTableCapacity-1);
    }
}

CUDA_CALLABLE
RoiBox get_roi_sub_box(RoiBox roi_box, int p, int q,
    int poolIndH, int poolIndW) {

    int roi_width = roi_box.xmax - roi_box.xmin + 1,
        roi_height = roi_box.ymax - roi_box.ymin + 1;
    int hRem = roi_height % p, wRem = roi_width % q;
    int ymin = roi_box.ymin + (roi_height / p) * poolIndH + min(poolIndH, hRem);
    int ymax = roi_box.ymin + (roi_height / p) * (poolIndH + 1) + min(poolIndH + 1, hRem) - 1;
    int xmin = roi_box.xmin + (roi_width / q) * poolIndW + min(poolIndW, wRem);
    int xmax = roi_box.xmin + (roi_width / q) * (poolIndW + 1) + min(poolIndW + 1, wRem) - 1;
    return {roi_box.img_indx, xmin, ymin, xmax, ymax};
}


__global__
void nonFunctionalSparseRoiPoolingKernel(int *in_loc, float *in_feats,
    int *out_loc, float *out_feats,
    int sparse_n, int n, int c,
    int h, int w, RoiBox roi_box, int p, int q, int roi_box_idx,
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
            in_loc[4 * j], roi_box_idx, in_loc[4 * j + 1], poolIdxH, poolIdxW);

        float value;
        my_lookup_key(pHashTable, pool_val_key, &value);
        if ((value == vEmpty)  || (value < in_feats[j])) {
            printf("Inserting, since prev value: %f, next value: %f\n",
                value, in_feats[j]);
            my_insert_key_value(pHashTable, pool_val_key, in_feats[j]);
        }
        thread_index += blockDim.x * gridDim.x;
    }
}

__global__
void naiveSparseRoiPoolingKernel(int *in_loc, float *in_feats,
    int *out_loc, float *out_feats, RoiBox *roi_boxes,
    int sparse_n, int n, int c,
    int h, int w, int b, int p, int q) {
    uint orig_ti = blockIdx.x * blockDim.x + threadIdx.x;
    uint thread_index = orig_ti;
    while (thread_index < p * q * b) {
        int roi_box_idx = thread_index / (p * q);
        int poolIdxW = thread_index % q;
        int poolIdxH = (thread_index / q) % p;
        bool is_first = true;
        RoiBox roi_sub_box = get_roi_sub_box(roi_boxes[roi_box_idx], p, q,
            poolIdxH, poolIdxW);

        out_loc[thread_index * 5] = 0;
        out_loc[thread_index * 5 + 1] = roi_box_idx;
        out_loc[thread_index * 5 + 2] = 0;
        out_loc[thread_index * 5 + 3] = poolIdxH;
        out_loc[thread_index * 5 + 4] = poolIdxW;

        for (uint i = 0; i < sparse_n; i++) {
            if (!(in_roi_box(i, in_loc, roi_sub_box))) {
                continue;
            }
            if (is_first) {
                out_feats[thread_index] = in_feats[i];
                is_first = false;
            } else {
                out_feats[thread_index] = ::fmaxf(out_feats[thread_index],
                    in_feats[i]);
            }
        }
        thread_index += blockDim.x * gridDim.x;
    }

    // __syncthreads();

    // // Compress sparse array
    // uint write_head = 0;
    // if (orig_ti == 0) {
    //     uint non_zeros = 0;
    //     for (uint read_head = 0; read_head < p * q * b; read_head++) {
    //         if (out_feats[read_head] != 0) {
    //             non_zeros++;
    //         }
    //     }
    //     // printf("Non zeros: %d\n", non_zeros);
    //     for (uint read_head = non_zeros; read_head < p * q * b; read_head++) {
    //         if (out_feats[read_head] != 0) {
    //             while (out_feats[write_head] != 0) {
    //                 write_head++;
    //             }
    //             // printf("Compacting %d to %d\n", read_head, write_head);
    //             out_loc[write_head * 5] = out_loc[read_head * 5];
    //             out_loc[write_head * 5 + 1] = out_loc[read_head * 5 + 1];
    //             out_loc[write_head * 5 + 2] = out_loc[read_head * 5 + 2];
    //             out_loc[write_head * 5 + 3] = out_loc[read_head * 5 + 3];
    //             out_loc[write_head * 5 + 4] = out_loc[read_head * 5 + 4];
    //             out_feats[write_head] = out_feats[read_head];
    //             // For now, not erasing read head
    //             write_head++;
    //         }
    //     }
    // }
}


void cudaSparseRoiPooling(int *d_in_loc, float *d_in_feats,
    int *d_out_loc, float *d_out_feats, int sparse_n, int n, int c,
    int h, int w, RoiBox *d_roi_boxes, int b, int p, int q,
    Implementation type) {

    int blocks = 512;
    int threadsPerBlock = 128;

    /*
    Iterate through all in_loc for sparse_n steps
    If it's in the right area, find its cell in the output
    If there's an entry, max it. Otherwise, just enter it.

    At the end, one thread reads through all entries and writes it to
    out_loc, out_feats
     */
    // KeyValue* pHashTable = create_hashtable();

    /* Call cudakernel here */
    if (type == NAIVE) {
        naiveSparseRoiPoolingKernel<<<blocks, threadsPerBlock>>>(
            d_in_loc, d_in_feats, d_out_loc, d_out_feats, d_roi_boxes,
            sparse_n, n, c,
            h, w, b, p, q
        );
    } else {
        assert(false);
    }
    // destroy_hashtable(pHashTable);
}
