// SOURCE CODE FROM https://github.com/nosferalatu/SimpleGPUHashTable
#pragma once
#include <thrust/tuple.h>
#include "sparse_utils.hpp"

typedef thrust::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> d_pool_key;

__device__ uint32_t tuple_hash(d_pool_key k);
__device__ uint32_t alt_tuple_hash(d_pool_key k);


class CustomThrustHash
{
    public:
    // Implement a hash function
    std::size_t operator()(const d_pool_key& k) const
    {
        // Inspired by https://github.com/python/cpython/blob/3.7/Objects/tupleobject.c#L348
        std::size_t h1 = thrust::get<0>(k);
        std::size_t h2 = thrust::get<1>(k);
        std::size_t h3 = thrust::get<2>(k);
        std::size_t h4 = thrust::get<3>(k);
        std::size_t h5 = thrust::get<4>(k);
        return ((((((((h1 ^ h2) * base_mult)
            ^ h3) * (base_mult + base_adder))
            ^ h4) * (base_mult + base_adder * 2))
            ^ h5) * (base_mult + base_adder * 3));
    }
};

struct KeyValue
{
    d_pool_key key;
    float value;
};

const uint32_t kHashTableCapacity = 128 * 1024 * 1024;

const uint32_t kNumKeyValues = 8 * 1024 * 1024; // kHashTableCapacity / 2;

constexpr uint32_t kEmptyElem = 0xffffffff;
constexpr float vEmpty = 0xffffffff;

KeyValue* create_hashtable();

void insert_hashtable(KeyValue* hashtable, const KeyValue* kvs, uint32_t num_kvs);

void lookup_hashtable(KeyValue* hashtable, KeyValue* kvs, uint32_t num_kvs);

void delete_hashtable(KeyValue* hashtable, const KeyValue* kvs, uint32_t num_kvs);

std::vector<KeyValue> iterate_hashtable(KeyValue* hashtable);

void destroy_hashtable(KeyValue* hashtable);