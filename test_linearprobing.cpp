// SOURCE CODE FROM https://github.com/nosferalatu/SimpleGPUHashTable
#include "algorithm"
#include "random"
#include "stdint.h"
#include "stdio.h"
#include "unordered_map"
#include "unordered_set"
#include "vector"
#include "chrono"
#include "linearprobing.h"

// Create random keys/values in the range [0, kEmpty)
// kEmpty is used to indicate an empty slot
std::vector<KeyValue> generate_random_keyvalues(std::mt19937& rnd, uint32_t numkvs)
{
    std::uniform_int_distribution<uint32_t> dis(0, kEmptyElem - 1);
    std::uniform_real_distribution<float> fdis(0,
        static_cast<float>(vEmpty - 1));

    std::vector<KeyValue> kvs;
    kvs.reserve(numkvs);

    for (uint32_t i = 0; i < numkvs; i++)
    {
        // uint32_t rand0 = dis(rnd);
        float rand1 = fdis(rnd);
        kvs.push_back(KeyValue{
            thrust::make_tuple(dis(rnd), dis(rnd), dis(rnd), dis(rnd), dis(rnd)),
            rand1
        });
    }

    return kvs;
}

// return numshuffledkvs random items from kvs
std::vector<KeyValue> shuffle_keyvalues(std::mt19937& rnd, std::vector<KeyValue> kvs, uint32_t numshuffledkvs)
{
    std::shuffle(kvs.begin(), kvs.end(), rnd);

    std::vector<KeyValue> shuffled_kvs;
    shuffled_kvs.resize(numshuffledkvs);

    std::copy(kvs.begin(), kvs.begin() + numshuffledkvs, shuffled_kvs.begin());

    return shuffled_kvs;
}

// using Time = std::chrono::time_point<std::chrono::steady_clock>;

// Time start_timer() 
// {
//     return std::chrono::high_resolution_clock::now();
// }

// double get_elapsed_time(Time start) 
// {
//     Time end = std::chrono::high_resolution_clock::now();

//     std::chrono::duration<double> d = end - start;
//     std::chrono::microseconds us = std::chrono::duration_cast<std::chrono::microseconds>(d);
//     return us.count() / 1000.0f;
// }

void test_unordered_map(std::vector<KeyValue> insert_kvs, std::vector<KeyValue> delete_kvs) 
{
    // Time timer = start_timer();

    printf("Timing std::unordered_map...\n");

    {
        std::unordered_map<d_pool_key, float, CustomThrustHash> kvs_map;
        for (auto& kv : insert_kvs) 
        {
            kvs_map[kv.key] = kv.value;
        }
        for (auto& kv : delete_kvs)
        {
            auto i = kvs_map.find(kv.key);
            if (i != kvs_map.end())
                kvs_map.erase(i);
        }
    }

    // double milliseconds = get_elapsed_time(timer);
    // double seconds = milliseconds / 1000.0f;
    // printf("Total time for std::unordered_map: %f ms (%f million keys/second)\n", 
    //     milliseconds, kNumKeyValues / seconds / 1000000.0f);
}

void test_correctness(std::vector<KeyValue> insert_kvs, std::vector<KeyValue> delete_kvs, std::vector<KeyValue> kvs)
{
    printf("Testing that there are no duplicate keys...\n");
    std::unordered_set<d_pool_key, CustomThrustHash> unique_keys;
    for (uint32_t i = 0; i < kvs.size(); i++)
    {
        if (i % 10000000 == 0)
            printf("    Verifying %d/%d\n", i, (uint32_t)kvs.size());

        KeyValue* node = &kvs[i];
        if (unique_keys.find(node->key) != unique_keys.end())
        {
            printf("Duplicate key found in GPU hash table at slot %d\n", i);
            exit(-1);
        }
        unique_keys.insert(node->key);
    }

    printf("Building unordered_map from original list...\n");
    std::unordered_map<d_pool_key, std::vector<float>, CustomThrustHash> all_kvs_map;
    for (unsigned int i = 0; i < insert_kvs.size(); i++)
    {
        if (i % 10000000 == 0)
            printf("    Inserting %d/%d\n", i, (uint32_t)insert_kvs.size());

        auto iter = all_kvs_map.find(insert_kvs[i].key);
        if (iter == all_kvs_map.end())
        {
            all_kvs_map[insert_kvs[i].key] = std::vector<float>({ insert_kvs[i].value });
        }
        else
        {
            iter->second.push_back(insert_kvs[i].value);
        }
    }

    for (unsigned int i = 0; i < delete_kvs.size(); i++)
    {
        if (i % 10000000 == 0)
            printf("    Deleting %d/%d\n", i, (uint32_t)delete_kvs.size());

        auto iter = all_kvs_map.find(delete_kvs[i].key);
        if (iter != all_kvs_map.end())
        {
            all_kvs_map.erase(iter);
        }
    }

    if (unique_keys.size() != all_kvs_map.size())
    {
        printf("# of unique keys in hashtable is incorrect\n");
        exit(-1);
    }

    printf("Testing that each key/value in hashtable is in the original list...\n");
    for (uint32_t i = 0; i < kvs.size(); i++)
    {
        if (i % 10000000 == 0)
            printf("    Verifying %d/%d\n", i, (uint32_t)kvs.size());

        auto iter = all_kvs_map.find(kvs[i].key);
        if (iter == all_kvs_map.end())
        {
            printf("Hashtable key not found in original list\n");
            exit(-1);
        }

        std::vector<float>& values = iter->second;
        if (std::find(values.begin(), values.end(), kvs[i].value) == values.end())
        {
            printf("Hashtable value not found in original list\n");
            exit(-1);
        }
    }

    printf("Deleting std::unordered_map and std::unique_set...\n");

    return;
}

int main() 
{
    // To recreate the same random numbers across runs of the program, set seed to a specific
    // number instead of a number from random_device
    std::random_device rd;
    uint32_t seed = rd();
    std::mt19937 rnd(seed);  // mersenne_twister_engine

    printf("Random number generator seed = %u\n", seed);

    while (true)
    {
        printf("Initializing keyvalue pairs with random numbers...\n");

        std::vector<KeyValue> insert_kvs = generate_random_keyvalues(rnd, kNumKeyValues);
        std::vector<KeyValue> delete_kvs = shuffle_keyvalues(rnd, insert_kvs, kNumKeyValues / 2);

        // Begin test
        printf("Testing insertion/deletion of %d/%d elements into GPU hash table...\n",
            (uint32_t)insert_kvs.size(), (uint32_t)delete_kvs.size());

        // Time timer = start_timer();

        KeyValue* pHashTable = create_hashtable();

        // Insert items into the hash table
        const uint32_t num_insert_batches = 16;
        uint32_t num_inserts_per_batch = (uint32_t)insert_kvs.size() / num_insert_batches;
        for (uint32_t i = 0; i < num_insert_batches; i++)
        {
            insert_hashtable(pHashTable, insert_kvs.data() + i * num_inserts_per_batch, num_inserts_per_batch);
        }

        // Delete items from the hash table
        const uint32_t num_delete_batches = 8;
        uint32_t num_deletes_per_batch = (uint32_t)delete_kvs.size() / num_delete_batches;
        for (uint32_t i = 0; i < num_delete_batches; i++)
        {
            delete_hashtable(pHashTable, delete_kvs.data() + i * num_deletes_per_batch, num_deletes_per_batch);
        }

        // Get all the key-values from the hash table
        std::vector<KeyValue> kvs = iterate_hashtable(pHashTable);

        destroy_hashtable(pHashTable);

        // Summarize results
        // double milliseconds = get_elapsed_time(timer);
        // double seconds = milliseconds / 1000.0f;
        // printf("Total time (including memory copies, readback, etc): %f ms (%f million keys/second)\n", milliseconds,
        //     kNumKeyValues / seconds / 1000000.0f);

        test_unordered_map(insert_kvs, delete_kvs);

        test_correctness(insert_kvs, delete_kvs, kvs);

        printf("Success\n");
    }

    return 0;
}