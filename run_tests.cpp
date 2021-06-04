#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <iostream>
#include <math.h>

#include <cuda_runtime.h>

#include "sparse_roi_pool_device.cuh"
#include "ta_utilities.hpp"
#include "sparse_utils.hpp"
#include "test_utils.hpp"
/*
 * NOTE: You can use this macro to easily check cuda error codes
 * and get more information.
 * 
 * Modified from:
 * http://stackoverflow.com/questions/14038589/
 *         what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
 */
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

/*
 * Fills fill with random numbers is [0, 1]. Size is number of elements to
 * assign.
 */
void randomFill(float *fill, int size) {
    for (int i = 0; i < size; i++) {
        float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        fill[i] = r;
    }
}

void checkSparseRoiPooling(const int *in_loc, const float *in_feats,
    const int *out_loc, const float *out_feats, int n) {
    bool correct = true;

    // TODO: Insert validation logic here
    // for (int i = 0; i < n; i++) {
        
    //         if (a[i + n * j] != b[j + n * i]) {
    //             correct = false;
    //             fprintf(stderr,
    //                 "Pooling failed: a[%d, %d] != b[%d, %d], %f != %f\n",
    //                 i, j, j, i, a[i + n * j], b[j + n * i]);
    //             assert(correct);
    //         }
    //     }
    // }

    assert(correct);
}

int main(int argc, char *argv[]) {

    // These functions allow you to select the least utilized GPU 
    // on your system as well as enforce a time limit on program execution.
    // Please leave these enabled as a courtesy to your fellow classmates
    // if you are using a shared computer. You may ignore or remove these 
    // functions if you are running on your local machine.
    // TA_Utilities::select_coldest_GPU();
    int max_time_allowed_in_seconds = 10;
    TA_Utilities::enforce_time_limit(max_time_allowed_in_seconds);
    
    // Seed random number generator
    srand(2016);

    std::string kernel = "all";
    int size_to_run = -1;

    // Check arguments
    assert(argc <= 3);
    if (argc >= 2)
        size_to_run = atoi(argv[1]);
    if (argc == 3)
        kernel = argv[2];

    if (!(size_to_run == -1  ||
         size_to_run == 512  ||
         size_to_run == 1024 ||
         size_to_run == 2048 ||
         size_to_run == 4096))
    {
        fprintf(stderr,
            "Program only designed to run sizes 512, 1024, 2048, 4096\n");
    }

    assert(kernel == "all"  ||
        kernel == "cpu"     ||
        kernel == "naive"   ||
        kernel == "optimal");


    run_mini_test1(CPU);
    run_mini_test2(CPU);
    run_mini_test1(NAIVE);
    run_mini_test2(NAIVE);
    return 1;

    // Run the implementations for all desired sizes (2^9 = 512, 
    // 2^12 = 4096)
    for (int _i = 9; _i < 13; _i++) {
        int n = 1 << _i;
        float sparsity_level = 0.05;
        int sparse_n = static_cast<int>(ceil(n * sparsity_level));
        int p = 3, q = 3;
        int num_images = 1;
        int c = 1;
        int h = n, w = n;
        std::vector<RoiBox> roi_boxes;
        // img_idx, xmin, ymin, xmax, ymax
        roi_boxes.push_back({0, 0, 0, n / 3, n / 3});
        roi_boxes.push_back({0, 0, 0, n-1, n-1});
        int out_size = num_images * roi_boxes.size() * c * p * q;

        if (size_to_run != -1 && size_to_run != n)
            continue;

        assert(n % 64 == 0);

        cudaEvent_t start;
        cudaEvent_t stop;

#define START_TIMER() {                                                        \
            gpuErrChk(cudaEventCreate(&start));                                \
            gpuErrChk(cudaEventCreate(&stop));                                 \
            gpuErrChk(cudaEventRecord(start));                                 \
        }

#define STOP_RECORD_TIMER(name) {                                              \
            gpuErrChk(cudaEventRecord(stop));                                  \
            gpuErrChk(cudaEventSynchronize(stop));                             \
            gpuErrChk(cudaEventElapsedTime(&name, start, stop));               \
            gpuErrChk(cudaEventDestroy(start));                                \
            gpuErrChk(cudaEventDestroy(stop));                                 \
        }

        // Initialize timers
        float cpu_ms = -1;
        float naive_gpu_ms = -1;
        float optimal_gpu_ms = -1;

        // Allocate host memory
        float *in_feats = new float[sparse_n];
        int *in_loc = new int[sparse_n * 4];  // num_images, c, h, w

        // Initialize in_feats data to random numbers in [0, 1]
        generate_random_sparse(in_loc, in_feats, 0.0f, 1.0f, sparse_n,
                               num_images, c, h, w);

        float *out_feats, *d_out_feats;
        int *out_loc, *d_out_loc;

        std::cout << "Setting output memory for size " << n << std::endl; 

        // CPU implementation
        if (kernel == "cpu" || kernel == "all") {
            // Output has maximum size of num_images * num_channels * num_boxes * p * q
            out_feats = new float[out_size];
            out_loc = new int[out_size * 5];  // num_images, b, c, p, q

            // Initialize output to 0
            memset(out_loc, 0, out_size * 5 * sizeof(int));
            memset(out_feats, 0, out_size * sizeof(float));

            START_TIMER();
            
            cpuSparseRoiPooling(in_loc, in_feats, out_loc, out_feats, sparse_n,
                num_images, c, h, w, roi_boxes, p, q);
            STOP_RECORD_TIMER(cpu_ms);
            // print_sparse(out_loc, out_feats, out_size, 5);

            printf("Size %d naive CPU: %f ms\n", n, cpu_ms);
        }

        // Naive GPU implementation
        if (kernel == "naive" || kernel == "all") {
            d_out_feats = new float[out_size];
            d_out_loc = new int[out_size * 5];  // num_images, b, c, p, q
            memset(d_out_feats, 0, out_size * sizeof(float));
            memset(d_out_loc, 0, out_size * 5 * sizeof(int));
            
            START_TIMER();
            cudaSparseRoiPooling(in_loc, in_feats, d_out_loc, d_out_feats,
                sparse_n, num_images, c, h, w, roi_boxes, p, q,
                NAIVE);
            // cpuSparseRoiPooling(in_loc, in_feats, d_out_loc, d_out_feats, sparse_n,
            //     num_images, c, h, w, roi_boxes, p, q);
            STOP_RECORD_TIMER(naive_gpu_ms);

            checkSparseRoiPooling(in_loc, in_feats, out_loc, out_feats, num_images);

            printf("Size %d naive GPU: %f ms\n", n, naive_gpu_ms);
        }

        // // Optimal GPU implementation TODO: no optimal implementation planned
        // if (kernel == "optimal"    || kernel == "all") {
        //     START_TIMER();
        //     cudaSparseRoiPooling(d_in_loc, d_in_feats, d_out_loc, d_out_feats,
        //         num_images, c, h, w, roi_boxes, p, q,
        //         OPTIMAL);
        //     STOP_RECORD_TIMER(optimal_gpu_ms);

        //     gpuErrChk(cudaMemcpy(out_feats, d_out_feats,
        //         sparse_n * sizeof(float), 
        //         cudaMemcpyDeviceToHost));
        //     gpuErrChk(cudaMemcpy(out_loc, d_out_loc,
        //         sparse_n * 5 * sizeof(int), 
        //         cudaMemcpyDeviceToHost));
        //     checkSparseRoiPooling(in_loc, in_feats, out_loc, out_feats, num_images);

        //     memset(out_feats, 0, sparse_n * sizeof(float));
        //     memset(out_loc, 0, sparse_n * 5 * sizeof(int));
        //     gpuErrChk(cudaMemset(d_out_feats, 0, sparse_n * sizeof(float)));
        //     gpuErrChk(cudaMemset(d_out_loc, 0, sparse_n * 5 * sizeof(int)));

        //     printf("Size %d optimal GPU: %f ms\n", n, optimal_gpu_ms);
        // }

        if (kernel == "all") {
            // check whether gpu roi pooling output matches the cpu output
            bool isEqual = is_sparse_equal(out_loc, out_feats, out_size,
                                           d_out_loc, d_out_feats, out_size);
            if (isEqual) {
                std::cout << "CPU output and GPU output are equal" << std::endl;
            } else {
                std::cout << "!!CPU output and GPU output do not match!!" << std::endl;
            }
        }

        delete [] in_feats;
        delete [] in_loc;
        delete [] out_feats;
        delete [] out_loc;
        delete [] d_out_feats;
        delete [] d_out_loc;

        printf("\n");
    }
}
