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
         size_to_run == 4096 ||
         size_to_run ==  16384))
    {
        fprintf(stderr,
            "Program only designed to run sizes 512, 1024, 2048, 4096\n");
    }

    assert(kernel == "all"  ||
        kernel == "cpu"     ||
        kernel == "gpu");


    run_mini_test(1, CPU);
    run_mini_test(2, CPU);
    run_mini_test(1, GPU);
    run_mini_test(2, GPU);

    // Run the implementations for all desired sizes (2^9 = 512, 
    // 2^12 = 4096)
    for (int _i = 9; _i < 15; _i++) {
        int n = 1 << _i;
        if (size_to_run != -1 && size_to_run != n)
            continue;

        float sparsity_level = 0.05;
        int sparse_n = static_cast<int>(ceil(n * sparsity_level));
        int p = 7, q = 7;
        int num_images = 1;
        int c = 1;
        int h = n, w = n;
        // Number of RoI boxes per batch in Fast RCNN Paper
        // https://arxiv.org/pdf/1504.08083.pdf
        int num_roi_boxes = 128;
        std::vector<RoiBox> roi_boxes = generate_random_roi_boxes(
            num_images, h, w, num_roi_boxes, p, q);

        int out_size = num_images * roi_boxes.size() * c * p * q;

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
        // float optimal_gpu_ms = -1;

        // Allocate host memory
        float *in_feats = new float[sparse_n];
        int *in_loc = new int[sparse_n * 4];  // num_images, c, h, w

        // Initialize in_feats data to random numbers in [0, 1]
        generate_random_sparse(in_loc, in_feats, 0.0f, 1.0f, sparse_n,
                               num_images, c, h, w);

        float *out_feats = NULL, *naive_out_feats = NULL;
        int *out_loc = NULL, *naive_out_loc = NULL;

        // Allocate device memory
        float *d_in_feats;
        float *d_out_feats;
        int *d_in_loc;
        int *d_out_loc;
        RoiBox *d_roi_boxes;

        gpuErrChk(cudaMalloc(&d_in_feats, sparse_n * sizeof(float)));
        gpuErrChk(cudaMalloc(&d_in_loc, sparse_n * 4 * sizeof(int)));

        gpuErrChk(cudaMalloc(&d_out_feats, out_size * sizeof(float)));
        gpuErrChk(cudaMalloc(&d_out_loc, out_size * 5 * sizeof(int)));

        gpuErrChk(cudaMalloc(&d_roi_boxes, roi_boxes.size() * sizeof(RoiBox)));

        // Copy input to GPU
        gpuErrChk(cudaMemcpy(d_in_feats, in_feats, sparse_n * sizeof(float),
            cudaMemcpyHostToDevice));
        gpuErrChk(cudaMemcpy(d_in_loc, in_loc, sparse_n * 4 * sizeof(int),
            cudaMemcpyHostToDevice));
        // Zero output
        gpuErrChk(cudaMemset(d_out_feats, 0, out_size * sizeof(float)));

        RoiBox *dst = d_roi_boxes;
        for (unsigned int i = 0; i < roi_boxes.size(); i++) {
            RoiBox *src = &roi_boxes[i];
            cudaMemcpy(dst, src, sizeof(RoiBox), cudaMemcpyHostToDevice);
            dst += 1;
        }

        // CPU implementation
        if (kernel == "cpu" || kernel == "all") {
            // Output has maximum size of num_images * num_channels * num_boxes * p * q
            out_loc = new int[out_size * 5];  // num_images, b, c, p, q
            out_feats = new float[out_size];

            // Initialize output to 0
            memset(out_loc, 0, out_size * 5 * sizeof(int));
            memset(out_feats, 0, out_size * sizeof(float));

            START_TIMER();
            
            cpuSparseRoiPooling(in_loc, in_feats, out_loc, out_feats, sparse_n,
                num_images, c, h, w, roi_boxes, p, q);
            STOP_RECORD_TIMER(cpu_ms);
            // print_sparse(out_loc, out_feats, out_size, 5);

            printf("Size %d CPU: %f ms\n", n, cpu_ms);
        }

        // Naive GPU implementation
        if (kernel == "gpu" || kernel == "all") {
            // Output has maximum size of num_images * num_channels * num_boxes * p * q
            naive_out_loc = new int[out_size * 5];  // num_images, b, c, p, q
            naive_out_feats = new float[out_size];

            // Initialize output to 0
            memset(naive_out_loc, 0, out_size * 5 * sizeof(int));
            memset(naive_out_feats, 0, out_size * sizeof(float));

            START_TIMER();
            cudaSparseRoiPooling(d_in_loc, d_in_feats, d_out_loc, d_out_feats,
                sparse_n, num_images, c, h, w, d_roi_boxes,
                roi_boxes.size(), p, q, GPU);
            STOP_RECORD_TIMER(naive_gpu_ms);

            gpuErrChk(cudaMemcpy(naive_out_feats, d_out_feats,
                out_size * sizeof(float),
                cudaMemcpyDeviceToHost));
            gpuErrChk(cudaMemcpy(naive_out_loc, d_out_loc,
                out_size * 5 * sizeof(int),
                cudaMemcpyDeviceToHost));

            printf("Size %d GPU: %f ms\n", n, naive_gpu_ms);
        }

        if (kernel == "all") {
            // check whether gpu roi pooling output matches the cpu output
            bool isEqual = is_sparse_equal(out_loc, out_feats, out_size,
                naive_out_loc, naive_out_feats, out_size);
            if (isEqual) {
                std::cout << "CPU output and GPU output are equal" << std::endl;
            } else {
                std::cout << "!!CPU output and GPU output do not match!!" << std::endl;
            }
        }

        delete [] in_feats;
        delete [] in_loc;

        if (out_feats != NULL) delete [] out_feats;
        if (out_loc != NULL) delete [] out_loc;
        if (naive_out_loc != NULL) delete [] naive_out_feats;
        if (naive_out_loc != NULL) delete [] naive_out_loc;

        // Free device memory
        gpuErrChk(cudaFree(d_in_feats));
        gpuErrChk(cudaFree(d_out_feats));
        gpuErrChk(cudaFree(d_in_loc));
        gpuErrChk(cudaFree(d_out_loc));
        gpuErrChk(cudaFree(d_roi_boxes));

        printf("\n");
    }
}
