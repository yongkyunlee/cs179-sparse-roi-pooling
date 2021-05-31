#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <cuda_runtime.h>

#include "sparse_roi_pool_device.cuh"
#include "ta_utilities.hpp"
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


void mini_test_0() {
    const int size = 8;
    const float data[size][size] =
        {{0.88, 0.44,    0, 0.16, 0.37,    0, 0.96, 0.27},
          {   0, 0.45, 0.57, 0.16, 0.63, 0.29,    0,    0},
          {0.66,    0, 0.82, 0.64, 0.54,    0, 0.59, 0.26},
          {0.85, 0.34, 0.76, 0.84, 0.29, 0.75, 0.62, 0.25},
          {0.32, 0.74, 0.21, 0.39, 0.34, 0.03, 0.33, 0.48},
          {0.20, 0.14, 0.16, 0.13, 0.73, 0.65, 0.96, 0.32},
          {0.19, 0.69, 0.09, 0.86, 0.88, 0.07, 0.01, 0.48},
          {0.83, 0.24, 0.97, 0.04, 0.24, 0.35, 0.50, 0.91}};
    std::vector<RoiBox> roi_boxes;

    roi_boxes.push_back({0, 0, 0, 2, 2});
    int p, q = 2;
    const float ans_1[roi_boxes.size()][p][q] = 
        {{{0.88, 0.57},
          {0.66, 0.82}}};    
    roi_boxes.pop_back();

    roi_boxes.push_back({0, 0, 3, 6, 7});
    roi_boxes.push_back({0, 5, 0, 7, 2});
    p, q = 3;
    const float ans_2[roi_boxes.size()][p][q] = 
        {{{0.85, 0.84, 0.75},
          {0.69, 0.88, 0.96},
          {0.97, 0.24, 0.50}},
         {{0, 0.96, 0.27},
          {0.29, 0, 0},
          {0, 0.59, 0.26}}
        };    
    roi_boxes.pop_back();    
    roi_boxes.pop_back();

    // TODO: convert to sparse form and test

}


int main(int argc, char *argv[]) {

    // These functions allow you to select the least utilized GPU 
    // on your system as well as enforce a time limit on program execution.
    // Please leave these enabled as a courtesy to your fellow classmates
    // if you are using a shared computer. You may ignore or remove these 
    // functions if you are running on your local machine.
    TA_Utilities::select_coldest_GPU();
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

    // Run the implementations for all desired sizes (2^9 = 512, 
    // 2^12 = 4096)
    for (int _i = 9; _i < 13; _i++) {
        int n = 1 << _i;
        float sparsity_level = 0.05;
        int sparse_n = n * sparsity_level;
        int p, q = 3;
        int num_images = 1;
        int c = 1;
        int h, w = sparse_n;
        std::vector<RoiBox> roi_boxes;
        roi_boxes.push_back({0, 0, 0, 2, 2});

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
        // Output is larger than needed, so that we don't need to dynamically
        // size
        float *out_feats = new float[sparse_n];
        // TODO: initialize loc
        int *in_loc = new int[sparse_n * 4];  // num_images, c, h, w
        int *out_loc = new int[sparse_n * 5];  // num_images, b, c, p, q

        // Allocate device memory
        float *d_in_feats;
        float *d_out_feats;
        int *d_in_loc;
        int *d_out_loc;
        gpuErrChk(cudaMalloc(&d_in_feats, sparse_n * sizeof(float)));
        gpuErrChk(cudaMalloc(&d_out_feats, sparse_n * sizeof(float)));
        gpuErrChk(cudaMalloc(&d_in_loc, sparse_n * 4 * sizeof(int)));
        gpuErrChk(cudaMalloc(&d_out_loc, sparse_n * 5 * sizeof(int)));

        // Initialize in_feats data to random numbers in [0, 1]
        // randomFill ...

        // Copy input to GPU
        gpuErrChk(cudaMemcpy(d_in_feats, in_feats, sparse_n * sizeof(float), 
            cudaMemcpyHostToDevice));
        gpuErrChk(cudaMemcpy(d_in_loc, in_loc, sparse_n * 4 * sizeof(int), 
            cudaMemcpyHostToDevice));

        // TODO: set up in_loc

        // CPU implementation
        if (kernel == "cpu" || kernel == "all") {
            START_TIMER();
            cpuSparseRoiPooling(in_loc, in_feats, out_loc, out_feats, sparse_n,
                num_images, c, h, w, roi_boxes, p, q);
            STOP_RECORD_TIMER(cpu_ms);

            checkSparseRoiPooling(in_loc, in_feats, out_loc, out_feats, num_images);
            memset(out_feats, 0, sparse_n * sizeof(float));
            memset(out_loc, 0, sparse_n * 5 * sizeof(int));

            printf("Size %d naive CPU: %f ms\n", n, cpu_ms);
        }

        // Naive GPU implementation
        if (kernel == "naive" || kernel == "all") {
            START_TIMER();
            cudaSparseRoiPooling(d_in_loc, d_in_feats, d_out_loc, d_out_feats,
                num_images, c, h, w, roi_boxes, p, q,
                NAIVE);
            STOP_RECORD_TIMER(naive_gpu_ms);

            gpuErrChk(cudaMemcpy(out_feats, d_out_feats,
                sparse_n * sizeof(float), 
                cudaMemcpyDeviceToHost));
            gpuErrChk(cudaMemcpy(out_loc, d_out_loc,
                sparse_n * 5 * sizeof(int), 
                cudaMemcpyDeviceToHost));
            checkSparseRoiPooling(in_loc, in_feats, out_loc, out_feats, num_images);

            memset(out_feats, 0, sparse_n * sizeof(float));
            memset(out_loc, 0, sparse_n * 5 * sizeof(int));
            gpuErrChk(cudaMemset(d_out_feats, 0, sparse_n * sizeof(float)));
            gpuErrChk(cudaMemset(d_out_loc, 0, sparse_n * 5 * sizeof(int)));

            printf("Size %d naive GPU: %f ms\n", n, naive_gpu_ms);
        }

        // // Optimal GPU implementation TODO: no optimal implementation planned
        // if (kernel == "optimal"    || kernel == "all") {
        //     START_TIMER();
            // cudaSparseRoiPooling(d_in_loc, d_in_feats, d_out_loc, d_out_feats,
            //     num_images, c, h, w, roi_boxes, p, q,
        //         OPTIMAL);
        //     STOP_RECORD_TIMER(optimal_gpu_ms);

            // gpuErrChk(cudaMemcpy(out_feats, d_out_feats,
            //     sparse_n * sizeof(float), 
            //     cudaMemcpyDeviceToHost));
            // gpuErrChk(cudaMemcpy(out_loc, d_out_loc,
            //     sparse_n * 5 * sizeof(int), 
            //     cudaMemcpyDeviceToHost));
        //     checkSparseRoiPooling(in_loc, in_feats, out_loc, out_feats, num_images);

            // memset(out_feats, 0, sparse_n * sizeof(float));
            // memset(out_loc, 0, sparse_n * 5 * sizeof(int));
            // gpuErrChk(cudaMemset(d_out_feats, 0, sparse_n * sizeof(float)));
            // gpuErrChk(cudaMemset(d_out_loc, 0, sparse_n * 5 * sizeof(int)));

        //     printf("Size %d optimal GPU: %f ms\n", n, optimal_gpu_ms);
        // }

        // Free host memory
        delete[] in_feats;
        delete[] out_feats;
        delete[] in_loc;
        delete[] out_loc;

        // Free device memory
        gpuErrChk(cudaFree(d_in_feats));
        gpuErrChk(cudaFree(d_out_feats));
        gpuErrChk(cudaFree(d_in_loc));
        gpuErrChk(cudaFree(d_out_loc));

        printf("\n");
    }
}