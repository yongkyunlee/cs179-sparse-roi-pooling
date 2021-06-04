# CS179 Project
## Alexander Cui and Yongkyun (Daniel) Lee

## Usage Instructions

Run ```make``` in the current directory to compile the code. Then run ```./run_tests``` to run the RoiSparsePooling on CPU and GPU.

There are two options: ```./run_tests [size_to_run] [kernel]```. size_to_run can be one of 512, 1024, 2048, 4096, and 16384, and it is the size of the matrix for the test. kernel can be one of 'all', 'cpu', and 'gpu'. Without the optional arguments, the program runs the tests in both cpu and gpu for all sizes if no option is selected.

## Project Description

ROI pooling is a widely used pooling operation in object detection tasks using CNN to perform max pooling on inputs of non-uniform sizes to obtain fixed-size feature maps. More details could be found on https://towardsdatascience.com/region-of-interest-pooling-f7c637f409af.

Our project is an implementation of Region-of-Interest (ROI) pooling for sparse matrices. There exist implementations of ROI pooling for dense matrices and implementations of CNN for sparse matrices. However, there is no CPU and GPU implementation of ROI pooling for sparse matrices as of our knowledge.

By running ```./run_tests```, we first run sparse ROI pooling on some test examples that can be found in ```test_utils.cpp```. Then, random sparse matrices and ROI boxes are generated for multiple sizes, and the sparse ROI pooling results are compared for the CPU output and GPU output. The result is described in the following section.

We initially attempted to integrate our implementation to SparseConvNet developed by Facebook as demonstrated in the initial cpu_demo_2021_submission.zip. Yet, SparseConvNet was a framework built for conventional convolutional and pooling operations for fixed sized kernels, resulting in limitations of usage. Thus, we decided to build the sparse ROI pooling functionalities from scratch. The code can also be found on https://github.com/yongkyunlee/cs179-sparse-roi-pooling.

## Results

For the test examples, the sparse ROI pooling output matches the expected answer for both CPU and GPU implemenations. "Is answer correct: 1" is printed when ```run_tests``` is executed.

For random matrices of larger size, we first see that the outputs of CPU and GPU implementations match. Also, we observe that GPU implementation decreases the runtime to about one third compared to the CPU implemenation. Note that this result is observed when there are 128 ROI boxes, which is the default configuration for our test script. More specific details on the comparison of performance for diverse cases will be explained in the next section.

## Performance Analysis

* How much better is the GPU version?
* Are there things that could be improved?

The history of experiments can be found in ```results.txt```.