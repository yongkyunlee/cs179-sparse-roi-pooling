# CS179 Project
## Alexander Cui and Yongkyun (Daniel) Lee

## Usage Instructions

Run ```make``` in the current directory to compile the code. Then run ```./run_tests``` to run the RoISparsePooling on CPU and GPU.

There are two options: ```./run_tests [size_to_run] [kernel]```. size_to_run can be one of 512, 1024, 2048, 4096, and 16384, and it is the size of the matrix for the test. kernel can be one of 'all', 'cpu', and 'gpu'. Without the optional arguments, the program runs the tests in both cpu and gpu for all sizes if no option is selected.

## Project Description

RoI pooling is a widely used pooling operation in object detection tasks using CNN to perform max pooling on inputs of non-uniform sizes to obtain fixed-size feature maps. More details could be found on https://towardsdatascience.com/region-of-interest-pooling-f7c637f409af.

Our project is an implementation of Region-of-Interest (RoI) pooling for sparse matrices. There exist implementations of RoI pooling for dense matrices and implementations of CNN for sparse matrices. However, there is no CPU and GPU implementation of RoI pooling for sparse matrices as of our knowledge. We focus on max pooling as it allows us avoid densify the output matrix in the way average pooling would do.

By running ```./run_tests```, we first run sparse RoI pooling on some test examples that can be found in ```test_utils.cpp```. Then, random sparse matrices and RoI boxes are generated for multiple sizes, and the sparse RoI pooling results are compared for the CPU output and GPU output. The result is described in the following section.

We initially attempted to integrate our implementation to SparseConvNet developed by Facebook as demonstrated in the initial cpu_demo_2021_submission.zip. Yet, SparseConvNet was a framework built for conventional convolutional and pooling operations for fixed sized kernels, resulting in limitations of usage. Thus, we decided to build the sparse RoI pooling functionalities from scratch. The code can also be found on https://github.com/yongkyunlee/cs179-sparse-RoI-pooling.

In particular, we represent our sparse matrices in two lists, a locations list
(image, channel, height, width) and a features list (float value) to take
advantage of the fact that most of the matrix is zeros. Our CPU approach iterates
over the input lists and stores the maximum of our outputs in a hash map,
which is then read into two output lists of similar formats.

For our CUDA approach, we initially attempted adapted a CUDA-based hash map to
create an equivalent implementation of the CPU algorithm, but parallelized over
the input list. However, we found that due to the parallel nature of the writes,
values that were not actual maximums of the RoI box were overwriting maximums
if they were viewed in the same clock cycle. Thus, instead, we switched have
each thread write only to a single coordinate in the dense output matrix (which
was generally quite small and dense regardless), and iterate over the input
list (which added a lot of time to our implementation), taking the maximum of that
input and our output cell if the input belonged in that cell's RoI box.

## Results

For the test examples, the sparse RoI pooling output matches the expected answer for both CPU and GPU implemenations. "Is answer correct: 1" is printed when ```run_tests``` is executed.

For random matrices of larger size, we first see that the outputs of CPU and GPU implementations match. Also, we observe that GPU implementation decreases the runtime to about one third compared to the CPU implemenation. Note that this result is observed when there are 128 RoI boxes, which is the default configuration for our test script, and was the number of RoI boxes used in Fast RCNN, the paper that introduced RoI pooling. More specific details on the comparison of performance for diverse cases will be explained in the next section.

The history of experiments can be found in ```results.txt```.

## Performance Analysis

### How much better is the GPU version?

The GPU version generally takes a third of the time the CPU implementation takes. This is mainly due to the fact that it can parallelize over the RoI boxes, wheras the CPU implementation iterates over them in linear time. When there are 2 RoI boxes instead of 128, we find that the GPU implementation is actually 5 times slower. We also see that the ratio of speed does not vary if the size of the input and sparsity ratio change, mainly because both the GPU and CPU vary linearly in time with those factors.

### Are there things that could be improved?

Further improvements can be made to parallelize more operations on the GPU. For instance, we can essentially formulate the GPU implementation as a set of maximum reduction, with each reduction occurring over only a subset of the input. By doing this, we can use a tree-based reduction that speed up the implementation and iterate over the input list in O(log n) time instead of O(n) time.

Finally, we could write the current maximum values to shared memory first before writing the final maximum value to global memory.