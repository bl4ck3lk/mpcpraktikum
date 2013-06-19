#include <stdio.h>
#include <assert.h>
//#include <cv.h>

#include <vector>


struct DMatch
{
	int distance;
};

// Simple utility function to check for CUDA runtime errors
void checkCUDAError(const char* msg);

// Part 2 of 2: implement the kernel
__global__ void checkRatio(float *dst, const float *src, const int _dim)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < _dim) {
		const DMatch d1 = src.at(i);

	//	if ((d1.distance / d2.distance) < 0.85f) {
	//		dst.insert(i, src[i]);
	//	}
	}
	 
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{

	std::vector< DMatch > *matches;		
	std::vector< DMatch > *cleanMatches;

	int numMatches = 200;
	// pointer for host memory and size
	//int *h_a;
	int dimA = numMatches; // 256K elements (1MB total)

	// pointer for device memory
	std::vector<DMatch> *d_matches, *d_cleanMatches;

	// define grid and block size
	int numThreadsPerBlock = 256;

	// Part 1 of 2: compute number of blocks needed based on array size and desired block size
	int numBlocks;
	// Compute the number of blocks needed.
	numBlocks = dimA / numThreadsPerBlock;
	// allocate host and device memory
	size_t memSize = numBlocks * numThreadsPerBlock * sizeof(int);
	matches = (std::vector<DMatch> *) malloc(memSize);
	cudaMalloc((void **) &d_matches, memSize);
	cudaMalloc((void **) &d_cleanMatches, memSize);

	// Initialize input array on host
	for (int i = 0; i < dimA; ++i)
	{
		DMatch m;
		m.distance = 1;
		matches[i].push_back(m);
	}

	// Copy host array to device array
	cudaMemcpy(d_matches, matches, memSize, cudaMemcpyHostToDevice);

	// launch kernel
	dim3 dimGrid(numBlocks);
	dim3 dimBlock(numThreadsPerBlock);
	checkRatio<<< dimGrid, dimBlock, numThreadsPerBlock * sizeof(std::vector<DMatch>) >>>( d_cleanMatches, d_matches, dimA );

	// block until the device has completed
	cudaThreadSynchronize();

	// check if kernel execution generated an error
	// Check for any CUDA errors
	checkCUDAError("kernel invocation");

	// device to host copy
	cudaMemcpy(matches, d_cleanMatches, memSize, cudaMemcpyDeviceToHost);

	// Check for any CUDA errors
	checkCUDAError("memcpy");

	// verify the data returned to the host is correct
	//for (int i = 0; i < dimA; i++)
	//{
	//	assert(matches[i] == dimA - 1 - i);
	//}

	// free device memory
	cudaFree(d_matches);
	cudaFree(d_cleanMatches);

	// free host memory
	free(matches);

	// If the program makes it this far, then the results are correct and
	// there are no run-time errors.  Good work!
	printf("Correct!\n");

	return 0;
}

void checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}
