#include <stdio.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */


#define CUDA_SAFE_CALL(err) {							\
    if (cudaSuccess != err)	{						\
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.",	\
                __FILE__, __LINE__, cudaGetErrorString(err) );	\
        exit(EXIT_FAILURE);						\
    }									\
}

#define CUDA_CHECK_ERROR() {							\
    cudaError_t err = cudaGetLastError();					\
    if (cudaSuccess != err) {						\
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.",	\
                __FILE__, __LINE__, cudaGetErrorString(err) );	\
        exit(EXIT_FAILURE);						\
    }									\
}

using namespace std;

struct Entry {
    float value;
    int i, j;
};

const int MAX_THREADS   = 512;


//dim < 10000 => random wrong results
__global__ void MaxReductionKernel(float* _dst, const float* _src, const int dim)
{
    extern __shared__ volatile float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + tid;
    unsigned int gridSize = blockDim.x*gridDim.x;

    float val = _src[i];
    i += gridSize;
    while (i < dim) {
        val = fmax(_src[i],val);
        i += gridSize;
    }
    sdata[tid] = val;
    __syncthreads();

    // This versions uses a single warp for the shared memory 
    // reduction
# pragma unroll
    for(int i=(tid+32); ((tid<32)&&(i<blockDim.x)); i+=32)
        sdata[tid] = fmax(sdata[tid], sdata[i]);

    if (tid < 16) sdata[tid] = fmax(sdata[tid], sdata[tid + 16]);
    if (tid < 8)  sdata[tid] = fmax(sdata[tid], sdata[tid + 8]);
    if (tid < 4)  sdata[tid] = fmax(sdata[tid], sdata[tid + 4]);
    if (tid < 2)  sdata[tid] = fmax(sdata[tid], sdata[tid + 2]);
    if (tid == 0) _dst[blockIdx.x] = fmax(sdata[tid], sdata[tid + 1]);

}
// this kernel computes, per-block, the sum
// of a block-sized portion of the input
// using a block-wide reduction
//__global__ void block_sum(const float *input,
//                          float *per_block_results,
//                          const size_t n)
//{
//  extern __shared__ float sdata[];
//
//  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
//
//  // load input into __shared__ memory
//  float x = 0;
//  if(i < n)
//  {
//    x = input[i];
//  }
//  sdata[threadIdx.x] = x;
//  __syncthreads();
//
//  // contiguous range pattern
//  for(int offset = blockDim.x / 2;
//      offset > 0;
//      offset >>= 1)
//  {
//    if(threadIdx.x < offset)
//    {
//      // add a partial sum upstream to our own
//      sdata[threadIdx.x] += sdata[threadIdx.x + offset];
//    }
//
//    // wait until all threads in the block have
//    // updated their partial sums
//    __syncthreads();
//  }
//
//  // thread 0 writes the final result
//  if(threadIdx.x == 0)
//  {
//    per_block_results[blockIdx.x] = sdata[0];
//  }
//}
#if 0
__global__ void MaxReductionKernel2(float* _dst, float* _src, const int dim)
{
    extern __shared__ volatile float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + tid;
    unsigned int gridSize = blockDim.x*gridDim.x;

    float val = _src[i];
    int maxIndex = -1;
    int arrayIndex1 = i;
    i += gridSize;
    while (i < dim) {
        int arrayIndex2 = i;
        if (_src[i] > val) 
        {
            val = _src[i];
            maxIndex = arrayIndex2;
        }
        else
        {
            maxIndex = arrayIndex1;
        }
        //val = fmax(_src[i],val);
        i += gridSize;
    }
    sdata[tid] = val;
    __syncthreads();

    // This versions uses a single warp for the shared memory 
    // reduction
# pragma unroll
    for(int i=(tid+32); ((tid<32)&&(i<blockDim.x)); i+=32)
    {
        if (sdata[tid] > sdata[i])
        {
            maxIndex = tid + blockIdx.x*blockDim.x;
        }
        else
        {
            sdata[tid] = sdata[i];
            maxIndex = i;
        }
        //sdata[tid] = fmax(sdata[tid], sdata[i]);
    }

    if (tid < 16)
    {
        if (sdata[tid] > sdata[tid+16])
        {
            maxIndex = tid + blockIdx.x*blockDim.x;
        }
        else
        {
            sdata[tid] = sdata[tid+16];
            maxIndex = (tid+16) + blockIdx.x*blockDim.x;
        }
        //sdata[tid] = fmax(sdata[tid], sdata[tid + 16]);
    }
    if (tid < 8)
    {
        if (sdata[tid] > sdata[tid+8])
        {
            maxIndex = tid + blockIdx.x*blockDim.x;
        }
        else
        {
            sdata[tid] = sdata[tid+8];
            maxIndex = (tid+8) + blockIdx.x*blockDim.x;
        }
        //sdata[tid] = fmax(sdata[tid], sdata[tid + 8]);
    }
    if (tid < 4)
    {
        if (sdata[tid] > sdata[tid+4])
        {
            maxIndex = tid + blockIdx.x*blockDim.x;
        }
        else
        {
            sdata[tid] = sdata[tid+4];
            maxIndex = (tid+4) + blockIdx.x*blockDim.x;
        }
        //sdata[tid] = fmax(sdata[tid], sdata[tid + 4]);
    }
    if (tid < 2)
    {
        if (sdata[tid] > sdata[tid+2])
        {
            maxIndex = tid + blockIdx.x*blockDim.x;
        }
        else
        {
            sdata[tid] = sdata[tid+2];
            maxIndex = (tid+2) + blockIdx.x*blockDim.x;
        }
        //sdata[tid] = fmax(sdata[tid], sdata[tid + 2]);
    }
    if (tid == 0)
    {
        if (sdata[tid] > sdata[tid+1])
        {
            _dst[blockIdx.x] = sdata[tid];
            maxIndex = tid + blockIdx.x*blockDim.x;
        }
        else
        {
            _dst[blockIdx.x] = sdata[tid+1];
            maxIndex = (tid+1) + blockIdx.x*blockDim.x;
        }
        //_dst[blockIdx.x] = fmax(sdata[tid], sdata[tid + 1]);
    }
}
#endif
int main(int argc, char* argv[]) {
    if (argc != 5)
    {
        printf("usage: findMax <dim> <max> <n-biggest> <mode[0: CPU, 1: GPU]>");
        exit(1);
    }

    int dim = atoi(argv[1]);
    printf("DIMENSION: %i",dim);
    int max = atoi(argv[2]);
    int nBiggest = atoi(argv[3]);

    Entry* nBiggestEntries = new Entry[nBiggest]; //list of nbiggest entries

    //initialize random matrix and nBiggestEntries list
    float* matrix = new float[dim]; //2d matrix
    srand(time(NULL));
    for (int i = 0; i < dim; i++)
    {
        //int iRand = rand() % max;
        int iRand = i;
        matrix[i] = iRand;
        //initialize nBiggestEntries list on the fly
        if (0 == i)
        {
            for (int j = 0; j < nBiggest; j++)
            {
                nBiggestEntries[j].value = -1;
                nBiggestEntries[j].i = -1;
                nBiggestEntries[j].j = -1;
            }
        }
        if (dim < 100) printf("%i ",iRand);
    }
    printf("\n");

    if (0 == atoi(argv[4])) //CPU
    {
        printf("Executing CPU code\n");
        //linear search
        for (int i = 0; i < dim; i++)
        {
            float value = matrix[i];

            int findMin = 0;
            float findMinValue = nBiggestEntries[0].value;
            for (int j = 0; j < nBiggest; j++)
            {
                if (findMinValue > nBiggestEntries[j].value)
                {
                    findMinValue = nBiggestEntries[j].value;
                    findMin = j;
                }
            }

            if (value > findMinValue)
            {
                nBiggestEntries[findMin].value = value;
                nBiggestEntries[findMin].i = i/dim;
                nBiggestEntries[findMin].j = i%dim;
            }
        }
    } 
    else //GPU
    {
        printf("Executing GPU code\n");
        //reduction
        int numBlocks = (dim+MAX_THREADS-1)/MAX_THREADS;
        dim3 blockGrid(numBlocks);
        dim3 threadBlock(MAX_THREADS);
        float* gpuResult1;
        float* gpuResult2;
        float* cpuResult = new float[1];
        cudaMalloc((void**) &gpuResult1, dim*sizeof(float));
        cudaMalloc((void**) &gpuResult2, numBlocks*sizeof(float));
        cudaMemcpy(gpuResult1, matrix, dim*sizeof(float), cudaMemcpyHostToDevice);

        MaxReductionKernel<<<blockGrid,threadBlock, MAX_THREADS*sizeof(float)>>>(gpuResult2, gpuResult1, dim);
        CUDA_CHECK_ERROR();

        MaxReductionKernel<<<1, numBlocks, numBlocks*sizeof(float)>>>(gpuResult1, gpuResult2, numBlocks);
        CUDA_CHECK_ERROR();

        //block_sum<<<numBlocks,MAX_THREADS, MAX_THREADS*sizeof(float)>>>(gpuResult1, gpuResult2, dim);
        //block_sum<<<1,numBlocks, numBlocks*sizeof(float)>>>(gpuResult2, gpuResult1, numBlocks);

        cudaMemcpy(cpuResult, gpuResult1, sizeof(float), cudaMemcpyDeviceToHost);

        float max = cpuResult[0];


        cudaFree(gpuResult1);
        cudaFree(gpuResult2);
        printf("max: %f\n",max);
    }

    printf("%i biggest entries:\n",nBiggest);
    for (int i = 0; i < nBiggest; i++)
    {
        printf("%i: %f at [%i,%i]\n",i,nBiggestEntries[i].value,nBiggestEntries[i].i,nBiggestEntries[i].j);
    }

}
