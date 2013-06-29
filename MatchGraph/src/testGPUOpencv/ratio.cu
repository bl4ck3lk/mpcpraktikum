#include <stdio.h>
#include <vector_types.h>

__global__
void ratio(const int2 * trainIdx1, int2 * trainIdx2,
                            const float2 * distance1, const float2 * distance2, int * out)
{
 
  	int i = blockIdx.x * blockDim.x + threadIdx.x;

    int x1 = trainIdx1[i].x;
    //int y1 = trainIdx1[i].y;
    float d11 = distance1[i].x;
    float d21 = distance1[i].y;

    if ((d11 / d21) < 0.85f) {

	//printf("%d\n",i);
	out[i] = x1;
        //printf("1. Points: (%d,%d)\n", x1, y1);
        //printf("1. Distances: %f,%f\n", d11, d21);
    }
	//out[i] = x1;
/*
    int x2 = trainIdx2[i].x;
    int y2 = trainIdx2[i].y;
    float d12 = distance2[i].x;
    float d22 = distance2[i].y;

    if ((d12 / d22) < 0.85f) {
	printf("%d\n",i);
        //printf("2. Points: (%d,%d)\n", x2, y2);
        //printf("2. Distances: %f,%f\n", d12, d22);
    }
*/
}

	//int2* trainIdx1 = trainIdxMat1.ptr<int2>();
	//float2* distance1 = distanceMat1.ptr<float2>();
	//int2* trainIdx2 = trainIdxMat2.ptr<int2>();
	//float2* distance2 = distanceMat2.ptr<float2>();

void ratio_aux(const int2 * trainIdx1, int2 * trainIdx2,
                            const float2 * distance1, const float2 * distance2)
{
  printf("About to run CUDA Kernel 'ratio'!\n");
  int *d_out, *h_out;
  size_t memSize = 3000 * sizeof(int);
  h_out = (int *) malloc(memSize);
  cudaMalloc((void **) &d_out, memSize);
  for (int i=0; i<3000;++i) {
    h_out[i] = -1;
  }
  cudaMemcpy(d_out, h_out, memSize, cudaMemcpyHostToDevice);
  const int thread = 32;
  const dim3 blockSize( thread );
  const dim3 gridSize( 16, 1 );
  ratio <<< gridSize, blockSize >>>(trainIdx1, trainIdx2,
                                    distance1, distance2, d_out);

  cudaMemcpy(h_out, d_out, memSize, cudaMemcpyDeviceToHost);

  for (int i=0; i<3000;++i) {
    if (h_out[i] != -1) {printf("%d\n", h_out[i]);}
  }
}
