#include <stdio.h>
#include <vector_types.h>

__global__
void ratio(const int2 * trainIdx1, int2 * trainIdx2,
                            const float2 * distance1, const float2 * distance2)
{
 
  	int i = blockIdx.x * blockDim.x + threadIdx.x;

		int t = trainIdx1[i].x;
		//printf("%d\n", t);
		//	if ((d1.distance / d2.distance) < 0.85f) {
	//		dst.insert(i, src[i]);
	//	}

}

	//int2* trainIdx1 = trainIdxMat1.ptr<int2>();
	//float2* distance1 = distanceMat1.ptr<float2>();
	//int2* trainIdx2 = trainIdxMat2.ptr<int2>();
	//float2* distance2 = distanceMat2.ptr<float2>();

void ratio_aux(const int2 * trainIdx1, int2 * trainIdx2,
                            const float2 * distance1, const float2 * distance2)
{
  printf("About to run CUDA Kernel 'ratio'!\n");
  const int thread = 16;
  const dim3 blockSize( thread );
  const dim3 gridSize( 1, 1 , 1);
  ratio <<< gridSize, blockSize >>>(trainIdx1, trainIdx2, distance1, distance2);
}
