#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>

#define BLOCK_SIZE 256

enum Kernels{SHARED = 0, GLOBAL = 1};

__global__ void dot(float *v1, float *v2, float *psums)
{
  __shared__ float data[BLOCK_SIZE];
  int tid = threadIdx.x;
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  data[tid] = v1[i]*v2[i];
  __syncthreads();
  for (int s = 1; s < blockDim.x; s <<= 1) {
    if (tid%(2*s) == 0) {
      data[tid] += data[tid + s];
    }

    __syncthreads();
  }

  if (0 == tid) {
    psums[blockIdx.x] = data[0];
  }

  if (i == 0) {
    printf("Done kernel with shared memory\n");
  }
}

__global__ void gdot(float *v1, float *v2, float *psums, float *tmp)
{
  int tid = threadIdx.x;
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  int block_shift = blockDim.x*blockIdx.x;
  tmp[i] = v1[i]*v2[i];
  __syncthreads();
  for (int s = 1; s < blockDim.x; s <<= 1) {
    if (tid%(2*s) == 0) {
      tmp[block_shift + tid] += tmp[block_shift + tid + s];
    }

    __syncthreads();
  }

  if (0 == tid) {
    psums[blockIdx.x] = tmp[block_shift];
  }

  if (i == 0) {
    printf("Done kernel with global memory\n");
  }
}

int main(int argc, char *argv[])
{
  int kernel = SHARED;
  if (argc > 1) {
    int opt_kernel = atoi(argv[1]);
    if (opt_kernel == SHARED || opt_kernel == GLOBAL) {
      kernel = opt_kernel;
    } else {
      printf("Unknown kernel type '%d'. Possible values are 0 (SHARED, default) and 1 (GLOBAL).\n", opt_kernel);
      return -1;
    }
  }
  
  const size_t n = BLOCK_SIZE*20;
  size_t num_blocks = n / BLOCK_SIZE;
  float *v1, *v2;
  float *v1_d, *v2_d, *tmp_d;
  float *psums;
  v1 = (float *)malloc(n * sizeof(float));
  v2 = (float *)malloc(n * sizeof(float));
  cudaMalloc((void **)&v1_d, n * sizeof(float));
  cudaMalloc((void **)&v2_d, n * sizeof(float));
  cudaMalloc((void **)&psums, num_blocks * sizeof(float));
  for (size_t i = 0; i < n; ++i) {
    v1[i] = (float)i * 0.0001;
    v2[i] = (float)i * 0.0001;
  }
  cudaMemcpy(v1_d, v1, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(v2_d, v2, n * sizeof(float), cudaMemcpyHostToDevice);
  if (kernel == GLOBAL) {
    cudaMalloc((void **)&tmp_d, n*sizeof(float));
    gdot<<<dim3(num_blocks), dim3(BLOCK_SIZE)>>>(v1_d, v2_d, psums, tmp_d);
  } else {
    dot<<<dim3(num_blocks), dim3(BLOCK_SIZE)>>>(v1_d, v2_d, psums);
  }
  cudaThreadSynchronize();
  float *psums_host = (float *)malloc(num_blocks*sizeof(float));
  cudaMemcpy(psums_host, psums, num_blocks * sizeof(float), cudaMemcpyDeviceToHost);
  float res = .0f;
  for (size_t i = 0; i < num_blocks; ++i) {
    res += psums_host[i];
  }
  free(v1);
  free(v2);
  free(psums_host);
  cudaFree(psums);
  if (kernel == GLOBAL) {
    cudaFree(tmp_d);
  }

  printf("%f\n", res);
  return 0;
}