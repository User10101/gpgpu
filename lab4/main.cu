#include <cuda.h>
#include <stdio.h>
#include <malloc.h>

#define BLOCK_SIZE 16

__global__ void gMultiply(float *a, float *b, float *c, int m, int n, int k)
{
  int aBegin = n * BLOCK_SIZE * blockIdx.y;
  int aEnd = aBegin + n - 1;
  int aStep = BLOCK_SIZE;

  int bBegin = BLOCK_SIZE * blockIdx.x;
  int bStep = BLOCK_SIZE * k;
  float sum = .0f;

  for (int ia = aBegin, ib = bBegin; ia < aEnd; ia += aStep, ib += bStep) {
    __shared__ float aSub[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float bSub[BLOCK_SIZE][BLOCK_SIZE];

    aSub[threadIdx.y][threadIdx.x] = a[ia + n*threadIdx.y + threadIdx.x];
    bSub[threadIdx.y][threadIdx.x] = b[ib + k*threadIdx.y + threadIdx.x];
    __syncthreads();

    for (int s = 0; s < BLOCK_SIZE; ++s) {
      sum += aSub[threadIdx.y][s] * bSub[s][threadIdx.x];
    }
    __syncthreads();
  }

  c[k*BLOCK_SIZE*blockIdx.y + threadIdx.y*k + blockIdx.x*BLOCK_SIZE + threadIdx.x] = sum;
}

void fillMatrixMult(float *a, float *b, int m, int n, int k)
{
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      a[j + i*n] = j*i;
    }
  }

  for (int i = 0; i < k; ++i) {
    for (int j = 0; j < n; ++j) {
      b[j + i*k] = (i == j) ? 1 : 0;
    }
  }
}

int main(int argc, char *argv[])
{  
  int m = BLOCK_SIZE*2, n = BLOCK_SIZE*3, k = BLOCK_SIZE*4;
  float *a, *b, *c;
  float *da, *db, *dc;
  
  a = (float *)malloc(m*n*sizeof(float));
  b = (float *)malloc(n*k*sizeof(float));
  c = (float *)malloc(m*k*sizeof(float));
  
  cudaMalloc((void **)&da, m*n*sizeof(float));
  cudaMalloc((void **)&db, n*k*sizeof(float));
  cudaMalloc((void **)&dc, m*k*sizeof(float));
  fillMatrixMult(a, b, m, n, k);
 
  cudaMemcpy(da, a, m*n*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(db, b, n*k*sizeof(float), cudaMemcpyHostToDevice);

  dim3 dimGrid(m / BLOCK_SIZE, k / BLOCK_SIZE);
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

  cudaEvent_t start, stop;
  float elapsedTime;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  cudaEventRecord(start, 0);
  gMultiply<<<dimGrid, dimBlock>>>(da, db, dc, m, n, k);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
 
  cudaEventElapsedTime(&elapsedTime, start, stop);
  cudaMemcpy(c, dc, m*k*sizeof(float), cudaMemcpyDeviceToHost);

  printf("Elapsed: %g\n", elapsedTime);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < k; ++j) {
      printf("%g ", c[j + i*k]);
    } 
    printf("\n");
  }
  
  return 0;
}