#include <cufft.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <malloc.h>

#define BLOCK_SIZE 128
#define NX (BLOCK_SIZE * 1)
#define BATCH 1
#define pi 3.141592

__global__ void gInitData(cufftComplex *data, curandState *state)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;

  data[i].x = curand_uniform(&state[i])*8.15*cosf(2*pi*3*i/NX) + curand_uniform(&state[i])*6.75*sinf(2*pi*5*i/NX);
  data[i].y = 0.0f;
}

__global__ void init_stuff(curandState *state)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  curand_init(1337, idx, 0, &state[idx]);
}

int main()
{
  cufftHandle plan;
  cufftComplex *data;
  cufftComplex *data_h = (cufftComplex *)calloc(NX, sizeof(cufftComplex));

  cudaMalloc((void **)&data, sizeof(cufftComplex) * NX * BATCH);
  if (cudaGetLastError() != cudaSuccess) {
    fprintf(stderr, "Cuda error: Failed to allocate\n");
    return -1;
  }

  curandState *d_state;
  cudaMalloc(&d_state, NX);
  init_stuff<<<(NX)/(BLOCK_SIZE), BLOCK_SIZE>>>(d_state);
  
  gInitData<<<(NX)/(BLOCK_SIZE), BLOCK_SIZE>>>(data, d_state);
  cudaDeviceSynchronize();

  // cudaMemcpy(data_h, data, NX * BATCH * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
  // for (size_t i = 0; i < NX; ++i) {
  //   printf("%f\t%f\n", data_h[i].x, data_h[i].y);
  // }
  // printf("\n");
  
  if (cufftPlan1d(&plan, NX, CUFFT_C2C, BATCH) != CUFFT_SUCCESS) {
    fprintf(stderr, "CUFFT error: Plan creation failed");
    return -1;
  }

  if (cufftExecC2C(plan, data, data, CUFFT_FORWARD) != CUFFT_SUCCESS) {
    fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
    return -1;
  }

  if (cudaDeviceSynchronize() != cudaSuccess) {
      fprintf(stderr, "Cuda error: Failed to syncrhonize\n");
      return -1;
  }

  cudaMemcpy(data_h, data, NX * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < NX/2; ++i) {
    printf("%d\t%f\n", i, sqrt(pow(data_h[i].x, 2) + pow(data_h[i].y, 2)));
  }

  cufftDestroy(plan);
  cudaFree(d_state);
  cudaFree(data);
  free(data_h);

  return 0;
}
