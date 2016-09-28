#include <cuda.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <malloc.h>

__global__ void gInit(float *a, int N, int stride)
{
  int thread_id = threadIdx.x + blockDim.x*blockIdx.x;

  unsigned int seed = thread_id;
  curandState s;
  curand_init(seed, 0, 0, &s);
  
  for (int i = thread_id; i < N; i += blockDim.x*gridDim.x) {
    a[i] = curand_uniform(&s);
  }
}

__global__ void gCpy(float *a, float *b, int N, int stride)
{
  int thread_id = (threadIdx.x + blockIdx.x*blockDim.x)*stride;
  for (int i = thread_id; i < N; i += blockDim.x*gridDim.x) {
    a[i] = b[i];
  }
}

int main(int argc, char *argv[])
{
  float *a, *b;
  float *ha, *hb;
  if (argc < 4) {
    fprintf(stderr, "USAGE:prog <blocks> <threads> <stride>\n");
    return 1;
  }

  int num_of_blocks = atoi(argv[1]);
  int threads_per_block = atoi(argv[2]);
  int N = num_of_blocks*threads_per_block;
  int stride = atoi(argv[3]);

  cudaMalloc((void **)&a, N*sizeof(float));
  cudaMalloc((void **)&b, N*sizeof(float));

  ha = (float *)calloc(N, sizeof(float));
  hb = (float *)calloc(N, sizeof(float));

  float cumTime = .0;
  int n_opts = 10;
  for (int i = 0; i < n_opts; ++i) {
    gInit<<<num_of_blocks, threads_per_block>>>(a, N, stride);
    cudaThreadSynchronize();

    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    gCpy<<<num_of_blocks, threads_per_block>>>(a, b, N, stride);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cumTime += elapsedTime;
  }
  printf("%g\n", cumTime / n_opts);

  cudaMemcpy(ha, a, N*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(hb, b, N*sizeof(float), cudaMemcpyDeviceToHost);

  // for (int i = 0; i < N; ++i) {
  //   printf("%g\t%g\t%g\n", ha[i], hb[i], hc[i]);
  // }

  cudaFree(a);
  cudaFree(b);
  free(ha);
  free(hb);

  return 0;
}