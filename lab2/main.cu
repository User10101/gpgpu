#include <cuda.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <malloc.h>

__global__ void gInit(float *a, float *b, int N, int offset)
{
  int thread_id = threadIdx.x + blockDim.x*blockIdx.x + offset;

  unsigned int seed = thread_id;
  curandState s;
  curand_init(seed, 0, 0, &s);
  
  for (int i = thread_id; i < N + offset; i += blockDim.x*gridDim.x) {
    a[i] = curand_uniform(&s);
    b[i] = curand_uniform(&s);
  }
}

__global__ void gSum(float *a, float *b, float *c, int N, int offset)
{
  int thread_id = threadIdx.x + blockIdx.x*blockDim.x + offset;
  for (int i = thread_id; i < N + offset; i += blockDim.x*gridDim.x) {
    c[i] = a[i] + b[i];
  }
}

int main(int argc, char *argv[])
{
  float *a, *b, *c;
  float *ha, *hb, *hc;
  if (argc < 4) {
    fprintf(stderr, "USAGE:prog <blocks> <threads> <offset>\n");
    return 1;
  }

  int num_of_blocks = atoi(argv[1]);
  int threads_per_block = atoi(argv[2]);
  int N = num_of_blocks*threads_per_block;
  int offset = atoi(argv[3]);

  cudaMalloc((void **)&a, (N + offset)*sizeof(float));
  cudaMalloc((void **)&b, (N + offset)*sizeof(float));
  cudaMalloc((void **)&c, (N + offset)*sizeof(float));

  ha = (float *)calloc(N + offset, sizeof(float));
  hb = (float *)calloc(N + offset, sizeof(float));
  hc = (float *)calloc(N + offset, sizeof(float));

  float cumTime = .0;
  int n_opts = 10;
  for (int i = 0; i < n_opts; ++i) {
    gInit<<<num_of_blocks, threads_per_block>>>(a, b, N, offset);
    cudaThreadSynchronize();

    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    gSum<<<num_of_blocks, threads_per_block>>>(a, b, c, N, offset);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cumTime += elapsedTime;
  }
  printf("%g\n", cumTime / n_opts);

  cudaMemcpy(ha, a, (N + offset)*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(hb, b, (N + offset)*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(hc, c, (N + offset)*sizeof(float), cudaMemcpyDeviceToHost);

  // for (int i = 0; i < N; ++i) {
  //   printf("%g\t%g\t%g\n", ha[i], hb[i], hc[i]);
  // }

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
  free(ha);
  free(hb);
  free(hc);

  return 0;
}