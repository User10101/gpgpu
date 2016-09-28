#include <cuda.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <malloc.h>
#include <math.h>

struct vec3d
{
  float x;
  float y;
  float z;
};

__global__ void strInit(struct vec3d *a)
{
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  
  a[i].x = i;
  a[i].y = i / 2;
  a[i].z = i + 1;
}

__global__ void strLength(struct vec3d *v, float *length)
{
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  length[i] = sqrtf(v[i].x*v[i].x + v[i].y*v[i].y + v[i].z*v[i].z);
}

int main(int argc, char *argv[])
{
  struct vec3d *v;
  struct vec3d *hv;
  float *length, *hlength;
  
  if (argc < 3) {
    fprintf(stderr, "USAGE:prog <blocks> <threads>\n");
    return 1;
  }

  int num_of_blocks = atoi(argv[1]);
  int threads_per_block = atoi(argv[2]);
  int N = num_of_blocks*threads_per_block;

  cudaMalloc((void **)&v, N*sizeof(struct vec3d));
  cudaMalloc((void **)&length, N*sizeof(float));
  hv = (struct vec3d *)calloc(N, sizeof(struct vec3d));
  hlength = (float *)calloc(N, sizeof(float));
  
  strInit<<<num_of_blocks, threads_per_block>>>(v);
  cudaThreadSynchronize();

  cudaEvent_t start, stop;
  float elapsedTime;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);
  strLength<<<num_of_blocks, threads_per_block>>>(v, length);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  fprintf(stderr, "gTest took %g\n", elapsedTime);

  cudaMemcpy(hlength, length, N*sizeof(float), cudaMemcpyDeviceToHost);

  // for (int i = 0; i < N; ++i) {
  //   printf("%g\t%g\t%g\n", ha[i], hb[i], hc[i]);
  // }

  cudaFree(v);
  cudaFree(length);
  free(hv);
  free(hlength);

  return 0;
}
