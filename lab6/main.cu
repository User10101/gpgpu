#include <iostream>

#define N 100000

int main(int argc, char *argv[])
{
  bool async_mode = false;
  if (argc > 1) {
    async_mode = true;
  }
  float *adev;
  cudaMalloc((void **)&adev, N * sizeof(float));

  if (!async_mode) {
    float *a = (float *)malloc(N * sizeof(float));
    cudaMemcpy(adev, a, N * sizeof(float), cudaMemcpyHostToDevice);
    free(a);
  } else {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    float *ap;
    cudaHostAlloc((void **)&ap, N * sizeof(float), cudaHostAllocDefault);
    cudaMemcpyAsync(adev, ap, N * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    cudaFreeHost(ap);
  }
  cudaFree(adev);
}