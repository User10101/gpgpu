#include <iostream>
#include <cstdlib>
#include <chrono>
#include <cmath>

unsigned long long size = (1L << 20); 

void fill_array(int *array, unsigned long long size)
{
  for (unsigned long long i = 0; i < size; ++i) {
    array[i] = rand();
  }
}

int main()
{
  std::cout << log(size)/log(2) << std::endl;
  auto start = std::chrono::steady_clock::now();
  int *array = new int[size];
  fill_array(array, size);
  
  int *dev_array;
  cudaMalloc((void **)&dev_array, size * sizeof(int));
  cudaMemcpy(dev_array, array, size * sizeof(int), cudaMemcpyHostToDevice);
  auto end = std::chrono::steady_clock::now();
  free(array);
  cudaFree(dev_array);
  auto duration = std::chrono::duration<double, std::milli>(end - start).count();
  std::cout << "Execution time: " << duration << std::endl;
  return 0;
}
