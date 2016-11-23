#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <cstdlib>
#include <chrono>

unsigned long long size = (1L << 20); 

int main()
{
  auto start = std::chrono::steady_clock::now();
  thrust::host_vector<int> h_vec(size);
  thrust::generate(h_vec.begin(), h_vec.end(), rand);
  thrust::device_vector<int> d_vec = h_vec;
  auto end = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration<double, std::milli>(end - start).count();
  std::cout << "Execution time: " << duration << std::endl;
  return 0;
}
