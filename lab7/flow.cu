#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>

#include <fstream>
#include <iostream>

// Распределение Релэя.
class Norm_distr
{
public:
  Norm_distr(double sigma, size_t size)
    : s(sigma), coef(size/10.) {}

  __host__ __device__ double operator()(double x)
  {
    x /= coef; 
    return 1./s/sqrt(2*3.14159)*exp(-(x - 5.)*(x - 5.)/2./s/s);
  }
  
private:
  double s;
  double coef;
};

class Upwind_functor
{
public:
  Upwind_functor(double c)
  : coef(c) {}

  __host__ __device__ double operator()(double a1, double a2) {
    return a1 + coef*(a2 - a1);
  }
  
private:
  double coef;
};

int main(int argc, char *argv[])
{
  int size = (1 << 7);
  if (argc > 1) {
    size = atoi(argv[1]);
  }

  std::cout << size << "\n";
  Norm_distr d(0.8, size);
  thrust::device_vector<double> d1(size);
  thrust::sequence(thrust::device, d1.begin(), d1.end());
  thrust::transform(thrust::device, d1.begin(), d1.end(), d1.begin(), d);
  thrust::host_vector<double> hv(size);
  thrust::device_vector<double> tmp(size - 1);
  Upwind_functor u(0.9);
  std::ofstream ost;
  thrust::copy(d1.begin(), d1.end(), hv.begin());
  ost.open("plotdata.dat");
  for (size_t i = 0; i < size; ++i) {
    ost << hv[i] << '\n';
  }
  ost << "\n\n";
    
  for (size_t i = 0; i < 70; ++ i) {
    thrust::transform(d1.begin(), --d1.end(), ++d1.begin(), tmp.begin(), u);
    thrust::copy(thrust::device, tmp.begin(), tmp.end(), d1.begin());
    thrust::copy(d1.begin(), d1.end(), hv.begin());
    for (size_t i = 0; i < size; ++i) {
      ost << hv[i] << '\n';
    }
    ost << "\n\n";
  }
  ost.close();
  
  return 0;
}
