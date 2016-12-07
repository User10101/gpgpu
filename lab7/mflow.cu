#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>

#include <fstream>
#include <iostream>
#include <chrono>

#define PI 3.14159
#define A 0.
#define B 3.
#define M 32
#define HV ((B - A) / M)

class Norm_distr
{
public:
  Norm_distr(double sigma, size_t size, int speeds)
    : s(sigma), coef(10./size), n_speeds(speeds) {}

  __host__ __device__ double operator()(double x)
  {
    x = (int)(x / n_speeds);
    x *= coef;
    return 1./s/sqrt(2*PI)*exp(-(x - 5.)*(x - 5.)/2./s/s);
  }

private:
  double s;
  double coef;
  int n_speeds;
};

class Init_distribution
{
public: 
  Init_distribution(double v, double T)
    :_v(v), _T(T) {}
  __host__ __device__ double operator() (double r, double u)
  {
    return r*exp((u - _v)*(u - _v)/2/_T)*sqrt(2*PI*_T);
  }
private:
  double _v, _T;
};

class Upwind_functor
{
public:
  Upwind_functor(double t, double hv) : tau(t), h(hv) {}

  __host__ __device__ thrust::tuple<double, double, double> operator()(thrust::tuple<double, double, double> t) const {
    double a1 = thrust::get<0>(t);
    double a2 = thrust::get<1>(t);
    double u = thrust::get<2>(t);
    return thrust::make_tuple(a1 + u*tau*(a2 - a1)/h, a2, u);
  }
  
private:
  double tau;
  double h;
};

class Speed_functor
{
public:
  Speed_functor(int speeds) : n_speeds(speeds) {}
  
  __host__ __device__ double operator()(double x)
  {
    int i = (int)x % n_speeds;
    return ((i + 0.5) * HV - A);
  }
  
private:
  int n_speeds;
};

void print(const thrust::host_vector<double> &v)
{
  for (size_t i = 4; i < v.size(); i += M) {
    printf("%lf ", v[i]);
  }
  printf("\n");
}

int main(int argc, char *argv[])
{
  auto start = std::chrono::steady_clock::now();
  int size = (1 << 4);
  if (argc > 1) {
    size = atoi(argv[1]);
  }

  std::cout << size << "\n";
  
  Norm_distr d(0.8, size, M);
  thrust::device_vector<double> d1(size * M);
  thrust::device_vector<double> d2(size * M);
  thrust::sequence(thrust::device, d1.begin(), d1.end());
  thrust::transform(thrust::device, d1.begin(), d1.end(), d1.begin(), d);

  Speed_functor c(M);
  thrust::device_vector<double> u(size * M);
  thrust::sequence(thrust::device, u.begin(), u.end());
  thrust::transform(thrust::device, u.begin(), u.end(), u.begin(), c);

  Init_distribution id(1. /10., 1);
  thrust::transform(thrust::device, d1.begin(), d1.end(), u.begin(), d1.begin(), id);
  thrust::host_vector<double> hv(size * M);

  thrust::host_vector<double> con(hv.size());
  thrust::host_vector<double> a(hv.size());
  std::ofstream ost;
  thrust::copy(d1.begin(), d1.end(), hv.begin());
  thrust::copy(hv.begin(), hv.end(), con.begin());
  thrust::copy(u.begin(), u.end(), a.begin());
  
  print(hv);
  ost.open("plotdata.dat");
  for (size_t i = 0; i < size; ++i) {
    for (size_t j = 0; j < M; ++j) {
      ost << hv[i * M + j] << ' ';
    }
    ost << '\n';
  }
  ost << "\n\n";
  
  Upwind_functor uf(1./50., HV);
  for (size_t i = 0; i < 70; ++ i) {
    for (int j = 0; j < con.size() - M; ++j) {
      con[j] += a[j]*(con[j + M] - con[j])*(1./50.)/HV;
      if (a[j] != a[j + M]) {
	std::cout << "Error1\n";
	exit(-1);
      }
      if (a[j]*(1./50.)/HV >= 1 || a[j]*(1./50.)/HV <= 0) {
	std::cout << "Error2\n";
	exit(-1);
      }
    }
    thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(d1.begin(), d1.begin() + M, u.begin())),
		      thrust::make_zip_iterator(thrust::make_tuple(d1.end() - M, d1.end(), u.end() - M)),
		      thrust::make_zip_iterator(thrust::make_tuple(d2.begin(), d1.begin() + M, u.begin())),
		      uf);
    thrust::copy(d2.begin(), d2.end(), d1.begin());
    thrust::copy(d1.begin(), d1.end(), hv.begin());
    if (i == 0) {
      for (size_t j = 0; j < hv.size(); ++j) {
	if (fabs(con[j] - hv[j]) > 1e-4) {
	  std::cout << "Error: " << con[j] << ' ' << hv[j] << "\n";
	}
      }
    }
    for (size_t i = 0; i < size; ++i) {
      for (size_t j = 0; j < M; ++j) {
	ost << hv[i * M + j] << ' ';
	if (hv[i * M + j] < 0) {
	  std::cout << "Error!!!" << hv[i * M + j] << "\n";
	  ost.close();
	  exit(-1);
	}
      }
      ost << '\n';
    }
    ost << "\n\n";
  }
  ost.close();
  auto end = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration<double, std::milli>(end - start).count();
  std::cout << "Execution time: " << duration << std::endl;
  
  return 0;
}
