#include <auxf.h>

 #include <iostream>
#include <chrono>

int main()
{
  float *a, *b, *c;

  a = (float *)malloc(m*n*sizeof(float));
  b = (float *)malloc(n*k*sizeof(float));
c = (float *)calloc(m*k, sizeof(float));

  fillMatrixMult(a, b, m, n, k);

auto start = std::chrono::steady_clock::now();
for (int i = 0; i < m; ++i) {
for (int j = 0; j < k; ++j) {
for (int s = 0; s < n; ++s) {
c[i*k + j] += a[i*n + s]*b[s*k + j];
}
}
}
auto end = std::chrono::steady_clock::now();
 auto diff = end - start;
 std::cout << "Elapsed: " << std::chrono::duration<double, std::milli>(diff).count()  << std::endl;
 checkResult(c, m, k);

 free(a);
 free(b);
 free(c);
 
  return 0;
}
