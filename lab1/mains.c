#include <stdio.h>
#include <malloc.h>

#define BLOCK_SIZE  256

int main()
{
  const size_t n = BLOCK_SIZE*20;
  size_t i = 0;
  float res = .0f;

  for (i = 0; i < n; ++i) {
    res = res + (float)i * 0.0001 * (float)i * 0.0001;
  }

  printf("%f\n", res);
  return 0;
}
