#ifdef __CUDACC__
#include <cuda.h>
#endif

#include <stdio.h>
#include <malloc.h>
#include <cmath>
#include <cstdlib>

#define BLOCK_SIZE 32

int m = BLOCK_SIZE*32, n = BLOCK_SIZE*32, k = BLOCK_SIZE*32;

void fillMatrixMult(float *a, float *b, int m, int n, int k)
{
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      a[j + i*n] = j + i;
    }
  }

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < k; ++j) {
      b[j + i*k] = (i == j) ? 1 : 0;
    }
  }
}

void checkResult(float *c, int m, int k)
{
  float tolerance = 1e-5;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < k; ++j) {
      if (abs(i + j - c[j + i*k]) > tolerance) {
	printf("Error!!!\n");
	exit(-1);
      }
    }
  }

  printf("\n Result is correct\n");
}


void printMatrix(float *mat, int m, int n)
{
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      printf("%g ", mat[j + i*n]);
    } 
    printf("\n");
  }
}
