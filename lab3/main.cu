#include <auxf.h>

#define MULTIPLY 0
#define MULTIPLY_TRANSPOSE 1

__global__ void transpose(float *in, float *out, int m, int n)
{
int i = threadIdx.x + blockIdx.x*blockDim.x;
int j = threadIdx.y + blockIdx.y*blockDim.y;

int arrayInIndex = j + i*n;
int arrayOutIndex = i + j*m;

out[arrayOutIndex] = in[arrayInIndex];
}

__global__ void multiply(float *a, float *b, float *c, int m, int n, int k)
{
int i = threadIdx.x + blockIdx.x*blockDim.x;
int j = threadIdx.y + blockIdx.y*blockDim.y;

double sum = .0;
for (int s = 0; s < n; ++s) {
sum += a[i*n + s]*b[s*k + j];
}

c[i*k + j] = sum;
}

__global__ void multiply_tr(float *a, float *b, float *c, int m, int n, int k)
{
int i = threadIdx.x + blockIdx.x*blockDim.x;
int j = threadIdx.y + blockIdx.y*blockDim.y;

double sum = .0;
for (int s = 0; s < n; ++s) {
sum += a[i*n + s]*b[j*n + s];
}

c[i*k + j] = sum;
}

int main(int argc, char *argv[])
{
int mode = MULTIPLY;
if (argc > 1) {
mode = atoi(argv[1]);
}

float *a, *b, *c, *tb;
float *da, *db, *dc;
  
a = (float *)malloc(m*n*sizeof(float));
b = (float *)malloc(n*k*sizeof(float));
c = (float *)malloc(m*k*sizeof(float));
  
cudaMalloc((void **)&da, m*n*sizeof(float));
cudaMalloc((void **)&db, n*k*sizeof(float));
cudaMalloc((void **)&dc, m*k*sizeof(float));
cudaMalloc((void **)&tb, n*k*sizeof(float));
fillMatrixMult(a, b, m, n, k);
 
cudaMemcpy(da, a, m*n*sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(db, b, n*k*sizeof(float), cudaMemcpyHostToDevice);

dim3 tDimGrid(n / BLOCK_SIZE, k / BLOCK_SIZE);
dim3 dimGrid(m / BLOCK_SIZE, k / BLOCK_SIZE);
dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

transpose<<<tDimGrid, dimBlock>>>(db, tb, n, k);
cudaDeviceSynchronize();

cudaEvent_t start, stop;
float elapsedTime;
cudaEventCreate(&start);
cudaEventCreate(&stop);
  
cudaEventRecord(start, 0);
if (mode == MULTIPLY_TRANSPOSE) {
multiply_tr<<<dimGrid, dimBlock>>>(da, tb, dc, m, n, k);
} else {
multiply<<<dimGrid, dimBlock>>>(da, db, dc, m, n, k);
}
cudaEventRecord(stop, 0);
cudaEventSynchronize(stop);

cudaEventElapsedTime(&elapsedTime, start, stop);
cudaMemcpy(c, dc, m*k*sizeof(float), cudaMemcpyDeviceToHost);

printf("Elapsed: %g\n", elapsedTime);
//printMatrix(c, m, k);
checkResult(c, m, k);

return 0;
}