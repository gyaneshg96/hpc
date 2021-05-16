#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>


#define BLOCK_SIZE 1024

void matrix_product(double* dot_ptr, const double* x, const double* y, long m, long n){
  // #pragma omp parallel for schedule(static) reduction(+:sum)
  for (long i = 0; i < m; i++){
    double sum = 0;
    for(long j = 0; j < n; j++)
      sum += x[i*n+j] *y[j];
  dot_ptr[i] = sum;
  }
}

void Check_CUDA_Error(const char *message){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
    exit(-1);
  }
}

__global__ void matrix_product_kernel(double *dotptr, const double*x, double *y, long N, int flag){
  __shared__ double smem[BLOCK_SIZE];
  int idx = (blockIdx.x) * blockDim.x + threadIdx.x;

  if (idx < N && flag == 1) smem[threadIdx.x] = x[idx]*y[idx];
  else if (idx < N && flag == 0) smem[threadIdx.x] = x[idx];
  else smem[threadIdx.x] = 0;

  __syncthreads();
  if (threadIdx.x < 512) smem[threadIdx.x] += smem[threadIdx.x + 512];
  __syncthreads();
  if (threadIdx.x < 256) smem[threadIdx.x] += smem[threadIdx.x + 256];
  __syncthreads();
  if (threadIdx.x < 128) smem[threadIdx.x] += smem[threadIdx.x + 128];
  __syncthreads();
  if (threadIdx.x <  64) smem[threadIdx.x] += smem[threadIdx.x +  64];
  __syncthreads();
  if (threadIdx.x <  32) {
    smem[threadIdx.x] += smem[threadIdx.x +  32];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +  16];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   8];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   4];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   2];
    __syncwarp();
    if (threadIdx.x == 0) dotptr[blockIdx.x] = smem[0] + smem[1];
  }
}

double error(double *a, double *b, long n){
  double err = 0.0; 
  for(long i = 0; i < n;i++){
    err += fabs(a[i] - b[i]);
  }
  return err;
}

int main() {
  
  long m = 1000;
  long n = 100000;

  double *x, *y, *prod_ref;
  cudaMallocHost((void**)&x, m*n * sizeof(double));
  cudaMallocHost((void**)&y, n * sizeof(double));
  cudaMallocHost((void**)&prod_ref, m * sizeof(double));

  #pragma omp parallel for schedule(static)
  for (long i = 0; i < m*n; i++) {
    x[i] = 1.0/(i+1);
  }

  for(long i = 0; i <n;i++){
    y[i] = 1.0/(n-i);
  }

  double tt = omp_get_wtime();

  matrix_product(prod_ref, x,y, m,n);

  printf("CPU Bandwidth = %f GB/s\n", (m+1)*(n+1)*sizeof(double) / (omp_get_wtime()-tt)/1e9);

  double *x_d, *y_d, *prod, *z_d, *prod_d;
  cudaMalloc(&x_d, m*n*sizeof(double));
  cudaMalloc(&y_d, n*sizeof(double));
  cudaMallocHost((void**)&prod, m * sizeof(double));


  long N_work = 1;
  for (long i = (n+BLOCK_SIZE-1)/(BLOCK_SIZE); i > 1; i = (i+BLOCK_SIZE-1)/(BLOCK_SIZE)) N_work += i;
  cudaMalloc(&z_d, N_work*sizeof(double)); // extra memory buffer for reduction across thread-blocks
  cudaMalloc(&prod_d, m*sizeof(double)); // extra memory buffer for reduction across thread-blocks

  cudaMemcpyAsync(x_d, x, m*n*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(y_d, y, n*sizeof(double), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  tt = omp_get_wtime();

  for (long i = 0; i < m;i++){
    double* tempdot = z_d;
    
    long Nb = (n+BLOCK_SIZE-1)/(BLOCK_SIZE);
    matrix_product_kernel<<<Nb,BLOCK_SIZE>>>(tempdot, x_d + i*n, y_d, n, 1);
    while (Nb > 1) {
    long N = Nb;
    Nb = (Nb+BLOCK_SIZE-1)/(BLOCK_SIZE);
    matrix_product_kernel<<<Nb,BLOCK_SIZE>>>(tempdot + N, tempdot, y_d, N, 0);
    tempdot += N;
    }
    
    //prod_d[i] = *tempdot;
    cudaMemcpy(prod_d + i, tempdot, sizeof(double), cudaMemcpyDeviceToDevice);
  }

  cudaMemcpyAsync(prod, prod_d, m*sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  printf("GPU Bandwidth = %f GB/s\n", (m+1)*(n+1)*sizeof(double) / (omp_get_wtime()-tt)/1e9);
  printf("Error = %f\n", error(prod, prod_ref, m));

  cudaFree(x_d);
  cudaFree(y_d);
  cudaFree(z_d);
  cudaFreeHost(x);
  cudaFreeHost(y);
}
