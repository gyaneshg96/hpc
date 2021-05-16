#include <stdio.h>
#include "utils.h"
#include <math.h>
#include <string>
#include <omp.h>
#include <iostream>


double norm(double *x, long N)
{
  double normval = 0;
  for (long i = 1; i < N-1; i++){
  for (long j = 1; j < N-1; j++)
  {
    double temp = 0;
    temp += 4.0 * (N+1)*(N+1)* *(x + i*N + j);
     if (i - 1 >= 0)
      temp += -1.0 *(N+1)*(N+1)*  *(x + (i - 1)*N +j);
     if (i + 1 < N)
      temp += -1.0 *(N+1)*(N+1)*  *(x + (i + 1)*N + j);
     if (j - 1 >= 0)
      temp += -1.0 *(N+1)*(N+1)*  *(x + i*N + j - 1);
     if (j + 1 < N)
      temp += -1.0 *(N+1)*(N+1)*  *(x + i*N + j + 1);
    temp -= 1.0;
    normval += temp * temp;
  }
  }
  return sqrt(normval);
}

void jacobi_serial(double *U, long N){
  long max_iterations = 5000;
  
  double initial_norm = norm(U,N);
  double *Uprev = (double *)calloc(N*N, sizeof(double));
  double h = 1.0/(N+1);
  for (int iter = 1; iter <= max_iterations; iter++){
    for (int i = 1; i < N - 1; i++){
      for (int j = 1; j < N - 1; j++){
        *(Uprev + i*N + j) = 0.25 * ( h*h +
                                + *(U + (i-1)*N + j) 
                                + *(U + (i*N + j - 1))
                                + *(U + (i+1)*N + j)
                                + *(U + i*N + j + 1));
        // cout<<*(U + i*N + j)<<endl;
      }
    }
    for (int i = 0;i < N*N; i++){
      *(U + i) = *(Uprev + i);
    }
    // if (iter % 500 == 0)
    // cout<<initial_norm<<" Norm of difference "<<norm(U, N)<<endl;
  }
}

__global__ void jacobi_kernel(double *U, double *Uprev, long N, double h){
  long idy = blockIdx.y * blockDim.y + threadIdx.y;
  long idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N && idy < N){
  Uprev[idy * N + idx] = h*h
  if (idy - 1 >= 0)
   Uprev[idy * N + idx] += U[(idy-1)*N + idx];
  if (idx - 1 >= 0)
   Uprev[idy * N + idx] += U[(idy)*N + idx - 1];
  if (idx + 1 < N )
   Uprev[idy * N + idx] += U[(idy)*N + idx + 1];
  if (idy + 1 < N )
   Uprev[idy * N + idx] += U[(idy+1)*N + idx];
  
  Uprev[idy * N + idx] *= 0.25;
  }
}

void jacobi_cuda(double *U, long N){

  long max_iterations = 5000;

  double initial_norm = norm(U,N);
  double *Uprev = (double *)cudaMalloc(N*N, sizeof(double));
  double h = 1.0/(N+1);
  for (int iter = 1; iter <= max_iterations; iter++){
    jacobi_kernel(U, Uprev, N, h);
    cudaMemcpy(U, Uprev, N*sizeof(double),  cudaMemcpyDeviceToDevice);
  }
}

int main(int argc, char **args)
{
  long N = 100;

  double *x;
  cudaMallocHost((void**)&x, N * N * sizeof(double));
  cudaMallocHost((void**)&y, N * N * sizeof(double));
  
  for (long i = 0; i < N*N; i++ ){
    x[i] = 0.0;
    y[i] = 0.0;
  }

  Timer t;
  t.tic();
  jacobi_serial(x, N);
  cout << "Time Taken for serial " << t.toc() << endl;

  double *y_d;
  cudaMalloc(&y_d, N*N*sizeof(double));
  cudaMemcpyAsync(y_d, y, N*N*sizeof(double), cudaMemcpyHostToDevice);
  t.tic();
  jacobi_cuda(y_d, N);
  cudaMemcpyAsync(y, y_d, N*N*sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  
  cout << "Time Taken for cuda " << t.toc() << endl;

  double err = 0.0;
  for (int i = 0; i < N*N; i++){
    err += max(err, abs(x[i] - y[i]));
  }
  printf("Error = %f\n", err);
}

  

