#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <string>
#include <iostream>

using namespace std;
// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long *prefix_sum, const long *A, long n)
{
  if (n == 0)
    return;
  // for good looking answers
  prefix_sum[0] = A[0];
  for (long i = 1; i < n; i++)
  {
    prefix_sum[i] = prefix_sum[i - 1] + A[i];
  }
}

void scan_omp(long *prefix_sum, const long *A, long n, int nthreads)
{

  if (n == 0)
    return;
  long *residuals = (long *)malloc(nthreads * sizeof(long));
  long block = n / nthreads;
  // cout<<block<<endl;

  for (long i = 0; i < nthreads; i++){
    prefix_sum[i*block] = A[i*block];
  }

#pragma omp parallel shared(prefix_sum, residuals, A, block)
  {
    // int nthreads = omp_get_num_threads();
    int tid = omp_get_thread_num();
    // cout<<"Parallel "<<tid<<endl;
    // long* residuals = (long *)malloc(block * sizeof(long));
    if (tid == 0){
      cout<<"Number of threads "<<omp_get_num_threads();
    }
    // cout<<"Thread id "<<tid<<endl;
    for (long i = tid * (block) + 1; i < block + tid * (block); i++)
    {
      // cout<<i<<" Parallel"<<endl;
      prefix_sum[i] = prefix_sum[i - 1] + A[i];
    }
  }
  cout<<"Serial"<<endl;
  //serial part
  residuals[0] = 0;
  for (long i = 1; i < nthreads; i++)
  {
    residuals[i] = residuals[i - 1] + prefix_sum[i * block - 1];
  }

#pragma omp parallel shared(prefix_sum, residuals, A, block)
  {
    int tid = omp_get_thread_num();
    for (long i = tid * (block); i < block + tid * (block); i++)
    {
      // cout<<i<<" Paralllel"<<endl;
      prefix_sum[i] += residuals[tid];
    }
  }
}

int main(int argc, char **args)
{

  long nthreads = stol(string(*(args + 1)));

  cout << "Number of threads to spawn " << nthreads << endl;

  long N = 100000000;
  long *A = (long *)malloc(N * sizeof(long));
  long *B0 = (long *)malloc(N * sizeof(long));
  long *B1 = (long *)malloc(N * sizeof(long));
  for (long i = 0; i < N; i++){
    A[i] = rand();
    // cout<<A[i]<<endl;
  }

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  omp_set_num_threads(nthreads);
  scan_omp(B1, A, N, nthreads);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++){
    err = std::max(err, std::abs(B0[i] - B1[i]));
    // cout<<B0[i]<<" "<<B1[i]<<endl;
  }
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
