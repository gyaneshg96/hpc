// g++ -std=c++11 -O3 -march=native MMult1.cpp && ./a.out

#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "utils.h"

#define BLOCK_SIZE 16

// Note: matrices are stored in column major order; i.e. the array elements in
// the (m x n) matrix C are stored in the sequence: {C_00, C_10, ..., C_m0,
// C_01, C_11, ..., C_m1, C_02, ..., C_0n, C_1n, ..., C_mn}
void MMult0(long m, long n, long k, double *a, double *b, double *c)
{
    for (long p = 0; p < k; p++)
    {
  for (long j = 0; j < n; j++)
  {
      for (long i = 0; i < m; i++)
      {
        double A_ip = a[i + p * m];
        double B_pj = b[p + j * k];
        double C_ij = c[i + j * m];
        C_ij = C_ij + A_ip * B_pj;
        c[i + j * m] = C_ij;
      }
    }
  }
}
void MMult1(long m, long n, long k, double *a, double *b, double *c)
{
  // TODO: See instructions below
  // #pragma omp parallel for
  for (long p = 0; p < k; p = p + BLOCK_SIZE)
  {
    for (long j = 0; j < n; j = j + BLOCK_SIZE)
    {
      for (long i = 0; i < m; i = i + BLOCK_SIZE)
      {
        for (long pp = p; pp < p + BLOCK_SIZE; pp++)
          {
        for (long jj = j; jj < j + BLOCK_SIZE; jj++)
        {
      for (long ii = i; ii < i + BLOCK_SIZE; ii++)
      {
            double sum = *(c + jj * m + ii);
            double aij = *(a + pp * m + ii);
            double bjp = *(b + jj * k + pp);
            sum += aij * bjp;
            *(c + jj * m + ii) = sum;
            // MMultUtil(m,n,k, a+p*m+i,  b+k*j + p, c + j*m + i)
          }
        }
      }
      }
    }
  }
}

int main(int argc, char **argv)
{
  const long PFIRST = BLOCK_SIZE;
  const long PLAST = 2000;
  const long PINC = std::max(50 / BLOCK_SIZE, 1) * BLOCK_SIZE; // multiple of BLOCK_SIZE

  printf(" Dimension     Time1  Time    Gflop/s       GB/s        Error\n");
  for (long p = PFIRST; p < PLAST; p += PINC)
  {
    long m = p, n = p, k = p;
    long NREPEATS = 1e9 / (m * n * k) + 1;
    // long NREPEATS = 1;
    double *a = (double *)aligned_malloc(m * k * sizeof(double));     // m x k
    double *b = (double *)aligned_malloc(k * n * sizeof(double));     // k x n
    double *c = (double *)aligned_malloc(m * n * sizeof(double));     // m x n
    double *c_ref = (double *)aligned_malloc(m * n * sizeof(double)); // m x n

    // Initialize matrices
    for (long i = 0; i < m * k; i++)
      a[i] = drand48();
    for (long i = 0; i < k * n; i++)
      b[i] = drand48();
    for (long i = 0; i < m * n; i++)
      c_ref[i] = 0;
    for (long i = 0; i < m * n; i++)
      c[i] = 0;

    Timer t;
    t.tic();
    for (long rep = 0; rep < NREPEATS; rep++)
    { // Compute reference solution
      MMult0(m, n, k, a, b, c_ref);
    }

    double t1 = t.toc();
    t.tic();
    for (long rep = 0; rep < NREPEATS; rep++)
    {
      MMult1(m, n, k, a, b, c);
    }
    double time = t.toc();
    double flops = 2 * m * n * k * NREPEATS / (time) / 1e9;                                            // TODO: calculate from m, n, k, NREPEATS, time
    double bandwidth = sizeof(double) * NREPEATS * (m * n + k * n + m * k + m * n * k) / (time * 1e9); // TODO: calculate from m, n, k, NREPEATS, time
    printf("%10ld %10f %10f %10f %10f", p, t1, time, flops, bandwidth);

    double max_err = 0;
    for (long i = 0; i < m * n; i++)
      max_err = std::max(max_err, fabs(c[i] - c_ref[i]));
    printf(" %10e\n", max_err);

    aligned_free(a);
    aligned_free(b);
    aligned_free(c);
  }

  return 0;
}

// * Using MMult0 as a reference, implement MMult1 and try to rearrange loops to
// maximize performance. Measure performance for different loop arrangements and
// try to reason why you get the best performance for a particular order?
//
//
// * You will notice that the performance degrades for larger matrix sizes that
// do not fit in the cache. To improve the performance for larger matrices,
// implement a one level blocking scheme by using BLOCK_SIZE macro as the block
// size. By partitioning big matrices into smaller blocks that fit in the cache
// and multiplying these blocks together at a time, we can reduce the number of
// accesses to main memory. This resolves the main memory bandwidth bottleneck
// for large matrices and improves performance.
//
// NOTE: You can assume that the matrix dimensions are multiples of BLOCK_SIZE.
//
//
// * Experiment with different values for BLOCK_SIZE (use multiples of 4) and
// measure performance.  What is the optimal value for BLOCK_SIZE?
//
//
// * Now parallelize your matrix-matrix multiplication code using OpenMP:
// To do that, you have to add -fopenmp to the compiler and comment in the
// omp.h header file
//
//
// * What percentage of the peak FLOP-rate do you achieve with your code?
//
//
// NOTE: Compile your code using the flag -march=native. This tells the compiler
// to generate the best output using the instruction set supported by your CPU
// architecture. Also, try using either of -O2 or -O3 optimization level flags.
// Be aware that -O2 can sometimes generate better output than using -O3 for
// programmer optimized code.
