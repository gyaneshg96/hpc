#include <stdio.h>
#include "utils.h"
#include <math.h>
#include <string>
#include <omp.h>
#include <iostream>

#ifdef _OPENMP
using namespace std;
double parallel_norm(double *x, long N){
  return 0;
}
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

void jacobi_omp(double *U, long N){
  long max_iterations = 5000;
  double *Uprev = (double *)calloc(N*N, sizeof(double));
  double h = 1.0/(N+1);

  for (int iter = 1; iter <= max_iterations; iter++){
    #pragma omp parallel
    {
      #pragma omp for
      for (int i = 1; i < N - 1; i++){
        for (int j = 1; j < N - 1; j++){
          *(U + i*N + j) = 0.25 * ( h*h
                                + *(Uprev + (i-1)*N + j) 
                                + *(Uprev + (i*N + j - 1))
                                + *(Uprev + (i+1)*N + j)
                                + *(Uprev + i*N + j + 1));
      }
    }
    }

    #pragma omp parallel for
    for (int i = 0;i < N*N; i++){
      *(Uprev + i) = *(U + i);
    }
    // if (iter % 500 == 0)
    // cout<<"Norm of difference "<<norm(U, N)<<endl;
  }

}

int main(int argc, char **args)
{

  long N = stol(string(*(args + 1)));

  double *U1 = (double *)calloc(N*N, sizeof(double)); //zero init
  double *U2 = (double *)calloc(N*N, sizeof(double)); //zero init
  // double *temp = (double *)malloc(N * sizeof(double));

  // double initial_norm = norm(U, N);
  Timer t;
  t.tic();
  jacobi_serial(U1, N);
  cout << "Time Taken for serial " << t.toc() << endl;

  t.tic();
  jacobi_omp(U2, N);
  cout << "Time Taken for parallel" << t.toc() << endl;

  double err = 0.0;
  for (int i = 0; i < N*N; i++){
    err = max(err, abs(U1[i] - U2[i]));
  }

  cout<< "Error "<< err<<endl;
  
  free(U1);
  free(U2);
  return 0;
}

#endif
