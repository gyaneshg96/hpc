/******************************************************************************
* FILE: omp_bug2.c
* DESCRIPTION:
*   Another OpenMP program with a bug. 
* AUTHOR: Blaise Barney 
* LAST REVISED: 04/06/05 
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

// The tid variable was not private, as it was declared outside the scope. So it was shared and constantly
// overwritten by the threads

// We need to specify total as a reduction variable for total

// Also we can put a barrier before thread is starting to ensure number of threads printed first
int main (int argc, char *argv[]) 
{
int i, tid;
double total = 0.0;

/*** Spawn parallel region ***/
#pragma omp parallel private(tid) shared(total)
  {
  /* Obtain thread number */
  tid = omp_get_thread_num();
  /* Only master thread does this */
  if (tid == 0) {
    int nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);
    }
  
  #pragma omp barrier
  printf("Thread %d is starting...\n",tid);

  #pragma omp barrier

  /* do some work */
  #pragma omp for schedule(dynamic,10) reduction(+: total)
  for (i=0; i<1000000; i++) 
     total = total + i*1.0;

  printf ("Thread %d is done! Total= %e\n",tid,total);

  } /*** End of parallel region ***/
}
