#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <string.h>

int main(int argc, char * argv[]){
  int mpirank, i, p;
  long N;
  long *singlevec, *offsets;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  sscanf(argv[1], "%ld", &N);

  long *eachvec = (long *)calloc(N/p, sizeof(long));
  long *eachcumvec = (long *)calloc(N/p, sizeof(long));
  offsets = (long *)calloc(p, sizeof(long));

  if (mpirank == 0){
    singlevec = (long *)calloc(N,sizeof(long));
    for (long i = 0; i<N;i++){
      singlevec[i] = i;
    }
  }
   
  MPI_Scatter(singlevec, N/p, MPI_LONG, eachvec, N/p, MPI_LONG, 0, MPI_COMM_WORLD);

  eachcumvec[0] = eachvec[0];
  for(long i = 1; i < N/p;i++){
    eachcumvec[i] = eachcumvec[i-1] + eachvec[i];
  }

  MPI_Allgather(&eachcumvec[N/p - 1], 1, MPI_LONG, offsets, 1, MPI_LONG, MPI_COMM_WORLD);

  long offset = 0;
  for (int i = 1; i<=mpirank;i++){
    offset += offsets[i-1];
  }

  MPI_Barrier(MPI_COMM_WORLD);

  printf("Rank : %d\n", mpirank);
  for (int i = 0; i<N/p; i++){
    eachcumvec[i] += offset;
    printf("%ld ", eachcumvec[i]);
  }
  printf("\n");

  MPI_Finalize();
  return 0;
}



