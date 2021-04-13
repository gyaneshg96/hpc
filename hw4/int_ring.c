#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

double ring_function(int *package, long length, int N, MPI_Comm comm){
  
  int rank, np;

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &np);
  
  MPI_Status status;
  

  MPI_Barrier(comm);
  double tt = MPI_Wtime();

  for (int i = 0; i < N; i++){
    if (rank == 0){
      if (i == 0){
        for (int j = 0; j<length;j++)
          *(package + j) = 0;
      }
      MPI_Send(package, length, MPI_INT, 1, i, comm);
      MPI_Recv(package, length, MPI_INT, np-1, i, comm, &status);
    }
    // else if (rank == np - 1 && i == N-1)
      // MPI_Recv(&package, 1, MPI_INT, rank - 1, i, comm, &status);
    else {
      MPI_Recv(package, length, MPI_INT, (rank - 1)%np, i, comm, &status);
      for (int j = 0; j < length; j++)
        *(package + j) += rank;
      MPI_Send(package, length, MPI_INT, (rank + 1)%np, i, comm);
    }
  }

  MPI_Barrier(comm);
  return MPI_Wtime() - tt;

}

int main(int argc, char** argv) {
  
  
  MPI_Init(&argc, &argv);
  
  MPI_Comm comm = MPI_COMM_WORLD;
  
  int rank, np;

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &np);
  
  int N = 10;
  /*
  int* package = (int *) calloc(sizeof(int), 1);
  int N = 10;
  for (int i = 0; i < N; i++){
    if (rank == 0){
      if (i == 0)
        package = 0;
      MPI_Send(package, length, MPI_INT, 1, i, comm);
      MPI_Recv(package, length, MPI_INT, np-1, i, comm, &status);
    }
    // else if (rank == np - 1 && i == N-1)
      // MPI_Recv(&package, 1, MPI_INT, rank - 1, i, comm, &status);
    else {
      MPI_Recv(package, length, MPI_INT, (rank - 1)%np, i, comm, &status);
      for (int j = 0; j < length; j++)
        *(package + j) += rank;
      MPI_Send(&package, length, MPI_INT, (rank + 1)%np, i, comm);
    }
  }
  */
  int* package = (int *) calloc(sizeof(int), 1);
  double tt = ring_function(package, 1, N, comm);
  double latency = tt/(N*np)*1000;
  if (rank == 0){
    printf("%d Value \n", *package);
    printf("Latency: %e ms \n", latency );
  }

  long size = 1000000; //1 int is 2 bytes, so approx 2 mb
  int* package2 = (int *) calloc(sizeof(int), size);

  tt = ring_function(package2, size, N, comm);
  double bandwidth = N*size/(tt*np*1e9);

  if (rank == 0){
    printf("Bandwidth: %e ms \n", bandwidth);
  }

  MPI_Finalize();
  return 0;
}
