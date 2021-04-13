#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <string.h>

/* compuate global residual, assuming ghost values are updated */
double compute_residual(double *lu, int ln, double invhsq){
  int i;
  double tmp, gres = 0.0, lres = 0.0;

  for (i = 1; i <= ln; i++){
    for (int j = 1; j <= ln; i++){
    tmp = ((4.0*lu[i*(ln+2) + j]
         - lu[(i-1)*(ln+2) + j]
         - lu[(i+1)*(ln+2) + j]
         - lu[i*(ln+2) + j-1]
         - lu[i*(ln+2) + j+1]) * invhsq - 1);
    lres += tmp * tmp;
  }
  }
  /* use allreduce for convenience; a reduce would also be sufficient */
  MPI_Allreduce(&lres, &gres, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return sqrt(gres);
}


int main(int argc, char * argv[]){
  int mpirank, i, p, N, lN, iter, max_iters;
  MPI_Status status, status1;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  /* get name of host running MPI process */
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);
  printf("Rank %d/%d running on %s.\n", mpirank, p, processor_name);

  sscanf(argv[1], "%d", &N);
  sscanf(argv[2], "%d", &max_iters);

  /* compute number of unknowns handled by each process */
  lN = N / p;
  if ((N % p != 0) && mpirank == 0 ) {
    printf("N: %d, local N: %d\n", N, lN);
    printf("Exiting. N must be a multiple of p\n");
    MPI_Abort(MPI_COMM_WORLD, 0);
  }
  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  double tt = MPI_Wtime();

  //square root
  int ln = (int) sqrt(lN);
  int rootp = (int) sqrt(p);

  /* Allocation of vectors, including left/upper and right/lower ghost points */
  double * lu    = (double *) calloc(sizeof(double), (ln + 2)*(ln + 2));
  double * lunew = (double *) calloc(sizeof(double), (ln + 2)*(ln + 2));
  
  // Arrays for storing vertical edges
  double * vertical1 = (double *) calloc(sizeof(double), ln);
  double * vertical2 = (double *) calloc(sizeof(double), ln);
  double * vertical3 = (double *) calloc(sizeof(double), ln);
  double * vertical4 = (double *) calloc(sizeof(double), ln);
  double * lutemp;

  double h = 1.0 / (N + 1)*(N + 1);
  double hsq = h * h;
  double invhsq = 1./hsq;
  double gres, gres0, tol = 1e-5;

  /* initial residual */
  gres0 = compute_residual(lu, lN, invhsq);
  gres = gres0;

  for (iter = 0; iter < max_iters && gres/gres0 > tol; iter++) {

    /* Jacobi step for local points */
    for (int i = 1; i <= ln; i++){
      for (int j = 1; j <= ln; j++){
        lunew[i*(ln+2) + j]  = 0.25 * (hsq
                              + lu[(i-1)*(ln+2) + j]
                              + lu[(i+1)*(ln+2) + j]
                              + lu[i*(ln+2) + j-1]
                              + lu[i*(ln+2) + j+1]);         
      }
    }
    
    //copy values to vertical arrays
    for (int i = 0; i < ln; i++){
      vertical1[i] = lunew[i*(ln+2) + 1];
      vertical2[i] = lunew[i*(ln+2) + ln];
      vertical3[i] = lunew[i*(ln+2) + ln+1];
      vertical4[i] = lunew[i*(ln+2)];
    }
    /* communicate ghost values */

    //can be done more elegantly

    //deal with corners
    if (mpirank == 0){
      MPI_Send(vertical2, ln, MPI_DOUBLE, 1, 123, MPI_COMM_WORLD);
      MPI_Recv(vertical3, ln, MPI_DOUBLE, 1, 124, MPI_COMM_WORLD, &status1);
      MPI_Send(&lunew[ln*(ln+2) + 1], ln, MPI_DOUBLE, rootp, 123, MPI_COMM_WORLD);
      MPI_Recv(&lunew[(ln+1)*(ln+2) + 1], ln, MPI_DOUBLE, rootp, 124, MPI_COMM_WORLD, &status1);
    }
    else if (mpirank == p - 1){
      MPI_Send(vertical1, ln, MPI_DOUBLE, mpirank-1, 123, MPI_COMM_WORLD);
      MPI_Recv(vertical4, ln, MPI_DOUBLE, mpirank-1, 124, MPI_COMM_WORLD, &status1);
      MPI_Send(&lunew[ln+2 + 1], ln, MPI_DOUBLE, mpirank-1, 123, MPI_COMM_WORLD);
      MPI_Recv(&lunew[1], ln, MPI_DOUBLE, mpirank-1, 124, MPI_COMM_WORLD, &status1);
    }
    else if (mpirank == rootp - 1){
      MPI_Send(vertical1, ln, MPI_DOUBLE, mpirank-1, 123, MPI_COMM_WORLD);
      MPI_Recv(vertical4, ln, MPI_DOUBLE, mpirank-1, 124, MPI_COMM_WORLD, &status1);
      MPI_Send(&lunew[ln*(ln+2) + 1], ln, MPI_DOUBLE, mpirank-1, 123, MPI_COMM_WORLD);
      MPI_Recv(&lunew[(ln+1)*(ln+2) +1], ln, MPI_DOUBLE, mpirank-1, 124, MPI_COMM_WORLD, &status1);
    }
    else if (mpirank == rootp*(rootp - 1)){
      MPI_Send(vertical2, ln, MPI_DOUBLE, mpirank-1, 123, MPI_COMM_WORLD);
      MPI_Recv(vertical3, ln, MPI_DOUBLE, mpirank-1, 124, MPI_COMM_WORLD, &status1);
      MPI_Send(&lunew[ln+2 + 1], ln, MPI_DOUBLE, mpirank-1, 123, MPI_COMM_WORLD);
      MPI_Recv(&lunew[1], ln, MPI_DOUBLE, mpirank-1, 124, MPI_COMM_WORLD, &status1);
    }

    //deal with edges
    else if (mpirank < rootp - 1) {
      MPI_Send(vertical1, ln, MPI_DOUBLE, mpirank-1, 123, MPI_COMM_WORLD);
      MPI_Recv(vertical4, ln, MPI_DOUBLE, mpirank-1, 124, MPI_COMM_WORLD, &status1);
      MPI_Send(vertical2, ln, MPI_DOUBLE, 1, 123, MPI_COMM_WORLD);
      MPI_Recv(vertical3, ln, MPI_DOUBLE, 1, 124, MPI_COMM_WORLD, &status1);
      MPI_Send(&lunew[ln*(ln+2) + 1], ln, MPI_DOUBLE, mpirank-1, 123, MPI_COMM_WORLD);
      MPI_Recv(&lunew[(ln+1)*(ln+2) +1], ln, MPI_DOUBLE, mpirank-1, 124, MPI_COMM_WORLD, &status1);
    }
    else if (mpirank > rootp*(rootp - 1)) {
      MPI_Send(vertical1, ln, MPI_DOUBLE, mpirank-1, 123, MPI_COMM_WORLD);
      MPI_Recv(vertical4, ln, MPI_DOUBLE, mpirank-1, 124, MPI_COMM_WORLD, &status1);
      MPI_Send(vertical2, ln, MPI_DOUBLE, 1, 123, MPI_COMM_WORLD);
      MPI_Recv(vertical3, ln, MPI_DOUBLE, 1, 124, MPI_COMM_WORLD, &status1);
      MPI_Send(&lunew[ln+2 + 1], ln, MPI_DOUBLE, mpirank-1, 123, MPI_COMM_WORLD);
      MPI_Recv(&lunew[1], ln, MPI_DOUBLE, mpirank-1, 124, MPI_COMM_WORLD, &status1);
    }
    else if (mpirank % rootp == 0){
      MPI_Send(vertical2, ln, MPI_DOUBLE, 1, 123, MPI_COMM_WORLD);
      MPI_Recv(vertical3, ln, MPI_DOUBLE, 1, 124, MPI_COMM_WORLD, &status1);
      MPI_Send(&lunew[ln+2 + 1], ln, MPI_DOUBLE, mpirank-1, 123, MPI_COMM_WORLD);
      MPI_Recv(&lunew[1], ln, MPI_DOUBLE, mpirank-1, 124, MPI_COMM_WORLD, &status1);
      MPI_Send(&lunew[ln*(ln+2) + 1], ln, MPI_DOUBLE, mpirank-1, 123, MPI_COMM_WORLD);
      MPI_Recv(&lunew[(ln+1)*(ln+2) +1], ln, MPI_DOUBLE, mpirank-1, 124, MPI_COMM_WORLD, &status1);
    }
    else if (mpirank % rootp == rootp - 1){
      MPI_Send(vertical1, ln, MPI_DOUBLE, 1, 123, MPI_COMM_WORLD);
      MPI_Recv(vertical4, ln, MPI_DOUBLE, 1, 124, MPI_COMM_WORLD, &status1);
      MPI_Send(&lunew[ln+2 + 1], ln, MPI_DOUBLE, mpirank-1, 123, MPI_COMM_WORLD);
      MPI_Recv(&lunew[1], ln, MPI_DOUBLE, mpirank-1, 124, MPI_COMM_WORLD, &status1);
      MPI_Send(&lunew[ln*(ln+2) + 1], ln, MPI_DOUBLE, mpirank-1, 123, MPI_COMM_WORLD);
      MPI_Recv(&lunew[(ln+1)*(ln+2) +1], ln, MPI_DOUBLE, mpirank-1, 124, MPI_COMM_WORLD, &status1);
    }
    //deal with remaining
    else {
      MPI_Send(vertical1, ln, MPI_DOUBLE, mpirank-1, 123, MPI_COMM_WORLD);
      MPI_Recv(vertical4, ln, MPI_DOUBLE, mpirank-1, 124, MPI_COMM_WORLD, &status1);
      MPI_Send(vertical2, ln, MPI_DOUBLE, 1, 123, MPI_COMM_WORLD);
      MPI_Recv(vertical3, ln, MPI_DOUBLE, 1, 124, MPI_COMM_WORLD, &status1);
      MPI_Send(&lunew[ln+2 + 1], ln, MPI_DOUBLE, mpirank-1, 123, MPI_COMM_WORLD);
      MPI_Recv(&lunew[1], ln, MPI_DOUBLE, mpirank-1, 124, MPI_COMM_WORLD, &status1);
      MPI_Send(&lunew[ln*(ln+2) + 1], ln, MPI_DOUBLE, mpirank-1, 123, MPI_COMM_WORLD);
      MPI_Recv(&lunew[(ln+1)*(ln+2) +1], ln, MPI_DOUBLE, mpirank-1, 124, MPI_COMM_WORLD, &status1);
    }

    for (int i = 0; i < ln; i++){
      lunew[i*(ln+2) + 1] = vertical1[i];
      lunew[i*(ln+2) + ln] = vertical2[i];
      lunew[i*(ln+2) + ln+1] = vertical3[i];
      lunew[i*(ln+2)] = vertical4[i];
    }


    /* copy newu to u using pointer flipping */
    lutemp = lu; lu = lunew; lunew = lutemp;
    if (0 == (iter % 10)) {
      gres = compute_residual(lu, lN, invhsq);
      if (0 == mpirank) {
	printf("Iter %d: Residual: %g\n", iter, gres);
      }
    }
  }

  /* Clean up */
  free(lu);
  free(lunew);
  free(vertical1);
  free(vertical2);
  free(vertical3);
  free(vertical4);

  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  double elapsed = MPI_Wtime() - tt;
  if (0 == mpirank) {
    printf("Time elapsed is %f seconds.\n", elapsed);
  }
  MPI_Finalize();
  return 0;
}