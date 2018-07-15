#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <math.h>

using namespace std;

#define BDIM 256

#define datafloat double

#define BX 16
#define BY 16

/* 

   Poisson problem:               diff(u, x, 2) + diff(u, y, 2) = f

   Coordinate transform:          x -> -1 + delta*i, 
                                  y -> -1 + delta*j

   2nd order finite difference:   4*u(j,i) - u(j-1,i) - u(j+1,i) - u(j,i-1) - u(j,i+1) = -delta*delta*f(j,i) 

   define: rhs(j,i) = -delta*delta*f(j,i)

   Jacobi iteration: newu(j,i) = 0.25*(u(j-1,i) + u(j+1,i) + u(j,i-1) + u(j,i+1) + rhs(j,i))

   To run with a 402x402 grid until the solution changes less than 1e-7 per iteration (in l2): ./main 400 1e-7  

*/

void jacobi(const int N,
	    const datafloat *rhs,
	    const datafloat *u,
	    datafloat *newu){
  
  for(int i=0;i<N;++i){
    for(int j=0;j<N;++j){

      // Get linear index into NxN inner nodes of (N+2)x(N+2) grid
      const int id = (j + 1)*(N + 2) + (i + 1);
      
      newu[id] = 0.25f*(rhs[id]
			+ u[id - (N+2)]
			+ u[id + (N+2)]
			+ u[id - 1]
			+ u[id + 1]);
    }
  }
}

datafloat reduceResidual(const int entries,
			 datafloat *u,
			 datafloat *newu){
  
  datafloat red = 0;
  for(int i=0;i<entries;++i){
    
    const datafloat diff = u[i] - newu[i];
    
    red += diff*diff;
  }

  return red;
}

void solve(int N, datafloat tol, datafloat *h_rhs, datafloat *h_u, datafloat *h_u2 ){

  const int iterationChunk = 100; // needs to be multiple of 2
  int iterationsTaken = 0;
  
  datafloat res;

  do {


    // Call jacobi [iterationChunk] times before calculating residual
    for(int i = 0; i < iterationChunk/2; ++i){

      // first iteration 
      jacobi(N, h_rhs, h_u, h_u2);
      
      // flip flop arguments
      jacobi(N, h_rhs, h_u2, h_u);
    }

    // Calculate residual
    res = reduceResidual(N*N, h_u, h_u2);
    res = sqrt(res);

    iterationsTaken += iterationChunk;

    printf("residual = %g after %d steps \n", res, iterationsTaken);

  } while(res > tol);

  printf("Residual                   : %7.9e\n"     , res);

}

int main(int argc, char** argv){

  // parse command line arguements
  if(argc != 3){
    printf("Usage: ./main N tol \n");
    return 0;
  }

  // Number of internal domain nodes in each direction
  const int N     = atoi(argv[1]);

  // Termination criterion || unew - u ||_2 < tol 
  const datafloat tol = atof(argv[2]);

  // Host Arrays
  datafloat *h_u   = (datafloat*) calloc((N+2)*(N+2), sizeof(datafloat));
  datafloat *h_u2  = (datafloat*) calloc((N+2)*(N+2), sizeof(datafloat));
  datafloat *h_rhs = (datafloat*) calloc((N+2)*(N+2), sizeof(datafloat));

  // FD node spacing
  datafloat delta = 2./(N+1);

  for(int j = 0; j < N+2; ++j){
    for(int i = 0; i < N+2; ++i){
      datafloat x = -1 + delta*i;
      datafloat y = -1 + delta*j;

      // solution is u = sin(pi*x)*sin(pi*y) so the rhs is: 
      h_rhs[i + (N+2)*j] = delta*delta*(2.*M_PI*M_PI*sin(M_PI*x)*sin(M_PI*y));
    }
  }

  // Solve discrete Laplacian
  solve(N, tol, h_rhs, h_u, h_u2);

  // Compute maximum error in finite difference solution and output solution
  FILE *fp = fopen("result.dat", "w");
  datafloat maxError = 0;
  for(int j = 0; j < N+2; ++j){
    for(int i = 0; i < N+2; ++i){
      datafloat x = -1 + delta*i;
      datafloat y = -1 + delta*j;
      datafloat error = fabs( sin(M_PI*x)*sin(M_PI*y) - h_u[i + (N+2)*j]);
      maxError = (error > maxError) ? error:maxError;
      fprintf(fp, "%g %g %g %g\n", x, y, h_u[i+(N+2)*j],error);
    }
  }
  fclose(fp);

  printf("Maximum absolute error     : %7.9e\n"     ,  maxError);

  // Free all the mess
  free(h_u);
  free(h_u2);
  free(h_rhs);
}
