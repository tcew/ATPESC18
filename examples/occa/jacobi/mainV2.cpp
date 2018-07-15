#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

#include "occa.hpp"

#define datafloat float

#define BX 16
#define BY 16
#define BDIM 256

void solve(int N, datafloat tol, datafloat *h_rhs, datafloat *h_res, datafloat *h_u, datafloat *h_u2 ){

  const int iterationChunk = 100; // needs to be multiple of 2
  int iterationsTaken = 0;

  /* hard code platform and device number */
  int plat = 0;
  int dev = 1;

  occa::device device;
  device.setup("OpenCL", plat, dev);

  // build jacobi kernel from source file
  const char *functionName = "";

  occa::kernelInfo flags;
  flags.addDefine("BDIM", BDIM);
  flags.addDefine("BX", BX);
  flags.addDefine("BY", BY);

  if(sizeof(datafloat)==sizeof(float))  
    flags.addDefine("datafloat", "float");
  if(sizeof(datafloat)==sizeof(double))  
    flags.addDefine("datafloat", "double");

  // build Jacobi kernel
  occa::kernel jacobi = device.buildKernelFromSource("jacobi.okl", "jacobi", flags);

  // build partial reduction kernel
  occa::kernel partialReduce = device.buildKernelFromSource("partialReduce.okl", "partialReduce", flags);

  int Nred = ((N+2)*(N+2) + BDIM-1)/BDIM; // number of blocks for partial reduction
#if 0
  {
    int dims = 1;
    occa::dim inner(BDIM);
    occa::dim outer(Nred);
    partialReduce.setWorkingDims(dims, inner, outer);
  }
#endif

  // build Device Arrays and transfer data from host arrays
  size_t sz = (N+2)*(N+2)*sizeof(datafloat);
  
  occa::memory c_u   = device.malloc(sz, h_u);
  occa::memory c_u2  = device.malloc(sz, h_u2);
  occa::memory c_rhs = device.malloc(sz, h_rhs);
  occa::memory c_res = device.malloc(sz, h_res);

  datafloat res;

  // Jacobi iteration loop
  do {

    // Call jacobi [iterationChunk] times before calculating residual
    for(int i = 0; i < iterationChunk/2; ++i){

      jacobi(N, c_rhs, c_u, c_u2);

      jacobi(N, c_rhs, c_u2, c_u);

    }
    
    // calculate norm(u-u2) with interval iterationChunk iterations
    {
      // design thread array for norm(u-u2)
      int N2 = (N+2)*(N+2);
      int Nred = (N2+BDIM-1)/BDIM;

      partialReduce(N2, c_u, c_u2, c_res);

      c_res.copyTo(h_res);
      
      res = 0;
      for(int i = 0; i < Nred; ++i)
	res += h_res[i];

      res = sqrt(res);
    }
    
    iterationsTaken += iterationChunk;
    
    printf("residual = %g after %d steps \n", res, iterationsTaken);

  } while(res > tol);

  printf("Residual                   : %7.9e\n"     , res);

  // blocking copy of solution from device to host 
  c_u.copyTo(h_u);

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
  datafloat *h_res = (datafloat*) calloc((N+2)*(N+2), sizeof(datafloat));

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
  solve(N, tol, h_rhs, h_res, h_u, h_u2);

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
  free(h_res);
  free(h_rhs);
}
