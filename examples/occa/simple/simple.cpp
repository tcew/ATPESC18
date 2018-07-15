#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

#include "occa.hpp"

int main(int argc, char **argv){

  /* hard code platform and device number */
  int plat = 0;
  int dev = 1;

  occa::device device;
  device.setup("OpenCL", plat, dev);

  // build jacobi kernel from source file
  const char *functionName = "";

  // build Jacobi kernel
  occa::kernel simple = device.buildKernelFromSource("simple.occa", "simple");

  // size of array
  int N = 256;

  // set thread array for Jacobi iteration
  int T = 32;
  int dims = 1;
  occa::dim inner(T);
  occa::dim outer((N+T-1)/T);
  simple.setWorkingDims(dims, inner, outer);


  size_t sz = N*sizeof(float);

  // allocate array on HOST
  float *h_x = (float*) malloc(sz);
  for(int n=0;n<N;++n)
    h_x[n] = 123;
  
  // allocate array on DEVICE (copy from HOST)
  occa::memory c_x = device.malloc(sz, h_x);

  // queue kernel
  simple(N, c_x);
  
  // copy result to HOST
  c_x.copyTo(h_x);
  
  /* print out results */
  for(int n=0;n<N;++n)
    printf("h_x[%d] = %g\n", n, h_x[n]);

  exit(0);
  
}
