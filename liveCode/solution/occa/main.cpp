
#include <stdio.h>
#include <stdlib.h>
#include "occa.hpp"

int main(int argc, char **argv){

  int N = 1000;
  double *h_a = (double*) malloc(N*sizeof(double));
  double *h_b = (double*) malloc(N*sizeof(double));
  double *h_c = (double*) malloc(N*sizeof(double));

  int n;

  for(n=0;n<N;++n){
    h_a[n] = 1+n;
    h_b[n] = 1-n;
  }

  occa::device device;
  device.setup("mode=CUDA, deviceID=0");

  occa::kernel addVectorsKernel = device.buildKernelFromSource("addVectors.okl", "addVectorsKernel");
  
  occa::memory o_a = device.malloc(N*sizeof(double), h_a);
  occa::memory o_b = device.malloc(N*sizeof(double), h_b);
  occa::memory o_c = device.malloc(N*sizeof(double), h_c);
  
  // execute the kernel code with TPB threads per block and B thread-blocks
  // (total of B*TPB threads)
  addVectorsKernel(N, o_a, o_b, o_c);

  o_c.copyTo(h_c);
  
  for(n=0;n<5;++n){
    printf("h_c[%d] = %g\n", n, h_c[n]);
  }

  o_a.free();
  o_b.free();
  o_c.free();

  return 0;
}



