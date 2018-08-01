
#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"

void serialAddVectors(int N, double *a, double *b, double *c){

  int n;
  for(n=0;n<N;++n){
    c[n] = a[n] + b[n];
  }

}

// code to be executed by each CUDA "thread"
__global__ void addVectorsKernel(int N, double *a, double *b, double *c){

  int threadRank = threadIdx.x; // thread rank in thread-block
  int blockRank = blockIdx.x;   // rank of thread-block
  int blockSize = blockDim.x;   // number of threads in each thread-block

  int n = threadRank + blockSize*blockRank;
  if(n<N)
    c[n] = a[n] + b[n];

}


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

  double *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, N*sizeof(double));
  cudaMalloc(&d_b, N*sizeof(double));
  cudaMalloc(&d_c, N*sizeof(double));

  cudaMemcpy(d_a, h_a, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, N*sizeof(double), cudaMemcpyHostToDevice);

  int TPB = 100;
  int B = (N + TPB -1)/TPB;

  // execute the kernel code with TPB threads per block and B thread-blocks
  // (total of B*TPB threads)
  addVectorsKernel <<< B, TPB >>> (N, d_a, d_b, d_c);

  cudaMemcpy(h_c, d_c, N*sizeof(double), cudaMemcpyDeviceToHost);
  
  for(n=0;n<5;++n){
    printf("h_c[%d] = %g\n", n, h_c[n]);
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}



