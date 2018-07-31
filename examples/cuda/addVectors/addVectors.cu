#include "cuda.h"

#include <stdio.h>
#include <stdlib.h>

// compute: d_c[n] = d_a[n] + d_b[n] for n \in [0,N)
__global__ void addVectorsKernel(int N, float *d_a, float *d_b, float *d_c){
	   
  // Convert thread and thread-block indices into array index 
  const int n  = threadIdx.x + blockDim.x*blockIdx.x;
	   
  // If index is in [0,N-1] add entries
  if(n<N){
    d_c[n] = d_a[n] + d_b[n];
  }
}

int main(int argc,char **argv){

  // size of array for this DEMO
  int N = 512; 

  // Allocate HOST arrays
  float *h_a = (float*) calloc(N, sizeof(float));
  float *h_b = (float*) calloc(N, sizeof(float));
  float *h_c = (float*) calloc(N, sizeof(float));

  // initialize a and b
  for(int n=0;n<N;++n){
    h_a[n] = 1.-n;
    h_b[n] = 3.+n;
  }
  
  // Allocate DEVICE array
  float *d_a, *d_b, *d_c;  

  cudaMalloc((void**) &d_a, N*sizeof(float));
  cudaMalloc((void**) &d_b, N*sizeof(float));
  cudaMalloc((void**) &d_c, N*sizeof(float));
 
  // copy data for a and b from HOST
  cudaMemcpy(d_a, h_a, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, N*sizeof(float), cudaMemcpyHostToDevice);

  // number of threads in each thread-block
  dim3 dimBlock(512,1,1);          // 512 threads per thread-block

  // compute number of thread-blocks so that we just exceed N threads
  dim3 dimGrid((N+511)/512, 1, 1); 
    
  // Queue kernel on DEVICE
  addVectorsKernel <<< dimGrid, dimBlock >>> (N, d_a, d_b, d_c);
    
  // Transfer result from DEVICE to HOST
  cudaMemcpy(h_c, d_c, N*sizeof(float), cudaMemcpyDeviceToHost);
    
  // Print out result
  for(int n=0;n<N;++n) printf("h_c[%d] = %f\n", n, h_c[n]);

  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
  free(h_a); free(h_b); free(h_c);
}
