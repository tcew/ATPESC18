// to compile on cooley: nvcc  -arch sm_30 -o mandelbrot mandelbrot.cu -lm 
// to run on cooley:    ./mandelbrot 
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define MXITER 1000
#define NPOINTS 2048

// we will use these later to specify a 16x16 thread-block size
#define TX 16
#define TY 16

typedef struct {
  
  double r;
  double i;
  
}d_complex;

// return 1 if c is outside the mandelbrot set
// return 0 if c is inside the mandelbrot set

// TASK 1: annotate this as a device function 
???? int testpoint(d_complex c){
  
  d_complex z = c;
  
  for(int iter=0; iter<MXITER; iter++){
    
    double temp = (z.r*z.r) - (z.i*z.i) + c.r;
    
    z.i = z.r*z.i*2. + c.i;
    z.r = temp;
    
    if((z.r*z.r+z.i*z.i)>4.0){
      return 1;
    }
  }
  
  return 0;
  
}

// FREEBIE: partial reduction DEVICE function
__device__ void partialReduction(int outside, int *outsideCounts){
  
  __shared__ int s_outside[TX*TY];
  int t = threadIdx.x + threadIdx.y*TX;
  s_outside[t] = outside;
  
  int alive = TX*TY;
  while(alive>1){
    
    __syncthreads();
    
    alive /= 2;
    if(t<alive && t+alive<TX*TY)
      s_outside[t] += s_outside[t+alive];
    
  }
  
  if(t==0){
    int b = blockIdx.x + gridDim.x*blockIdx.y;
    outsideCounts[b] = s_outside[0];
  }
}

// TASK 2: make this a kernel that processes 
// (i,j) \in   [blockIdx.x*blockDim.x,(blockIdx.x+1)*blockDim.x) 
//           x [blockIdx.y*blockDim.y,(blockIdx.y+1)*blockDim.y) 

// TASK 2a: annotate this to indicate it is a kernel and change return type to void
???? void mandeloutside(int * outsideCounts){

  double eps = 1e-5;

  d_complex c;

  // TASK 2b: replace loop structures with (i,j) defined from blockIdx, blockDim, threadIdx
  //  for(i=0;i<NPOINTS;i++){
  //    for(j=0;j<NPOINTS;j++){
  int i = ????;
  int j = ????;

  c.r = -2. + 2.5*((double)i)/(double)(NPOINTS)+eps;
  c.i =       1.125*((double)j)/(double)(NPOINTS)+eps;
  
  // TASK 2c: check that (i,j) is in bounds
  int outside = 0; 
  if(i<???? && j<???){
    outside = testpoint(c);
  }
  //   }
  // }

  // FREEBIE: reduction of TX*TY values to one value on each thread-block
  partialReduction(outside, outsideCounts);

}

int main(int argc, char **argv){

  // TASK 3a: compute the number of blocks in the 2D thread array
  int GX = ????; // enough blocks in the x direction of size TX to cover NPOINTS
  int GY = ????; // enough blocks in the y direction of size TY to cover NPOINTS
  dim3 dimGrid(GX,GY,1);
  dim3 dimBlock(TX,TY,1);

  // TASK 3b: use cudaMalloc to create a DEVICE array that has one entry for each thread-block
  int *c_outsideCounts;
  cudaMalloc(????);

  // FREEBIE: create CUDA events for timing
  cudaEvent_t start, end;

  cudaEventCreate(&start);
  cudaEventCreate(&end);
  
  cudaEventRecord(start);
  
  // TASK 3c: specify the grid size and thread-block size
  mandeloutside <<< ????, ???? >>> (c_outsideCounts);
  
  // FREEBIE: timing
  float elapsed;
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&elapsed, start, end);
  elapsed /= 1000;
  printf("elapsed = %g\n", elapsed);

  // TASK 3d: allocate a HOST array to receive the contents of the c_outsideCounts array
  int *h_outsideCounts = (int*) calloc(GX*GY, sizeof(int));
  
  // TASK 3e: use cudaMemcpy to copy the contents of the entries of c_outsideCounts to h_outsideCounts
  cudaMemcpy(????);

  // FREEBIE: sum up the outsideCounts on the HOST
  int numoutside = 0;
  for(int n=0;n<GX*GY;++n){
    numoutside += h_outsideCounts[n];
  }

  printf("numoustide = %d\n", numoutside);

  double area = 2.*2.5*1.125*(NPOINTS*NPOINTS-numoutside)/(NPOINTS*NPOINTS);

  printf("area = %17.15lf\n", area);

  return 0;
}  
