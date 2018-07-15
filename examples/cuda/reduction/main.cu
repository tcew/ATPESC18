#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <math.h>

using namespace std;

#define datafloat float

#define BDIM 1024

__global__ void partialSum(const int N,
			   datafloat* __restrict__ u,
			   datafloat* __restrict__ blocksum){

  __shared__ datafloat s_blocksum[BDIM];

  int t = threadIdx.x;  
  int b = blockIdx.x;
  int n = b*blockDim.x + t;
  const int M = blockDim.x*gridDim.x;
  
  // start reduction in registers
  datafloat bs = 0;
  while(n<N){
    bs += u[n];
    n += M;
  }

  s_blocksum[t] = bs;
  
  // initially tag all threads as alive
  int alive = blockDim.x;

  while(alive>1){

    __syncthreads();  // barrier (make sure s_red is ready)
    
    alive /= 2;
    if(t < alive) s_blocksum[t] += s_blocksum[t+alive];
  }
  
  // value in s_blocksum[0] is sum of block of values
  if(t==0) blocksum[b] = s_blocksum[0];
}
  

// same partial sum reduction, but with unrolled while loop
__global__ void unrolledPartialSum(const int N,
				   datafloat* __restrict__ u,
				   datafloat* __restrict__ blocksum){
  
  __shared__ datafloat s_blocksum[BDIM];
  
  int t = threadIdx.x;  
  int b = blockIdx.x;
  int n = b*blockDim.x + t;
  const int M = blockDim.x*gridDim.x;

  datafloat bs = 0;
  while(n<N){
    bs += u[n];
    n+=M;
  }
  s_blocksum[t] = bs;    
    
  
  __syncthreads();  // barrier (make sure s_blocksum is ready)
  
  // manually unrolled blocksumuction (assumes BDIM=1024)
  if(BDIM>512){
    if(t<512) s_blocksum[t] += s_blocksum[t+512];
    
    __syncthreads();  // barrier (make sure s_blocksum is ready)
  }
  
  if(BDIM>256){
    if(t<256) s_blocksum[t] += s_blocksum[t+256];
    
    __syncthreads();  // barrier (make sure s_blocksum is ready)
  }
  
  if(BDIM>128){
    if(t<128) s_blocksum[t] += s_blocksum[t+128];
    
    __syncthreads();  
  }
  
  if(BDIM>64){
    if(t<64) s_blocksum[t] += s_blocksum[t+64];
    
    __syncthreads();  
  }

  if(BDIM>32){
    if(t<32) s_blocksum[t] += s_blocksum[t+32];

    __syncthreads();  
  }

  if(BDIM>16){
    if(t<16) s_blocksum[t] += s_blocksum[t+16];

    __syncthreads();  
  }

  if(BDIM>8){
    if(t<8)  s_blocksum[t] += s_blocksum[t+8];

    __syncthreads();  
  }

  if(BDIM>4){
    if(t<4) s_blocksum[t] += s_blocksum[t+4];

    __syncthreads();  
  }

  if(BDIM>2){
    if(t<2) s_blocksum[t] += s_blocksum[t+2];

    __syncthreads(); 
  }

  if(BDIM>1){
    if(t<1) s_blocksum[t] += s_blocksum[t+1];
  }

  // store result of this block blocksumuction
  if(t==0)
    blocksum[b] = s_blocksum[t];
}

// same partial sum reduction, but with unrolled while loop and
// less syncthread barriers (relies on 32 way SIMT concurrency)
__global__ void harrisUnrolledPartialSum(const int N,
					 datafloat* __restrict__ u,
					 datafloat* __restrict__ blocksum){

  // need to declare shared memory volatile to force write backs 
  volatile __shared__ datafloat s_blocksum[BDIM];

  int t = threadIdx.x;  
  int b = blockIdx.x;
  int n = b*blockDim.x + t;
  const int M = blockDim.x*gridDim.x;

  datafloat bs = 0;
  while(n<N){
    bs += u[n];
    n += M;
  }
  s_blocksum[t] = bs;

  __syncthreads();  // barrier (make sure s_blocksum is ready)

  // manually unrolled blocksumuction (assumes BDIM=1024)
  if(BDIM>512){
    if(t<512) s_blocksum[t] += s_blocksum[t+512];
    
    __syncthreads();  // barrier (make sure s_blocksum is ready)
  }
  
  if(BDIM>256){
    if(t<256) s_blocksum[t] += s_blocksum[t+256];
    
    __syncthreads();  // barrier (make sure s_blocksum is ready)
  }
    
  if(BDIM>128){
    if(t<128) s_blocksum[t] += s_blocksum[t+128];

    __syncthreads();  // barrier (make sure s_blocksum is ready)
  }

  if(BDIM>64){
    if(t<64) s_blocksum[t] += s_blocksum[t+64];

    __syncthreads();  // barrier (make sure s_blocksum is ready)
  }

  if(BDIM>32){
    if(t<32) s_blocksum[t] += s_blocksum[t+32];

    // should use sync(this_warp()); to safely guarantee warp synchronization
  }

  if(BDIM>16){
    if(t<16) s_blocksum[t] += s_blocksum[t+16];

    // should use sync(this_warp()); to safely guarantee warp synchronization
  }

  if(BDIM>8){
    if(t<8)  s_blocksum[t] += s_blocksum[t+8];

    // should use sync(this_warp()); to safely guarantee warp synchronization
  }

  if(BDIM>4){
    if(t<4) s_blocksum[t] += s_blocksum[t+4];

    // should use sync(this_warp()); to safely guarantee warp synchronization
  }

  if(BDIM>2){
    if(t<2) s_blocksum[t] += s_blocksum[t+2];

    // should use sync(this_warp()); to safely guarantee warp synchronization
  }

  if(BDIM>1){
    if(t<1) s_blocksum[t] += s_blocksum[t+1];

  }
  
  // store result of this block blocksumuction
  if(t==0)
    blocksum[b] = s_blocksum[t];
}

#define SIMT 32
 
// two step partial sum reduction relying on 32 way SIMT concurrency
__global__ void singleBarrierPartialSum(const int N, const datafloat* __restrict__  u, datafloat* __restrict__   partialsum){

  volatile __shared__ datafloat s_u[SIMT][SIMT];
  volatile __shared__ datafloat s_partialsum[SIMT];

  int b = blockIdx.x;
  int s = threadIdx.x;
  int g = threadIdx.y;

  // global thread count
  int M = gridDim.x*SIMT*SIMT;

  // global index
  int id = b*SIMT*SIMT + g*SIMT + s;

  // each thread grabs enough entries to cover array
  datafloat bs = 0;
  while(id<N){
    bs += u[id];
    id += M;
  }
  s_u[g][s] = bs;  // sync(this_warp()); 


  // 32 separate tree reductions
  if(s<16) s_u[g][s] += s_u[g][s + 16]; // sync(this_warp()); 
  if(s< 8) s_u[g][s] += s_u[g][s +  8]; // sync(this_warp()); 
  if(s< 4) s_u[g][s] += s_u[g][s +  4]; // sync(this_warp()); 
  if(s< 2) s_u[g][s] += s_u[g][s +  2]; // sync(this_warp()); 
  if(s==0) s_partialsum[g] = s_u[g][0] + s_u[g][1];

  // make sure all thread blocks got to here
  __syncthreads();
  
  // one thread block finishes partial reduction
  if(g==0){
    if(s<16) s_partialsum[s] += s_partialsum[s + 16]; // sync(this_warp()); 
    if(s< 8) s_partialsum[s] += s_partialsum[s +  8]; // sync(this_warp()); 
    if(s< 4) s_partialsum[s] += s_partialsum[s +  4]; // sync(this_warp()); 
    if(s< 2) s_partialsum[s] += s_partialsum[s +  2]; // sync(this_warp()); 
    if(s==0) partialsum[b] = s_partialsum[0] + s_partialsum[1];
  }
}



void sum(int N, datafloat *h_u){

  // Device Arrays
  datafloat *c_u, *c_partialsum;

  // Host array for partial sum
  datafloat *h_partialsum;

  // number of thread-blocks to partial sum u
  int GDIM = (N+BDIM-1)/BDIM;
  int RATIO = 32; // 32 loads per thread
  GDIM = (GDIM+RATIO-1)/RATIO;

  // allocate host array
  h_partialsum = (datafloat*) calloc(GDIM, sizeof(datafloat));

  // allocate device arrays
  cudaMalloc((void**) &c_u  , N*sizeof(datafloat));
  cudaMalloc((void**) &c_partialsum , GDIM*sizeof(datafloat));

  // copy from h_u to c_u (HOST to DEVICE)
  cudaMemcpy(c_u ,  h_u ,  N*sizeof(datafloat), cudaMemcpyHostToDevice);
  
  // Create CUDA events
  cudaEvent_t startEvent, endEvent;
  cudaEventCreate(&startEvent);
  cudaEventCreate(&endEvent);

  cudaEventRecord(startEvent, 0);

  // perform reduction 10 times
  int Ntests = 10, test;
  datafloat psum = 0;

  for(test=0;test<Ntests;++test){
    // perform tree wise block reduction on DEVICE
    // unrolledPartialSum <<< dim3(GDIM), dim3(BDIM) >>> (N, c_u, c_partialsum);

    // use harris optimized reduction
    //harrisUnrolledPartialSum <<< dim3(GDIM), dim3(BDIM) >>> (N, c_u, c_partialsum);
    
    // use single barrier kernel
    singleBarrierPartialSum <<< dim3(256,1,1), dim3(SIMT,SIMT,1) >>> (N, c_u, c_partialsum);
   
    // copy array of partially summed values to HOST
    cudaMemcpy(h_partialsum, c_partialsum, GDIM*sizeof(datafloat), cudaMemcpyDeviceToHost);

    // Finish reduce on host
    psum = 0;
    for(int n=0;n<GDIM;++n){
      psum += h_partialsum[n];
    }
  }

  // do timing
  cudaEventRecord(endEvent, 0);
  cudaEventSynchronize(endEvent);

  // Get time taken
  float timeTaken;
  cudaEventElapsedTime(&timeTaken, startEvent, endEvent);
  timeTaken /= 1000.f; // convert to seconds

  // print statistics
  double bytes = (N+GDIM)*sizeof(datafloat); // bytes moves
  double aveTimePerTest = timeTaken/Ntests;
  double GB = 1024*1024*1024;
  printf("average time per test = %g\n",   aveTimePerTest);
  printf("bandwidth estimate = %g GB/s\n", bytes/(aveTimePerTest*GB));
  printf("device memory used: %g GB\n",    bytes/GB);

  // output summation result
  printf("sum total = %g\n", psum);

  
  // free device arrays
  cudaFree(c_u);
  cudaFree(c_partialsum);

  // free HOST array
  free(h_partialsum);

}

int main(int argc, char** argv){

  // parse command line arguements
  if(argc != 2){
    printf("Usage: ./main N \n");
    return 0;
  }

  // Number of internal domain nodes in each direction
  const int N     = atoi(argv[1]);

  // Host Arrays
  datafloat *h_u   = (datafloat*) calloc(N, sizeof(datafloat));
  
  // initialize host array
  for(int n = 0;n < N; ++n){
    h_u[n] = 1;
  }

  // Solve discrete Laplacian
  sum(N, h_u);

  // Free the host array
  free(h_u);
}
