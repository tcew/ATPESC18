#include <iostream>
#include <stdio.h>
#include <cuda.h>
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

__global__ void jacobi(const int N,
                       const datafloat *rhs,
                       const datafloat *u,
                       datafloat *newu){

  // Get thread indices
  const int i = blockIdx.x*blockDim.x + threadIdx.x;
  const int j = blockIdx.y*blockDim.y + threadIdx.y;

  if((i < N) && (j < N)){

    // Get linear index onto interior NxN nodes of (N+2)x(N+2) grid 
    const int id = (j + 1)*(N + 2) + (i + 1);

    newu[id] = 0.25f*(rhs[id]
		      + u[id - (N+2)]
		      + u[id + (N+2)]
		      + u[id - 1]
		      + u[id + 1]);
  }
}

/* CUDA kernel using shared memory */
__global__ void jacobiShared(const int N,
			     const datafloat *rhs,
			     const datafloat *u,
			     datafloat *newu){

  // Get thread indices
  const int i = blockIdx.x*blockDim.x + threadIdx.x;
  const int j = blockIdx.y*blockDim.y + threadIdx.y;

  __shared__ float s_u[BY+2][BX+2];

  // load block of data in to shared memory
  for(int n=threadIdx.y; n<BY+2; n+=BY){
    for(int m=threadIdx.x; m<BX+2; m+=BX){

      const int i = blockIdx.x*blockDim.x + m;
      const int j = blockIdx.y*blockDim.y + n;
      
      datafloat val = 0.;

      if(i < N+2 &&  j < N+2)
        val = u[j*(N+2) + i];
      
      s_u[n][m] = val;
    }
  }

  // barrier until all the shared memory entries are loaded
  __syncthreads();

  if((i < N) && (j < N)){

    // Get padded grid ID
    const int pid = (j + 1)*(N + 2) + (i + 1);

    datafloat invD = 0.25;

    newu[pid] = invD*(rhs[pid]
		      + s_u[threadIdx.y+0][threadIdx.x+1]
		      + s_u[threadIdx.y+2][threadIdx.x+1]
		      + s_u[threadIdx.y+1][threadIdx.x+0]
		      + s_u[threadIdx.y+1][threadIdx.x+2]
		      );
  }
}


__global__ void partialReduceResidual(const int entries,
				      datafloat *u,
				      datafloat *newu,
				      datafloat *blocksum){

  __shared__ datafloat s_blocksum[BDIM];

  const int id = blockIdx.x*blockDim.x + threadIdx.x;

  s_blocksum[threadIdx.x] = 0;

  if(id < entries){
    const datafloat diff = u[id] - newu[id];
    s_blocksum[threadIdx.x] = diff*diff;
  }

  int alive = blockDim.x;
  int t = threadIdx.x;

  while(alive>1){

    __syncthreads();  // barrier (make sure s_red is ready)
    
    alive /= 2;
    if(t < alive)
      s_blocksum[t] += s_blocksum[t+alive];
  }
  
  if(t==0) 
    blocksum[blockIdx.x] = s_blocksum[0];
}
  


__global__ void unrolledPartialReduceResidual(const int entries,
                               datafloat *u,
                               datafloat *newu,
                               datafloat *red){
  __shared__ datafloat s_red[BDIM];

  const int id = blockIdx.x*blockDim.x + threadIdx.x;

  s_red[threadIdx.x] = 0;

  if(id < entries){
    const datafloat diff = u[id] - newu[id];
    s_red[threadIdx.x] = diff*diff;
  }

  __syncthreads();  // barrier (make sure s_red is ready)

  // manually unrolled reduction (assumes BDIM=256)
  if(BDIM>128) {
    if(threadIdx.x<128)
      s_red[threadIdx.x] += s_red[threadIdx.x+128];

    __syncthreads();  // barrier (make sure s_red is ready)
  }

  if(BDIM>64){
    if(threadIdx.x<64)
      s_red[threadIdx.x] += s_red[threadIdx.x+64];

    __syncthreads();  // barrier (make sure s_red is ready)
  }

  if(BDIM>32){
    if(threadIdx.x<32)
      s_red[threadIdx.x] += s_red[threadIdx.x+32];

    __syncthreads();  // barrier (make sure s_red is ready)
  }

  if(BDIM>16){
    if(threadIdx.x<16)
      s_red[threadIdx.x] += s_red[threadIdx.x+16];

    __syncthreads();  // barrier (make sure s_red is ready)
  }

  if(BDIM>8){
    if(threadIdx.x<8)
      s_red[threadIdx.x] += s_red[threadIdx.x+8];

    __syncthreads();  // barrier (make sure s_red is ready)
  }

  if(BDIM>4){
    if(threadIdx.x<4)
      s_red[threadIdx.x] += s_red[threadIdx.x+4];

    __syncthreads();  // barrier (make sure s_red is ready)
  }

  if(BDIM>2){
    if(threadIdx.x<2)
      s_red[threadIdx.x] += s_red[threadIdx.x+2];

    __syncthreads();  // barrier (make sure s_red is ready)
  }

  if(BDIM>1){
    if(threadIdx.x<1)
      s_red[threadIdx.x] += s_red[threadIdx.x+1];
  }

  // store result of this block reduction
  if(threadIdx.x==0)
    red[blockIdx.x] = s_red[threadIdx.x];
}

void solve(int N, datafloat tol, datafloat *h_rhs, datafloat *h_res, datafloat *h_u, datafloat *h_u2 ){

  const int iterationChunk = 100; // needs to be multiple of 2
  int iterationsTaken = 0;
  
  // Setup jacobi kernel block-grid sizes
  dim3 jBlock(BX,BY);
  dim3 jGrid((N + jBlock.x - 1)/jBlock.x, (N + jBlock.y - 1)/jBlock.y);

  // Setup reduceResidual kernel block-grid sizes
  dim3 rBlock(BDIM);
  dim3 rGrid((N*N + rBlock.x - 1)/rBlock.x);

  // Device Arrays
  datafloat *c_u, *c_u2, *c_rhs, *c_res;

  cudaMalloc((void**) &c_u  , (N+2)*(N+2)*sizeof(datafloat));
  cudaMalloc((void**) &c_u2 , (N+2)*(N+2)*sizeof(datafloat));
  cudaMalloc((void**) &c_rhs ,(N+2)*(N+2)*sizeof(datafloat));

  cudaMalloc((void**) &c_res, rGrid.x*sizeof(datafloat));

  // Setting device vectors to 0
  cudaMemcpy(c_u ,  h_u ,  (N+2)*(N+2)*sizeof(datafloat), cudaMemcpyHostToDevice);
  cudaMemcpy(c_u2,  h_u2,  (N+2)*(N+2)*sizeof(datafloat), cudaMemcpyHostToDevice);
  cudaMemcpy(c_rhs, h_rhs, (N+2)*(N+2)*sizeof(datafloat), cudaMemcpyHostToDevice);

  // Create CUDA events
  cudaEvent_t startEvent, endEvent;
  cudaEventCreate(&startEvent);
  cudaEventCreate(&endEvent);

  cudaEventRecord(startEvent, 0);

  datafloat res;

  do {


    // Call jacobi [iterationChunk] times before calculating residual
    for(int i = 0; i < iterationChunk/2; ++i){

      // first iteration 
      jacobi<<<jGrid, jBlock>>>(N, c_rhs, c_u, c_u2);
      
      // flip flop arguments
      jacobi<<<jGrid, jBlock>>>(N, c_rhs, c_u2, c_u);
    }

    // Calculate residual
    partialReduceResidual<<<rGrid, rBlock>>>(N*N, c_u, c_u2, c_res);

    // Finish reduce in host
    cudaMemcpy(h_res, c_res, rGrid.x*sizeof(datafloat), cudaMemcpyDeviceToHost);

    res = 0;
    for(int i = 0; i < rGrid.x; ++i)
      res += h_res[i];

    res = sqrt(res);

    iterationsTaken += iterationChunk;

    printf("residual = %g after %d steps \n", res, iterationsTaken);

  } while(res > tol);

  cudaEventRecord(endEvent, 0);
  cudaEventSynchronize(endEvent);

  // Get time taken
  float timeTaken;
  cudaEventElapsedTime(&timeTaken, startEvent, endEvent);

  // Copy final solution from device array to host
  cudaMemcpy(h_u, c_u, (N+2)*(N+2)*sizeof(datafloat), cudaMemcpyDeviceToHost);

  const datafloat avgTimePerIteration = timeTaken/((datafloat) iterationsTaken);

  printf("Residual                   : %7.9e\n"     , res);
  printf("Iterations                 : %d\n"        , iterationsTaken);
  printf("Average time per iteration : %3.5e ms\n"  , avgTimePerIteration);
  printf("Bandwidth                  : %3.5e GB/s\n", (1.0e-6)*(6*N*N*sizeof(datafloat))/avgTimePerIteration);

  // free device arrays
  cudaFree(c_u);
  cudaFree(c_u2);
  cudaFree(c_res);
  cudaFree(c_rhs);

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
