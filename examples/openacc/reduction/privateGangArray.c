//  pgcc -fast -openmp -acc -Minfo=accel -ta=nvidia,keepgpu,keepptx -o privateGangArray privateGangArray.c -lm

#include <stdlib.h>
#include <time.h>
#include <openacc.h>
#include <omp.h>

int main(int argc, char **argv){

  
  // hard coded for vector of size T=128
#define B 30000
#define T 256

  int N = 32*B*T; // ratio 32 to 1: 200GB/s on OpenACC on hokiespeed

  float sumv1[T];
  float *v    = (float*) calloc(N, sizeof(float));
  float *redv = (float*) calloc(B, sizeof(float));

  for(int n=0;n<N;++n){
    v[n] = 1;
  }

  float *d_v    = (float*) acc_malloc(N*sizeof(float));
  float *d_redv = (float*) acc_malloc(B*sizeof(float));

  acc_memcpy_to_device(d_v, v, N*sizeof(float));
  acc_memcpy_to_device(d_redv, redv, B*sizeof(float));
  
  double c1 = omp_get_wtime();
  
  // #pragma acc data  copyin(v[0:B*T]), copyout(redv[0:B])
#pragma acc data  deviceptr(d_v, d_redv)
#pragma acc parallel num_gangs(B), vector_length(T), private(sumv1)
  {
#pragma acc loop gang 
    for(int b=0;b<B;++b){

#pragma acc loop vector
      for(int t=0;t<T;++t){
	int id = t + b*T;
	float tmp  = 0.f;

	while(id<N){
	  tmp = d_v[id];
	  id += B*T;
	}
	sumv1[t] = tmp;
      }

      //#pragma acc loop vector
      //      for(int t=0;t<T;++t)
      //	if(T>=512 && t<256) sumv1[t] += sumv1[t+256];

#pragma acc loop vector
      for(int t=0;t<T;++t)
	if(T>=256 && t<128) sumv1[t] += sumv1[t+128];

#pragma acc loop vector
      for(int t=0;t<T;++t)
	if(T>=128 && t<64) sumv1[t] += sumv1[t+64];
      
#pragma acc loop vector
      for(int t=0;t<T;++t)
	if(T>=64 && t<32) sumv1[t] += sumv1[t+32];

#pragma acc loop vector
      for(int t=0;t<T;++t)
	if(t<16) sumv1[t] += sumv1[t+16];
      
#pragma acc loop vector
      for(int t=0;t<T;++t)
	if(t<8) sumv1[t] += sumv1[t+8];

#pragma acc loop vector
      for(int t=0;t<T;++t)
	if(t<4) sumv1[t] += sumv1[t+4];
      
#pragma acc loop vector
      for(int t=0;t<T;++t)
	if(t<2) sumv1[t] += sumv1[t+2];

#pragma acc loop vector
      for(int t=0;t<T;++t)
	if(t<1){
	  sumv1[t] += sumv1[t+1];
	  d_redv[b] = sumv1[0];
	}
      //#pragma acc loop vector
      //      for(int t=0;t<T;++t){
      //	if(t==0) 
      //      }
    }  
  }

  acc_wait_all();

  double c2 = omp_get_wtime();

  /* convert from miliseconds to seconds */
  double elapsed_time = (c2-c1);

  /* output elapsed time */
  printf("elapsed time for cpu: %g\n", elapsed_time);
  printf("BW: %g GB/s\n", (N+1.*B)*sizeof(float)/(1.e9*elapsed_time));
  printf("N=%d (%g GB)\n", N, N*sizeof(float)/1.e9);


  acc_memcpy_from_device(redv, d_redv, B*sizeof(float));

  // on HOST
  for(int b=0;b<B;++b){
    if(b<10 || b>B-10)
    printf("redv[%03d]=%g\n", b, redv[b]);
  }

  free(v);
  free(redv);

}
