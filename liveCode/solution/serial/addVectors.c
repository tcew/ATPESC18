
#include <stdio.h>
#include <stdlib.h>

void addVectors(int N, double *a, double *b, double *c){

  int n;
  for(n=0;n<N;++n){
    c[n] = a[n] + b[n];
  }

}


int main(int argc, char **argv){

  int N = 1000;
  double *a = (double*) malloc(N*sizeof(double));
  double *b = (double*) malloc(N*sizeof(double));
  double *c = (double*) malloc(N*sizeof(double));

  int n;

  for(n=0;n<N;++n){
    a[n] = 1+n;
    b[n] = 1-n;
  }

  addVectors(N, a, b, c);

  for(n=0;n<5;++n){
    printf("c[%d] = %g\n", n, c[n]);
  }

  return 0;
}



