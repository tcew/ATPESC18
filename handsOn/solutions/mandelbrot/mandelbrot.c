
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define MXITER 1000
#define NPOINTS 2048

typedef struct {
  
  double r;
  double i;
  
}d_complex;

// return 1 if c is outside the mandelbrot set
// reutrn 0 if c is inside the mandelbrot set
int testpoint(d_complex c){
  
  d_complex z;
  
  int iter;
  double temp;
  
  z = c;
  
  for(iter=0; iter<MXITER; iter++){
    
    temp = (z.r*z.r) - (z.i*z.i) + c.r;
    
    z.i = z.r*z.i*2. + c.i;
    z.r = temp;
    
    if((z.r*z.r+z.i*z.i)>4.0){
      return 1;
    }
  }
  
  return 0;
  
}

int  mandeloutside(){

  int i,j;
  double eps = 1e-5;

  d_complex c;

  int numoutside = 0;

#pragma omp parallel for reduction(+:numoutside) private(j,c)
  for(i=0;i<NPOINTS;i++){
    for(j=0;j<NPOINTS;j++){
      c.r = -2. + 2.5*(double)(i)/(double)(NPOINTS)+eps;
      c.i =       1.125*(double)(j)/(double)(NPOINTS)+eps;
      numoutside += testpoint(c);
    }
  }

  return numoutside;
}

int main(int argc, char **argv){

  double start = omp_get_wtime();
 
  double numoutside = mandeloutside();
  
  double end = omp_get_wtime();
  
  printf("elapsed = %g\n", end-start);

  double area = 2.*2.5*1.125*(NPOINTS*NPOINTS-numoutside)/(NPOINTS*NPOINTS);

  printf("area = %17.15lf\n", area);

  return 0;
}  
