/* Derived from MLIFE exercise */

#include <stdio.h>
#include <stdlib.h>
#include "occa.hpp"

#define BORN 1
#define DIES 0

#define id(r,c) ((r)*Ncolumns+(c))

/* build board */
void init(int Nrows, int Ncolumns, int **board, int **newboard, 
	  occa::device &device, occa::kernel &update, occa::memory &c_board, occa::memory &c_newboard){

  int r,c,n;

  *board    = (int*) calloc(Nrows*Ncolumns, sizeof(int));
  *newboard = (int*) calloc(Nrows*Ncolumns, sizeof(int));

  /* death at the border */
  for(r=0;r<Nrows;++r){
    (*board)[id(r,0)] = DIES;
    (*board)[id(r,Ncolumns-1)] = DIES;

    (*newboard)[id(r,0)] = DIES;
    (*newboard)[id(r,Ncolumns-1)] = DIES;
  }
  for(c=0;c<Ncolumns;++c){
    (*board)[id(0,c)] = DIES;
    (*board)[id(Nrows-1,c)] = DIES;

    (*newboard)[id(0,c)] = DIES;
    (*newboard)[id(Nrows-1,c)] = DIES;
  }

  /* random life */
  srand48(12345);
  for(r=1;r<Nrows-1;++r){
    for(c=1;c<Ncolumns-1;++c){
      double rn = drand48();
      (*board)[id(r,c)] = BORN*(rn<0.5) + DIES*(rn>=0.5);
    }
  }

  /* EX01: allocate DEVICE arrays for c_board and c_newboard here using cudaMalloc */
  /* EX02: copy board state from HOST board to DEVICE c_board using cudaMemcpy */
  //  device.setup("mode = CUDA, deviceID = 0");
  //device.setup("mode = OpenCL, deviceID = 0, platformID = 0");
  device.setup("mode = OpenMP  , schedule = compact, chunk = 10");
  //  device.setup("mode = Serial");

  update = device.buildKernelFromSource("update.okl", "update");

  c_board = device.malloc(Nrows*Ncolumns*sizeof(int), *board);
  c_newboard = device.malloc(Nrows*Ncolumns*sizeof(int), *newboard);
  
}

void destroy(int *board, int *newboard){
  free(board);
  free(newboard);
}


/* EX04: add a copy from DEVICE to HOST using cudaMemcpy */
void print(int Nrows, int Ncolumns, int *board, occa::memory &c_board){

  /* EX04: put cudaMemcpy here to copy from DEVICE c_board to HOST board*/
  c_board.copyTo(board);

  /* No need tochange this bit */
  system("clear");
  for(int r=0;r<Nrows;++r){
    for(int c=0;c<Ncolumns;++c){
      if(board[id(r,c)]==BORN) printf("*");
      else printf(" ");
    }
    printf("\n");
  }
}


int main(int argc, char **argv){

  if(argc<3){
    printf("usage: main [Nrows] [Ncolumns]\n");
    exit(1);
  }

  /* initialize board */
  int Nrows    = atoi(argv[1]);
  int Ncolumns = atoi(argv[2]);
  int *board, *newboard;

  /* occa objects */
  occa::device device;
  occa::memory c_board, c_newboard;
  occa::kernel update;

  init(Nrows, Ncolumns, &board, &newboard, 
       device, update, c_board, c_newboard);
  
  /* run some iterations */
  int Nit = 100;
  for(int it=0;it<Nit;++it){
    
    /* EX05b: add kernel launch syntax here */
    update(Nrows, Ncolumns, c_board, c_newboard);

    /* EX05c: add kernel launch syntax here */
    update(Nrows, Ncolumns, c_newboard, c_board);
    
    print(Nrows, Ncolumns, board, c_board);
  }

  destroy(board, newboard);

  exit(0);
  return 0;
}
