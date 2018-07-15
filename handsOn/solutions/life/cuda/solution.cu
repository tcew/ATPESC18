/* Derived from MLIFE exercise */

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define BORN 1
#define DIES 0

#define id(r,c) ((r)*Ncolumns+(c))

/* build board */
void init(int Nrows, int Ncolumns, int **board, int **newboard, int **c_board, int **c_newboard){

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
  cudaMalloc(c_board,    Nrows*Ncolumns*sizeof(int));
  cudaMalloc(c_newboard, Nrows*Ncolumns*sizeof(int));

  /* EX02: copy board state from HOST board to DEVICE c_board using cudaMemcpy */
  cudaMemcpy(*c_board,    *board, Nrows*Ncolumns*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(*c_newboard, *newboard, Nrows*Ncolumns*sizeof(int), cudaMemcpyHostToDevice);
}

void destroy(int *board, int *newboard){
  free(board);
  free(newboard);
}

/* EX03: convert this to a CUDA kernel */
/* EX03a: annotate to indicate a kernel */
__global__ void update(int Nrows, int Ncolumns, int *board, int *newboard){

  /* EX03b: replace double loop with 2D thread array */
  /* EX03c: convert thread indices and block indices into r,c */

  int r = 1 + threadIdx.y + blockIdx.y*blockDim.y;
  int c = 1 + threadIdx.x + blockIdx.x*blockDim.x;
  
  if(r<Nrows-1 && c<Ncolumns-1){
    /* this does not change */
    int s = 
      board[id(r-1,c-1)]+board[id(r-1,c-0)]+board[id(r-1,c+1)]+
      board[id(r+0,c-1)]+                   board[id(r+0,c+1)]+
      board[id(r+1,c-1)]+board[id(r+1,c-0)]+board[id(r+1,c+1)];
    
    newboard[id(r,c)]
      = (s<2)*DIES + (s==2)*board[id(r,c)] + (s==3)*BORN + (s>3)*DIES;
  }
}

/* EX04: add a copy from DEVICE to HOST using cudaMemcpy */
void print(int Nrows, int Ncolumns, int *board, int *c_board){

  /* EX04: put cudaMemcpy here to copy from DEVICE c_board to HOST board*/
  cudaMemcpy(board, c_board, Nrows*Ncolumns*sizeof(int), cudaMemcpyDeviceToHost);

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
  int *c_board, *c_newboard;
  
  init(Nrows, Ncolumns, &board, &newboard, &c_board, &c_newboard);

  /* run some iterations */
  int Nit = 100;
  for(int it=0;it<Nit;++it){
    
    /* EX05a: define thread-block size and grid size here */
    int T = 16;
    dim3 bDim(T,T,1);
    dim3 gDim((Ncolumns-2+T-1)/T, (Nrows-2+T-1)/T,1);
    
    /* EX05b: add kernel launch syntax here */
    update <<< gDim, bDim >>> (Nrows, Ncolumns, c_board, c_newboard);

    /* EX05c: add kernel launch syntax here */
    update <<< gDim, bDim >>> (Nrows, Ncolumns, c_newboard, c_board);
    
    print(Nrows, Ncolumns, board, c_board);
  }

  destroy(board, newboard);

  exit(0);
  return 0;
}
