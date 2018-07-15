/* Derived from MLIFE exercise */

/* To compile: gcc -o life life.c */
/* To run on a grid of 32x128: ./life 32 128 */

#include <stdio.h>
#include <stdlib.h>

#define BORN 1
#define DIES 0

#define id(r,c) ((r)*Ncolumns+(c))

/* build board */
void init(int Nrows, int Ncolumns, int **board, int **newboard){

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
}

void destroy(int *board, int *newboard){
  free(board);
  free(newboard);
}

void update(int Nrows, int Ncolumns, int *board, int *newboard){
  int r,c;
  
  for(r=1;r<Nrows-1;++r){
    for(c=1;c<Ncolumns-1;++c){
      int s = 
	board[id(r-1,c-1)]+board[id(r-1,c-0)]+board[id(r-1,c+1)]+
	board[id(r+0,c-1)]+                   board[id(r+0,c+1)]+
	board[id(r+1,c-1)]+board[id(r+1,c-0)]+board[id(r+1,c+1)];

      newboard[id(r,c)]
	= (s<2)*DIES + (s==2)*board[id(r,c)] + (s==3)*BORN + (s>3)*DIES;
    }
  }
}

void print(int Nrows, int Ncolumns, int *board){
  int r,c;

  system("clear");
  for(r=0;r<Nrows;++r){
    for(c=0;c<Ncolumns;++c){
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

  init(Nrows, Ncolumns, &board, &newboard);

  /* run some iterations */
  int Nit = 400, it;
  for(it=0;it<Nit;++it){
    update(Nrows, Ncolumns, board, newboard);
    update(Nrows, Ncolumns, newboard, board);

    print(Nrows, Ncolumns, board);
    
  }

  destroy(board, newboard);

  exit(0);
  return 0;
}
