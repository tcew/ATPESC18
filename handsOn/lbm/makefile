# define variables
HDRDIR  = ./

# set options for this machine
# specify which compilers to use for c and linking
CC	= gcc
LD	= gcc
NCC	= nvcc
NLD	= nvcc

# compiler flags to be used (set to compile with debugging on)
CFLAGS = -I$(HDRDIR)  -fopenmp -g -O3
NCFLAGS = -I$(HDRDIR)   -g --use_fast_math --fmad=true -arch=sm_30


# link flags to be used 
LDFLAGS	=  -fopenmp -g -O3
NLDFLAGS =

# libraries to be linked in
LIBS	=  -lpng -lm 

# types of files we are going to construct rules for
.SUFFIXES: .c .cu

# rule for .c files
.c.o:
	$(CC) $(CFLAGS) -o $*.o -c $*.c
.cu.o:
	$(NCC) $(NCFLAGS) -o $*.o -c $*.cu 

# list of objects to be compiled
SOBJS    = serialLBM.o png_util.o
OOBJS    = openmpLBM.o png_util.o
COBJS    = cudaLBM.o png_util.o

all: serial openmp cuda

serial:$(SOBJS) 
	$(LD)  $(LDFLAGS) -o serialLBM $(SOBJS) $(LIBS)

openmp:$(OOBJS) 
	$(LD)  $(LDFLAGS) -o openmpLBM $(OOBJS) $(LIBS)

cuda:$(COBJS) 
	$(NLD)  $(NLDFLAGS) -o cudaLBM $(COBJS) $(LIBS)


# what to do if user types "make clean"
clean :
	rm -r $(OOBJS) $(NOBJS) $(COBJS)
