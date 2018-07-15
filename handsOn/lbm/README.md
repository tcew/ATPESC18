# lbm
Lattice Boltzmann Reference Code: simple implementations in Serial C, CUDA, and using the OCCA portability library.

# Instructions:
To clone from github:
 
git clone https://github.com/tcew/lbm

## To compile and run the serial LBM code
make serial

./serialLBM images/fsm.png 400

## To compile and run the CUDA LBM code
make cuda

./cudaLBM images/fsm.png 400

## To compile the OCCA library 
git clone https://github.com/libocca/occa occa

cd occa

make -j

export OCCA_DIR=`pwd`

cd ../../

make -f makefile.occa

## To compile and run the OCCA LBM code (notice the different makefile)
make -f makefile.occa 

./occaLBM images/fsm.png 400

# To create movie
ffmpeg -start_number 0 -r 24 -i bah%06d.png -c:v mpeg4 test.mp4
