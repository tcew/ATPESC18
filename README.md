# ATPESC18
GPU programming module for Argonne Training Program in Extreme Scale Computing 2018

Instructions for Accelerator Presentation

Slides: www.math.vt.edu/people/tcew/ATPESC18 

OCCA repo (version 0.2 branch): `git clone https://github.com/libocca/occa -b 0.2`

Paranumal website: http://www.paranumal.com

Paranumal blog including tips on GPU performance: https://www.paranumal.com/blog 

Ray Loy's on-boarding ALCF slides: https://extremecomputingtraining.anl.gov/sessions/presentation-quick-start-on-atpesc-resources/  

## Examples: 

To access the demos from the GPU programming lecture:

`git clone https://github.com/tcew/ATPESC18`  

Note: `You may need to use ssh-keygen on cooley and update the github ssh public keys associated with the ssh public key from your cooley account)`  

## Flow animation hands-on:

Follow these instructions to get access to a cooley GPU node and configure for CUDA

### 1. ssh to cooley
`ssh username@cooley.alcf.anl.gov`  

### 2.  Make sure these options or similar appear in your .soft.cooley file:
`+mvapich2`  
`+cuda-7.5.18`  
`+ffmpeg-1.0.1`  
`@default`  

### 3. make sure packages are loaded
`resoft`  

## CUDA version of flow simulation

#### A. get files for GPU tutorial
`git clone https://github.com/tcew/ATPESC18`  

#### B. build Lattice Boltzmann flow solver
`cd ATPESC18/handsOn`  
`cd lbm`  
`make`  

#### C. submit job request
`qsub -A ATPESC2018 -I -n 1 -t 120 -q training`  

#### D. change path/name of your png file to your image
`./cudaLBM ./images/fsm.png 400`  

#### E. create movie
`ffmpeg -start_number 0 -r 24 -i bah%06d.png -c:v mpeg4 test.mp4`  

## OCCA version of flow simulation

### A. retrieve OCCA
`git clone https://github.com/libocca/occa -b 0.2`  

### B. retrieve ATPESC18
`git clone https://github.com/tcew/ATPESC18`  

### C. submit job request
`qsub -A ATPESC2018 -I -n 1 -t 120 -q training`  

### D. build OCCA
`cd occa`  
export OCCA_DIR=\`pwd\`
`export LD_LIBRARY_PATH=$OCCA_DIR/lib:$LD_LIBRARY_PATH:$OCCA_DIR/lib`  
`make -j`  
`./bin/occainfo`  

### E. build Lattice Boltzmann flow solver
`cd ../ATPESC18/handsOn`  
`cd lbm`  
`make -f makefile.occa`  

### F. run the lbm code with an example png image (using 400 as a flow volume threshold)
`./occaLBM ./images/fsm.png 400`  

### G. on the compute node convert the output bah png files to foo.mp4
`ffmpeg -r 24 -i bah%06d.png -b:v 16384k -vf scale=1024:-1 foo.mp4`  
`ffmpeg -r 24 -i bah%06d.png -b:v 16384k -vf scale=768:-1 -vcodec mpeg4 foo.mp4`  



