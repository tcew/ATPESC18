cudaFlags = -arch=sm_30 -m64 -O3 --use_fast_math

# To use clang in OSX
ifeq ($(shell uname -s),Darwin)
	cudaFlags += -ccbin /usr/bin/clang++ -Xcompiler -stdlib=libstdc++
endif

addVectors: addVectors.cu
	nvcc -o addVectors $(cudaFlags) addVectors.cu

clean:
	rm -f addVectors 
