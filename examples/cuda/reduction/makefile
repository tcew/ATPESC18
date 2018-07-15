cudaFlags =  -m64 -O3 --use_fast_math

# To use clang in OSX
ifeq ($(shell uname -s),Darwin)
	cudaFlags += -ccbin /usr/bin/clang++ -Xcompiler -stdlib=libstdc++
endif

main: main.cu
	nvcc -o main $(cudaFlags) main.cu

clean:
	rm -f main
