flags = -m64 -O3 

ifeq ($(shell uname -s),Darwin)
libs = -framework OpenCL
else
libs = -lOpenCL
endif


simple: simple.cpp
	g++ -o simple $(flags) simple.cpp $(libs)

clean:
	rm -f simple.cp
