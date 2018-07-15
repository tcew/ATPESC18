flags = -m64 -O3 

ifeq ($(shell uname -s),Darwin)
libs = -framework OpenCL
else
libs = -lOpenCL
endif


main: main.cpp
	g++ -o main $(flags) main.cpp $(libs)

clean:
	rm -f main
