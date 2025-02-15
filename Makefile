CXX=mpicxx
files=$(wildcard *.cpp)
programs=$(files:%.cpp=%)

Includes=-I/usr/include/opencv4
CXXFLAGS=-O0 $(Includes) -fopenmp
LDLIBS=-L/usr/lib/x86_64-linux-gnu/ $(shell pkg-config --libs opencv4)

all: $(programs)

clean:
	rm $(programs)