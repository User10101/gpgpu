all: main cpu_main
main: main.cu ../include/auxf.h
	nvcc -I ../include main.cu -o $@
cpu_main: mult.cpp ../include/auxf.h
	g++ -std=c++11 -I ../include mult.cpp -o $@
