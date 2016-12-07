all: cuda thrust flow mflow
cuda: cuda.cu
	nvcc -std=c++11 cuda.cu -o cuda
thrust: thrust.cu
	nvcc -std=c++11 thrust.cu -o thrust
flow: flow.cu
	nvcc -std=c++11 flow.cu -o flow
mflow: mflow.cu
	nvcc -std=c++11 mflow.cu -o mflow
