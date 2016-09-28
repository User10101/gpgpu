all : main main1 main2 main3
main : main.cu
	nvcc -Xptxas -dlcm=cg main.cu -o main
main1 : main1.cu
	nvcc -Xptxas -dlcm=cg main1.cu -o main1
main2 : main2.cu
	nvcc main2.cu -o main2
main3 : main3.cu
	nvcc -Xptxas -dlcm=cg main3.cu -o main3
