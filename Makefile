all: driverTest libCudaMemory.so matSumKernel.ptx

driverTest: driverTest.c matSumKernel.h
	nvcc -g -lcuda -o $@ $< -Xcompiler -Wno-deprecated-declarations

libCudaMemory.so: cuda_memory.o
	gcc -shared -o $@ $<

cuda_memory.o: cuda_memory.c
	nvcc -g -Xcompiler -fPIC -o $@ -c $<

matSumKernel.ptx: matSumKernel.cu
	nvcc -arch=sm_70 -ptx -o $@ $<

run: all
	./driverTest

preload: all
	env \
	  LD_PRELOAD=./libCudaMemory.so \
	  ./driverTest
debug: all
	gdb -ex 'set env LD_PRELOAD=./libCudaMemory.so' ./driverTest
