/*
 * drivertest.cpp
 * Vector addition (host code)
 *
 * Andrei de A. Formiga, 2012-06-04
 * https://gist.github.com/tautologico/2879581
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <cuda.h>
#include <builtin_types.h>

#include "matSumKernel.h"

// This will output the proper CUDA error strings
// in the event that a CUDA host call returns an error
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

static inline void __checkCudaErrors(CUresult err, const char *file, const int line )
{
    if(CUDA_SUCCESS != err) {
        fprintf(stderr,
                "CUDA Driver API error = %d from file <%s>, line %i.\n",
                err, file, line );
        exit(-1);
    }
}

// --- global variables ----------------------------------------------------
CUdevice   device;
CUcontext  context;
CUmodule   module;
CUfunction function;
size_t     totalGlobalMem;

char       *module_file = (char*) "matSumKernel.ptx";
char       *kernel_name = (char*) "matSum";


#define BUFFER_SIZE sizeof(char)*1024*1024 * 512 // 512 MB
#define NB_BUFFERS 8 // 4 GB
#define RESERVED_MB 100

CUdeviceptr buffers[NB_BUFFERS];

// --- functions -----------------------------------------------------------
void initCUDA()
{
    int deviceCount = 0;
    CUresult err = cuInit(0);
    int major = 0, minor = 0;

    if (err == CUDA_SUCCESS)
        checkCudaErrors(cuDeviceGetCount(&deviceCount));

    if (deviceCount == 0) {
        fprintf(stderr, "Error: no devices supporting CUDA\n");
        exit(-1);
    }

    // get first CUDA device
    checkCudaErrors(cuDeviceGet(&device, 0));
    char name[100];
    cuDeviceGetName(name, 100, device);
    printf("> Using device 0: %s\n", name);

    // get compute capabilities and the devicename
    checkCudaErrors(cuDeviceComputeCapability(&major, &minor, device) );
    printf("> GPU Device has SM %d.%d compute capability\n", major, minor);

    checkCudaErrors(cuDeviceTotalMem(&totalGlobalMem, device) );
    printf("  Total amount of global memory:   %llu bytes\n",
           (unsigned long long)totalGlobalMem);
    printf("  64-bit Memory Address:           %s\n",
           (totalGlobalMem > (unsigned long long)4*1024*1024*1024L)?
           "YES" : "NO");

    err = cuCtxCreate(&context, 0, device);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "* Error initializing the CUDA context.\n");
        cuCtxDetach(context);
        exit(-1);
    }

    err = cuModuleLoad(&module, module_file);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "* Error loading the module %s %d\n", module_file, err);
        cuCtxDetach(context);
        exit(-1);
    }

    err = cuModuleGetFunction(&function, module, kernel_name);

    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "* Error getting kernel function %s\n", kernel_name);
        cuCtxDetach(context);
        exit(-1);
    }
}

void finalizeCUDA()
{
    cuCtxDetach(context);
}

void setupDeviceMemory(void)
{
    int i;
    size_t total, free;

    for (i = 0; i < NB_BUFFERS; i++) {
        fprintf(stderr, "* Allocating buffer %d\n", i);
        checkCudaErrors(cuMemAlloc(&buffers[i], BUFFER_SIZE));
        checkCudaErrors(cuMemGetInfo(&free, &total));
        fprintf(stderr, "@used=%d MB, total=%dMB\n", (total-free)/1024/1024 - RESERVED_MB, total/1024/1024);
        sleep(2);
    }

    fprintf(stderr, "* Waiting 30s ...\n");
    sleep(30);

    for (i = 0; i < NB_BUFFERS; i++) {
        fprintf(stderr, "* Releasing buffer %d\n", i);
        checkCudaErrors(cuMemFree(buffers[i]));
        checkCudaErrors(cuMemGetInfo(&free, &total));
        fprintf(stderr, "@used=%d MB, total=%dMB\n", (total-free)/1024/1024 - RESERVED_MB, total/1024/1024);
        sleep(10);
    }
}


int main(int argc, char **argv)
{
    // initialize
    printf("- Initializing cuda ...\n");
    initCUDA();

    // allocate memory
    printf("- Allocating memory...\n");
    setupDeviceMemory();
    printf("- Done.\n");

    finalizeCUDA();
    return 0;
}
