#define _GNU_SOURCE
#include <dlfcn.h>
#include <stddef.h>
#include <cuda.h>
#include <nvml.h>
#include <stdio.h>
#include <string.h>

#define MEMORY_FRACTION 0.5

struct CudaLibrary {
    CUresult (*cuDeviceTotalMem) (size_t* bytes, CUdevice dev);
    CUresult (*cuCtxCreate) ( CUcontext* pctx, unsigned int flags, CUdevice dev );
    CUresult (*cuMemAlloc) (CUdeviceptr* dptr, size_t bytesize);
    CUresult (*cuMemcpyHtoD) ( CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount );
    CUresult (*cuMemcpyDtoH) ( void* dstHost, CUdeviceptr srcDevice, size_t ByteCount );
    CUresult (*cuMemFree) (CUdeviceptr dptr);
    CUresult (*cuMemGetInfo) (size_t* free, size_t* total);
    nvmlReturn_t (*nvmlDeviceGetMemoryInfo) (nvmlDevice_t device, nvmlMemory_t* memory);
    CUresult (*cuLaunchKernel) ( CUfunction f,
                                 unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ,
                                 unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ,
                                 unsigned int  sharedMemBytes, CUstream hStream, void** kernelParams,
                                 void** extra );
} cuda_lib;

struct RtLib {
    void *(*dlsym) (void *handle, const char *symbol);
} rt_lib;

CUcontext ctx;

struct ld_bindings_s {
    const char *name;
    void **real_address;
};

extern void *__libc_dlsym(void *handle, const char *symbol);

static struct ld_bindings_s bindings[] = {
    {"cuDeviceTotalMem", (void **) &cuda_lib.cuDeviceTotalMem},
    {"cuCtxCreate", (void **) &cuda_lib.cuCtxCreate},
    {"cuMemAlloc", (void **) &cuda_lib.cuMemAlloc},
    {"cuMemFree", (void **) &cuda_lib.cuMemFree},
    {"cuMemcpyHtoD", (void **) &cuda_lib.cuMemcpyHtoD},
    {"cuMemcpyDtoH", (void **) &cuda_lib.cuMemcpyDtoH},
    {"cuLaunchKernel", (void **) &cuda_lib.cuLaunchKernel},

    {"cuMemGetInfo", (void **) &cuda_lib.cuMemGetInfo},
    {"nvmlDeviceGetMemoryInfo", (void **) &cuda_lib.nvmlDeviceGetMemoryInfo},
    {NULL, NULL}
};

static int inited = 0;
static void __attribute__((constructor)) init_bindings(void) {
    int i = 0;

    if (inited) {
        return;
    }
    printf("Loading librt ... \n");

    printf("Loading bindings ...\n");

    void *libdl_handle = dlopen("libdl.so", RTLD_NOW);
    if (libdl_handle == NULL) {
        fprintf(stderr, "ERROR: failed to dlopen(libdl.so)\n");
        return;
    }

    rt_lib.dlsym = dlvsym(libdl_handle, "dlsym", "GLIBC_2.2.5");
    printf("dlsym address: %p\n", rt_lib.dlsym);

    while (bindings[i].name) {
        //printf("Loading bindings ... %s\n", bindings[i].name);
        *bindings[i].real_address = rt_lib.dlsym(RTLD_NEXT, bindings[i].name);
        //printf("Loading bindings ... %p\n", rt_lib.dlsym(RTLD_NEXT, bindings[i].name));
        //printf("Loading bindings ... %p\n", *bindings[i].real_address);
        if (*bindings[i].real_address == NULL) {
            fprintf(stderr, "Error in `dlsym` of %s: %s\n",
                    bindings[i].name, dlerror());
        }
        i++;
    }
    printf("Loading bindings ... done.\n");
    inited = 1;
}

static void __attribute__((constructor)) init_memory(void) {
    printf("Initializing CUDA memory hack ...\n");

    printf("Initializing CUDA memory hack ... done.\n");
}

static struct DeviceMemory *getMemoryStruct() {
    return NULL;
}

/***************************/

void *dlsym(void *handle, const char *symbol) {
    if (strcmp(symbol, "nvmlDeviceGetMemoryInfo") == 0) {
        cuda_lib.nvmlDeviceGetMemoryInfo = rt_lib.dlsym(handle, symbol);
        return &nvmlDeviceGetMemoryInfo;
    }

    return rt_lib.dlsym(handle, symbol);
}

/***************************/

CUresult cuCtxCreate(CUcontext* pctx, unsigned int flags, CUdevice dev) {
    return cuda_lib.cuCtxCreate(pctx, flags, dev);
}

CUresult cuMemcpyHtoD ( CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount ) {
    CUcontext ctx;
    cuCtxGetCurrent(&ctx);
    printf("# cuMemcpyHtoD context is: %p\n", ctx);
    CUdevice device;
    CUresult r = cuCtxGetDevice (&device) ;
    char name[100];
    cuDeviceGetName(name, 100, device);
    printf("> Using device 0: %s [%d]\n", name, __LINE__);

    return cuda_lib.cuMemcpyHtoD(dstDevice, srcHost, ByteCount);
}

CUresult cuMemcpyDtoH ( void* dstHost, CUdeviceptr srcDevice, size_t ByteCount ) {
    CUcontext ctx;
    cuCtxGetCurrent(&ctx);
    printf("# cuMemcpyDtoH context is: %p\n", ctx);
    return cuda_lib.cuMemcpyDtoH(dstHost, srcDevice, ByteCount);
}

CUresult cuLaunchKernel( CUfunction f,
                         unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ,
                         unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ,
                         unsigned int  sharedMemBytes, CUstream hStream, void** kernelParams,
                         void** extra ) {
    printf("# cuLaunchKernel\n");

    return cuda_lib.cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ,
                                   blockDimX, blockDimY, blockDimZ,
                                   sharedMemBytes, hStream, kernelParams, extra);
}

// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gc6a0d6551335a3780f9f3c967a0fde5d
/*
Returns the total amount of memory on the device.
*/
CUresult cuDeviceTotalMem (size_t *bytes, CUdevice dev)  {
    CUresult ret = cuda_lib.cuDeviceTotalMem(bytes, dev);
    if (ret != CUDA_SUCCESS) {
        return ret;
    }

    size_t reserved = *bytes * MEMORY_FRACTION;
    *bytes -= reserved;

    return CUDA_SUCCESS;
}

// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467
/*
Allocates device memory.
 dptr: Returned device pointer
 bytesize: Requested allocation size in bytes
*/
CUresult cuMemAlloc (CUdeviceptr* dptr, size_t bytesize) {
    /*
    size_t free, total;
    CUresult ret = cuda_lib.cuMemGetInfo(&free, &total);
    if (ret != CUDA_SUCCESS) {
        return ret;
    }

    if (bytesize >= free) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    */
    CUcontext ctx;
    CUresult r = cuCtxGetCurrent (&ctx) ;
    printf("# cuMemAlloc context is %p\n", ctx);
    return cuda_lib.cuMemAlloc(dptr, bytesize);
}

// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g808f555540d0143a331cc42aa98835c0
CUresult cuMemGetInfo (size_t* free, size_t* total) {
    CUresult ret = cuda_lib.cuMemGetInfo(free, total);
    if (ret != CUDA_SUCCESS) {
        return ret;
    }

    size_t reserved = *total * MEMORY_FRACTION;
    if (*free < reserved) {
        *free = 0;
    } else {
        *free -= reserved;
    }
    *total -= reserved;

    return CUDA_SUCCESS;
}

// https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html#group__nvmlDeviceQueries_1g2dfeb1db82aa1de91aa6edf941c85ca8
nvmlReturn_t nvmlDeviceGetMemoryInfo (nvmlDevice_t device, nvmlMemory_t* memory) {
    printf("Running nvmlDeviceGetMemoryInfo\n");
    nvmlReturn_t ret = cuda_lib.nvmlDeviceGetMemoryInfo(device, memory);
    if (ret != NVML_SUCCESS) {
        return ret;
    }

    unsigned long long reserved = memory->total * MEMORY_FRACTION;
    memory->total -= reserved;

    return NVML_SUCCESS;
}
