//
//  ofxLibRealSense2Gpu.cpp
//

#include "ofxLibRealSense2Gpu.h"
#include "ofxLibRealSense2Processing.h"

#include <librealsense2/rsutil.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <mutex>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#define OFXRS2_GPU_API __stdcall
#else
#include <dlfcn.h>
#define OFXRS2_GPU_API
#endif

namespace
{
    // The kernels: source for OpenCL, PTX for CUDA.
    const char kernelSource[] =
#include "ofxLibRealSense2Kernels.h"
        ;
#include "ofxLibRealSense2KernelsPtx.h"

    enum Kernel
    {
        K_DEPTH_TO_DISPARITY,
        K_DISPARITY_TO_DEPTH,
        K_SPATIAL_ROWS,
        K_SPATIAL_COLUMNS,
        K_TEMPORAL,
        K_DEPTH_TO_MM,
        K_COLOR_MAP,
        K_ALIGN_SPLAT,
        K_FILL32,
        K_ALIGNED_TO_MM,
        K_CLOUD_SAMPLES,
        K_SCAN,
        K_NORMALS,
        K_TRANSFORM,
        K_SPATIAL_ROWS_SHARED,
        K_TRANSPOSE,
        K_POINTS_ROW_COUNT,
        K_POINTS_COMPACT,
        K_TRIANGLES_ROW_COUNT,
        K_TRIANGLES_EMIT,
        K_COUNT
    };
    const char* kernelNames[K_COUNT] = {
        "k_depthToDisparity", "k_disparityToDepth", "k_spatialRows", "k_spatialColumns", "k_temporal", "k_depthToMM",
        "k_colorMap", "k_alignSplat", "k_fill32", "k_alignedToMM", "k_cloudSamples", "k_scan", "k_normals", "k_transform",
        "k_spatialRowsShared", "k_transpose", "k_pointsRowCountGroup", "k_pointsCompactGroup", "k_trianglesRowCountGroup",
        "k_trianglesEmitGroup",
    };
    // As in the kernels
    const int groupSize = 128;
    const int maxLine = 2048;

    void* openLibrary(const std::vector<const char*>& names)
    {
        for (const char* name : names)
        {
#ifdef _WIN32
            if (HMODULE h = LoadLibraryA(name)) return reinterpret_cast<void*>(h);
#else
            if (void* h = dlopen(name, RTLD_NOW | RTLD_LOCAL)) return h;
#endif
        }
        return nullptr;
    }

    void* librarySymbol(void* library, const char* name)
    {
#ifdef _WIN32
        return reinterpret_cast<void*>(GetProcAddress(reinterpret_cast<HMODULE>(library), name));
#else
        return dlsym(library, name);
#endif
    }

    template<typename F>
    bool loadSymbol(void* library, F& f, const char* name)
    {
        f = reinterpret_cast<F>(librarySymbol(library, name));
        return f != nullptr;
    }

    // ------------------------------------------------------------------------------------
    // CUDA driver API, the parts used here (cuda.h, declared locally so no Toolkit is needed)
    // ------------------------------------------------------------------------------------

    typedef int CUresult;
    typedef int CUdevice;
    typedef unsigned long long CUdeviceptr;
    typedef struct CUctx_st* CUcontext;
    typedef struct CUmod_st* CUmodule;
    typedef struct CUfunc_st* CUfunction;
    typedef struct CUstream_st* CUstream;
    const int CU_JIT_ERROR_LOG_BUFFER = 5;
    const int CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6;
    const unsigned int CU_STREAM_NON_BLOCKING = 1;
    const int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75;
    const int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76;

    struct CudaApi
    {
        bool loaded = false;
        std::string error;
        CUresult(OFXRS2_GPU_API* cuInit)(unsigned int);
        CUresult(OFXRS2_GPU_API* cuDeviceGetCount)(int*);
        CUresult(OFXRS2_GPU_API* cuDeviceGet)(CUdevice*, int);
        CUresult(OFXRS2_GPU_API* cuDeviceGetName)(char*, int, CUdevice);
        CUresult(OFXRS2_GPU_API* cuDeviceGetAttribute)(int*, int, CUdevice);
        CUresult(OFXRS2_GPU_API* cuDevicePrimaryCtxRetain)(CUcontext*, CUdevice);
        CUresult(OFXRS2_GPU_API* cuDevicePrimaryCtxRelease)(CUdevice);
        CUresult(OFXRS2_GPU_API* cuCtxSetCurrent)(CUcontext);
        CUresult(OFXRS2_GPU_API* cuModuleLoadDataEx)(CUmodule*, const void*, unsigned int, int*, void**);
        CUresult(OFXRS2_GPU_API* cuModuleUnload)(CUmodule);
        CUresult(OFXRS2_GPU_API* cuModuleGetFunction)(CUfunction*, CUmodule, const char*);
        CUresult(OFXRS2_GPU_API* cuMemAlloc)(CUdeviceptr*, size_t);
        CUresult(OFXRS2_GPU_API* cuMemFree)(CUdeviceptr);
        CUresult(OFXRS2_GPU_API* cuMemAllocHost)(void**, size_t);
        CUresult(OFXRS2_GPU_API* cuMemFreeHost)(void*);
        CUresult(OFXRS2_GPU_API* cuMemcpyHtoDAsync)(CUdeviceptr, const void*, size_t, CUstream);
        CUresult(OFXRS2_GPU_API* cuMemcpyDtoHAsync)(void*, CUdeviceptr, size_t, CUstream);
        CUresult(OFXRS2_GPU_API* cuStreamCreate)(CUstream*, unsigned int);
        CUresult(OFXRS2_GPU_API* cuStreamDestroy)(CUstream);
        CUresult(OFXRS2_GPU_API* cuStreamSynchronize)(CUstream);
        CUresult(OFXRS2_GPU_API* cuLaunchKernel)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int,
                                                 unsigned int, unsigned int, CUstream, void**, void**);
        CUresult(OFXRS2_GPU_API* cuGetErrorString)(CUresult, const char**);

        // Loaded once per process; false (with error) when there is no usable NVIDIA driver.
        static CudaApi& get()
        {
            static CudaApi api;
            static std::once_flag once;
            std::call_once(once, [] { api.load(); });
            return api;
        }

        std::string describe(CUresult r) const
        {
            const char* s = nullptr;
            if (cuGetErrorString) cuGetErrorString(r, &s);
            return s ? std::string(s) : "CUDA error " + ofToString(r);
        }

    private:
        void load()
        {
            void* lib = openLibrary({
#ifdef _WIN32
                "nvcuda.dll"
#elif defined(__APPLE__)
                "/usr/local/cuda/lib/libcuda.dylib"
#else
                "libcuda.so.1", "libcuda.so"
#endif
            });
            if (!lib)
            {
                error = "no NVIDIA CUDA driver";
                return;
            }
            bool ok = loadSymbol(lib, cuInit, "cuInit") && loadSymbol(lib, cuDeviceGetCount, "cuDeviceGetCount")
                && loadSymbol(lib, cuDeviceGet, "cuDeviceGet") && loadSymbol(lib, cuDeviceGetName, "cuDeviceGetName")
                && loadSymbol(lib, cuDeviceGetAttribute, "cuDeviceGetAttribute")
                && loadSymbol(lib, cuDevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain")
                && loadSymbol(lib, cuDevicePrimaryCtxRelease, "cuDevicePrimaryCtxRelease_v2")
                && loadSymbol(lib, cuCtxSetCurrent, "cuCtxSetCurrent") && loadSymbol(lib, cuModuleLoadDataEx, "cuModuleLoadDataEx")
                && loadSymbol(lib, cuModuleUnload, "cuModuleUnload") && loadSymbol(lib, cuModuleGetFunction, "cuModuleGetFunction")
                && loadSymbol(lib, cuMemAlloc, "cuMemAlloc_v2") && loadSymbol(lib, cuMemFree, "cuMemFree_v2")
                && loadSymbol(lib, cuMemAllocHost, "cuMemAllocHost_v2") && loadSymbol(lib, cuMemFreeHost, "cuMemFreeHost")
                && loadSymbol(lib, cuMemcpyHtoDAsync, "cuMemcpyHtoDAsync_v2") && loadSymbol(lib, cuMemcpyDtoHAsync, "cuMemcpyDtoHAsync_v2")
                && loadSymbol(lib, cuStreamCreate, "cuStreamCreate") && loadSymbol(lib, cuStreamDestroy, "cuStreamDestroy_v2")
                && loadSymbol(lib, cuStreamSynchronize, "cuStreamSynchronize") && loadSymbol(lib, cuLaunchKernel, "cuLaunchKernel");
            loadSymbol(lib, cuGetErrorString, "cuGetErrorString");
            if (!ok)
            {
                error = "the NVIDIA driver is too old (CUDA driver API functions missing)";
                return;
            }
            const CUresult r = cuInit(0);
            if (r != 0)
            {
                error = "cuInit: " + describe(r);
                return;
            }
            loaded = true;
        }
    };

    // ------------------------------------------------------------------------------------
    // OpenCL 1.2, the parts used here (CL/cl.h, declared locally)
    // ------------------------------------------------------------------------------------

    typedef int32_t cl_int;
    typedef uint32_t cl_uint;
    typedef uint64_t cl_ulong;
    typedef cl_ulong cl_bitfield;
    typedef intptr_t cl_context_properties;
    typedef struct _cl_platform_id* cl_platform_id;
    typedef struct _cl_device_id* cl_device_id;
    typedef struct _cl_context* cl_context;
    typedef struct _cl_command_queue* cl_command_queue;
    typedef struct _cl_mem* cl_mem;
    typedef struct _cl_program* cl_program;
    typedef struct _cl_kernel* cl_kernel;
    typedef struct _cl_event* cl_event;
    const cl_bitfield CL_DEVICE_TYPE_GPU = 1 << 2;
    const cl_uint CL_DEVICE_MAX_COMPUTE_UNITS = 0x1002;
    const cl_uint CL_DEVICE_SINGLE_FP_CONFIG = 0x101B;
    const cl_uint CL_DEVICE_AVAILABLE = 0x1027;
    const cl_uint CL_DEVICE_NAME = 0x102B;
    const cl_uint CL_DEVICE_HOST_UNIFIED_MEMORY = 0x1035;
    const cl_bitfield CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT = 1 << 7;
    const cl_context_properties CL_CONTEXT_PLATFORM = 0x1084;
    const cl_uint CL_PROGRAM_BUILD_LOG = 0x1183;
    const cl_bitfield CL_MEM_READ_WRITE = 1 << 0;
    const cl_bitfield CL_MEM_ALLOC_HOST_PTR = 1 << 4;
    const cl_bitfield CL_MAP_READ = 1 << 0;
    const cl_bitfield CL_MAP_WRITE = 1 << 1;
    const cl_uint CL_TRUE = 1;
    const cl_uint CL_FALSE = 0;

    struct ClApi
    {
        bool loaded = false;
        std::string error;
        cl_int(OFXRS2_GPU_API* clGetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*);
        cl_int(OFXRS2_GPU_API* clGetDeviceIDs)(cl_platform_id, cl_bitfield, cl_uint, cl_device_id*, cl_uint*);
        cl_int(OFXRS2_GPU_API* clGetDeviceInfo)(cl_device_id, cl_uint, size_t, void*, size_t*);
        cl_context(OFXRS2_GPU_API* clCreateContext)(const cl_context_properties*, cl_uint, const cl_device_id*, void*, void*, cl_int*);
        cl_int(OFXRS2_GPU_API* clReleaseContext)(cl_context);
        cl_command_queue(OFXRS2_GPU_API* clCreateCommandQueue)(cl_context, cl_device_id, cl_bitfield, cl_int*);
        cl_int(OFXRS2_GPU_API* clReleaseCommandQueue)(cl_command_queue);
        cl_program(OFXRS2_GPU_API* clCreateProgramWithSource)(cl_context, cl_uint, const char**, const size_t*, cl_int*);
        cl_int(OFXRS2_GPU_API* clBuildProgram)(cl_program, cl_uint, const cl_device_id*, const char*, void*, void*);
        cl_int(OFXRS2_GPU_API* clGetProgramBuildInfo)(cl_program, cl_device_id, cl_uint, size_t, void*, size_t*);
        cl_int(OFXRS2_GPU_API* clReleaseProgram)(cl_program);
        cl_kernel(OFXRS2_GPU_API* clCreateKernel)(cl_program, const char*, cl_int*);
        cl_int(OFXRS2_GPU_API* clReleaseKernel)(cl_kernel);
        cl_int(OFXRS2_GPU_API* clSetKernelArg)(cl_kernel, cl_uint, size_t, const void*);
        cl_mem(OFXRS2_GPU_API* clCreateBuffer)(cl_context, cl_bitfield, size_t, void*, cl_int*);
        cl_int(OFXRS2_GPU_API* clReleaseMemObject)(cl_mem);
        cl_int(OFXRS2_GPU_API* clEnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_uint, size_t, size_t, const void*, cl_uint,
                                                     const cl_event*, cl_event*);
        cl_int(OFXRS2_GPU_API* clEnqueueReadBuffer)(cl_command_queue, cl_mem, cl_uint, size_t, size_t, void*, cl_uint,
                                                    const cl_event*, cl_event*);
        cl_int(OFXRS2_GPU_API* clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*,
                                                       const size_t*, cl_uint, const cl_event*, cl_event*);
        cl_int(OFXRS2_GPU_API* clFinish)(cl_command_queue);
        void*(OFXRS2_GPU_API* clEnqueueMapBuffer)(cl_command_queue, cl_mem, cl_uint, cl_bitfield, size_t, size_t, cl_uint, const cl_event*,
                                                   cl_event*, cl_int*);
        cl_int(OFXRS2_GPU_API* clEnqueueUnmapMemObject)(cl_command_queue, cl_mem, void*, cl_uint, const cl_event*, cl_event*);

        static ClApi& get()
        {
            static ClApi api;
            static std::once_flag once;
            std::call_once(once, [] { api.load(); });
            return api;
        }

        struct Gpu
        {
            cl_platform_id platform;
            cl_device_id device;
            ofxLibRealSense2Gpu::DeviceInfo info;
            cl_uint computeUnits;
        };

        std::vector<Gpu> gpus() const
        {
            std::vector<Gpu> out;
            if (!loaded) return out;
            cl_uint platformCount = 0;
            if (clGetPlatformIDs(0, nullptr, &platformCount) != 0 || platformCount == 0) return out;
            std::vector<cl_platform_id> platforms(platformCount);
            clGetPlatformIDs(platformCount, platforms.data(), nullptr);
            for (auto platform : platforms)
            {
                cl_uint deviceCount = 0;
                if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &deviceCount) != 0 || deviceCount == 0) continue;
                std::vector<cl_device_id> devices(deviceCount);
                clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, deviceCount, devices.data(), nullptr);
                for (auto device : devices)
                {
                    cl_uint available = 0, unified = 0, units = 0;
                    clGetDeviceInfo(device, CL_DEVICE_AVAILABLE, sizeof(available), &available, nullptr);
                    if (!available) continue;
                    clGetDeviceInfo(device, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(unified), &unified, nullptr);
                    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(units), &units, nullptr);
                    char name[256] = {};
                    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name) - 1, name, nullptr);
                    Gpu g{ platform, device, {}, units };
                    g.info.name = ofTrim(name);
                    g.info.discrete = !unified;
                    out.push_back(g);
                }
            }
            return out;
        }

    private:
        void load()
        {
            void* lib = openLibrary({
#ifdef _WIN32
                "OpenCL.dll"
#elif defined(__APPLE__)
                "/System/Library/Frameworks/OpenCL.framework/OpenCL"
#else
                "libOpenCL.so.1", "libOpenCL.so"
#endif
            });
            if (!lib)
            {
                error = "no OpenCL runtime";
                return;
            }
            loaded = loadSymbol(lib, clGetPlatformIDs, "clGetPlatformIDs") && loadSymbol(lib, clGetDeviceIDs, "clGetDeviceIDs")
                && loadSymbol(lib, clGetDeviceInfo, "clGetDeviceInfo") && loadSymbol(lib, clCreateContext, "clCreateContext")
                && loadSymbol(lib, clReleaseContext, "clReleaseContext") && loadSymbol(lib, clCreateCommandQueue, "clCreateCommandQueue")
                && loadSymbol(lib, clReleaseCommandQueue, "clReleaseCommandQueue")
                && loadSymbol(lib, clCreateProgramWithSource, "clCreateProgramWithSource")
                && loadSymbol(lib, clBuildProgram, "clBuildProgram") && loadSymbol(lib, clGetProgramBuildInfo, "clGetProgramBuildInfo")
                && loadSymbol(lib, clReleaseProgram, "clReleaseProgram") && loadSymbol(lib, clCreateKernel, "clCreateKernel")
                && loadSymbol(lib, clReleaseKernel, "clReleaseKernel") && loadSymbol(lib, clSetKernelArg, "clSetKernelArg")
                && loadSymbol(lib, clCreateBuffer, "clCreateBuffer") && loadSymbol(lib, clReleaseMemObject, "clReleaseMemObject")
                && loadSymbol(lib, clEnqueueWriteBuffer, "clEnqueueWriteBuffer")
                && loadSymbol(lib, clEnqueueReadBuffer, "clEnqueueReadBuffer")
                && loadSymbol(lib, clEnqueueNDRangeKernel, "clEnqueueNDRangeKernel") && loadSymbol(lib, clFinish, "clFinish")
                && loadSymbol(lib, clEnqueueMapBuffer, "clEnqueueMapBuffer")
                && loadSymbol(lib, clEnqueueUnmapMemObject, "clEnqueueUnmapMemObject");
            if (!loaded) error = "the OpenCL runtime is missing OpenCL 1.2 functions";
        }
    };

    // ------------------------------------------------------------------------------------
    // Backends
    // ------------------------------------------------------------------------------------

    struct Buffer
    {
        uint64_t handle = 0;     // device memory: CUdeviceptr or cl_mem
        size_t bytes = 0;
        void* host = nullptr;    // pinned host mirror, for transfers
        size_t hostBytes = 0;
        uint64_t hostHandle = 0; // OpenCL: the mapped cl_mem behind `host`
    };

    struct Arg
    {
        enum Type { BUFFER, INT, UINT, FLOAT } type;
        const Buffer* buffer = nullptr;
        int32_t i = 0;
        uint32_t u = 0;
        float f = 0.f;
        Arg(const Buffer& b) : type(BUFFER), buffer(&b) {}
        Arg(int v) : type(INT), i(v) {}
        Arg(uint32_t v) : type(UINT), u(v) {}
        Arg(float v) : type(FLOAT), f(v) {}
    };

    // memcpy, split over threads when it's big
    void copyParallel(void* dst, const void* src, size_t bytes, int threads)
    {
        const size_t chunk = 1 << 20;
        const int chunks = static_cast<int>((bytes + chunk - 1) / chunk);
        if (chunks <= 1 || threads <= 1)
        {
            std::memcpy(dst, src, bytes);
            return;
        }
        ofxLibRealSense2Processing::parallelRows(chunks * 16, threads, [&](int, int c0, int c1) {
            const size_t a = std::min(bytes, static_cast<size_t>(c0) * chunk / 16);
            const size_t b = std::min(bytes, static_cast<size_t>(c1) * chunk / 16);
            if (b > a) std::memcpy(static_cast<uint8_t*>(dst) + a, static_cast<const uint8_t*>(src) + a, b - a);
        });
    }
}

namespace ofxLibRealSense2Gpu
{
    // What the processor needs from CUDA / OpenCL. Transfers go through each buffer's pinned
    // host mirror and are asynchronous, like the launches: nothing waits until finish().
    class Backend
    {
    public:
        virtual ~Backend() = default;
        virtual void makeCurrent() {}
        /// Grows the device buffer to at least `bytes` (contents are lost when it grows).
        virtual bool reserve(Buffer& b, size_t bytes) = 0;
        /// Grows the pinned host mirror to at least `bytes`.
        virtual bool reserveHost(Buffer& b, size_t bytes) = 0;
        virtual void release(Buffer& b) = 0;
        /// Copies `data` into the mirror and queues the upload (the buffer must not be
        /// uploaded again before finish()).
        bool upload(Buffer& b, const void* data, size_t bytes)
        {
            if (!reserve(b, bytes) || !reserveHost(b, bytes)) return false;
            if (bytes) std::memcpy(b.host, data, bytes);
            return uploadMirror(b, bytes);
        }
        /// Queues the upload of the first `bytes` of the mirror.
        virtual bool uploadMirror(Buffer& b, size_t bytes) = 0;
        /// Queues a download into the mirror (same offset); read b.host after finish().
        virtual bool download(Buffer& b, size_t bytes, size_t offset = 0) = 0;
        /// 1D (height 0) or 2D (16 x 16 groups) launch over width x height threads.
        virtual bool launch(Kernel k, int width, int height, std::initializer_list<Arg> args) = 0;
        /// `groups` groups of groupSize threads.
        virtual bool launchGroups(Kernel k, int groups, std::initializer_list<Arg> args) = 0;
        virtual bool finish() = 0;

        std::string name;
        std::string error;
    };
}

namespace
{
    using ofxLibRealSense2Gpu::Backend;

    class CudaBackend : public Backend
    {
    public:
        static std::unique_ptr<CudaBackend> create(int index, std::string& error)
        {
            auto& api = CudaApi::get();
            if (!api.loaded)
            {
                error = api.error;
                return nullptr;
            }
            int count = 0;
            api.cuDeviceGetCount(&count);
            if (count == 0)
            {
                error = "no CUDA GPU";
                return nullptr;
            }
            if (index < 0) index = 0;
            if (index >= count)
            {
                error = "no CUDA GPU " + ofToString(index) + " (" + ofToString(count) + " found)";
                return nullptr;
            }
            std::unique_ptr<CudaBackend> b(new CudaBackend());
            if (!b->init(index))
            {
                error = b->error;
                return nullptr;
            }
            return b;
        }

        ~CudaBackend() override
        {
            auto& api = CudaApi::get();
            if (!context) return;
            api.cuCtxSetCurrent(context);
            if (stream) api.cuStreamDestroy(stream);
            if (module) api.cuModuleUnload(module);
            api.cuDevicePrimaryCtxRelease(device);
        }

        void makeCurrent() override { CudaApi::get().cuCtxSetCurrent(context); }

        bool reserve(Buffer& b, size_t bytes) override
        {
            bytes = std::max<size_t>(bytes, 4);
            if (b.handle && b.bytes >= bytes) return true;
            auto& api = CudaApi::get();
            if (b.handle) api.cuMemFree(static_cast<CUdeviceptr>(b.handle));
            b.handle = 0;
            b.bytes = 0;
            CUdeviceptr p = 0;
            if (!check(api.cuMemAlloc(&p, bytes), "cuMemAlloc")) return false;
            b.handle = p;
            b.bytes = bytes;
            return true;
        }

        bool reserveHost(Buffer& b, size_t bytes) override
        {
            bytes = std::max<size_t>(bytes, 4);
            if (b.host && b.hostBytes >= bytes) return true;
            auto& api = CudaApi::get();
            if (b.host) api.cuMemFreeHost(b.host);
            b.host = nullptr;
            b.hostBytes = 0;
            if (!check(api.cuMemAllocHost(&b.host, bytes), "cuMemAllocHost")) return false;
            b.hostBytes = bytes;
            return true;
        }

        void release(Buffer& b) override
        {
            auto& api = CudaApi::get();
            if (b.handle) api.cuMemFree(static_cast<CUdeviceptr>(b.handle));
            if (b.host) api.cuMemFreeHost(b.host);
            b = Buffer();
        }

        bool uploadMirror(Buffer& b, size_t bytes) override
        {
            return bytes == 0 || check(CudaApi::get().cuMemcpyHtoDAsync(b.handle, b.host, bytes, stream), "upload");
        }

        bool download(Buffer& b, size_t bytes, size_t offset) override
        {
            if (bytes == 0) return true;
            if (!reserveHost(b, offset + bytes)) return false;
            return check(CudaApi::get().cuMemcpyDtoHAsync(static_cast<uint8_t*>(b.host) + offset, b.handle + offset, bytes, stream), "download");
        }

        bool launch(Kernel k, int width, int height, std::initializer_list<Arg> args) override
        {
            if (width <= 0 || height < 0) return true;
            if (height == 0) return run(k, (width + 63) / 64, 1, 64, 1, args);
            return run(k, (width + 15) / 16, (height + 15) / 16, 16, 16, args);
        }

        bool launchGroups(Kernel k, int groups, std::initializer_list<Arg> args) override
        {
            return groups <= 0 || run(k, groups, 1, groupSize, 1, args);
        }

        bool finish() override { return check(CudaApi::get().cuStreamSynchronize(stream), "synchronize"); }

    private:
        bool run(Kernel k, unsigned int gx, unsigned int gy, unsigned int bx, unsigned int by, std::initializer_list<Arg> args)
        {
            union Slot { CUdeviceptr p; int32_t i; uint32_t u; float f; };
            Slot slots[24];
            void* params[24];
            int n = 0;
            for (const auto& a : args)
            {
                Slot& s = slots[n];
                s.p = 0;
                switch (a.type)
                {
                case Arg::BUFFER: s.p = a.buffer->handle; break;
                case Arg::INT: s.i = a.i; break;
                case Arg::UINT: s.u = a.u; break;
                case Arg::FLOAT: s.f = a.f; break;
                }
                params[n] = &slots[n];
                ++n;
            }
            return check(CudaApi::get().cuLaunchKernel(functions[k], gx, gy, 1, bx, by, 1, 0, stream, params, nullptr), kernelNames[k]);
        }

        bool init(int index)
        {
            auto& api = CudaApi::get();
            if (!check(api.cuDeviceGet(&device, index), "cuDeviceGet")) return false;
            char deviceName[256] = {};
            api.cuDeviceGetName(deviceName, sizeof(deviceName) - 1, device);
            int major = 0, minor = 0;
            api.cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
            api.cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
            name = std::string(deviceName) + " (sm_" + ofToString(major) + ofToString(minor) + ")";
            if (!check(api.cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain")) return false;
            if (!check(api.cuCtxSetCurrent(context), "cuCtxSetCurrent")) return false;

            // The PTX is compiled for this GPU by the driver (and cached by it).
            char log[8192] = {};
            int options[2] = { CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES };
            void* values[2] = { log, reinterpret_cast<void*>(static_cast<uintptr_t>(sizeof(log) - 1)) };
            const CUresult r = api.cuModuleLoadDataEx(&module, ofxLibRealSense2KernelsPtx, 2, options, values);
            if (r != 0)
            {
                error = "loading the kernels: " + api.describe(r) + (log[0] ? std::string(": ") + log : "")
                    + " (the NVIDIA driver may be older than the CUDA version the kernels were built with)";
                module = nullptr;
                return false;
            }
            for (int k = 0; k < K_COUNT; ++k)
            {
                if (!check(api.cuModuleGetFunction(&functions[k], module, kernelNames[k]), kernelNames[k])) return false;
            }
            return check(api.cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "cuStreamCreate");
        }

        bool check(CUresult r, const char* what)
        {
            if (r == 0) return true;
            error = std::string(what) + ": " + CudaApi::get().describe(r);
            return false;
        }

        CUdevice device = 0;
        CUcontext context = nullptr;
        CUmodule module = nullptr;
        CUstream stream = nullptr;
        CUfunction functions[K_COUNT] = {};
    };

    class ClBackend : public Backend
    {
    public:
        static std::unique_ptr<ClBackend> create(int index, std::string& error)
        {
            auto& api = ClApi::get();
            if (!api.loaded)
            {
                error = api.error;
                return nullptr;
            }
            const auto gpus = api.gpus();
            if (gpus.empty())
            {
                error = "no OpenCL GPU";
                return nullptr;
            }
            if (index < 0)
            {
                // the discrete GPU with the most compute units, else the integrated one
                index = 0;
                for (int i = 1; i < static_cast<int>(gpus.size()); ++i)
                {
                    const auto& a = gpus[i];
                    const auto& b = gpus[index];
                    if (a.info.discrete != b.info.discrete ? a.info.discrete : a.computeUnits > b.computeUnits) index = i;
                }
            }
            if (index >= static_cast<int>(gpus.size()))
            {
                error = "no OpenCL GPU " + ofToString(index) + " (" + ofToString(gpus.size()) + " found)";
                return nullptr;
            }
            std::unique_ptr<ClBackend> b(new ClBackend());
            if (!b->init(gpus[index]))
            {
                error = b->error;
                return nullptr;
            }
            return b;
        }

        ~ClBackend() override
        {
            auto& api = ClApi::get();
            for (auto k : kernels)
            {
                if (k) api.clReleaseKernel(k);
            }
            if (program) api.clReleaseProgram(program);
            if (queue) api.clReleaseCommandQueue(queue);
            if (context) api.clReleaseContext(context);
        }

        bool reserve(Buffer& b, size_t bytes) override
        {
            bytes = std::max<size_t>(bytes, 4);
            if (b.handle && b.bytes >= bytes) return true;
            auto& api = ClApi::get();
            if (b.handle) api.clReleaseMemObject(reinterpret_cast<cl_mem>(b.handle));
            b.handle = 0;
            b.bytes = 0;
            cl_int err = 0;
            cl_mem m = api.clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, nullptr, &err);
            if (!check(err, "clCreateBuffer")) return false;
            b.handle = reinterpret_cast<uint64_t>(m);
            b.bytes = bytes;
            return true;
        }

        // Pinned memory in OpenCL: a host-allocated buffer, mapped for good.
        bool reserveHost(Buffer& b, size_t bytes) override
        {
            bytes = std::max<size_t>(bytes, 4);
            if (b.host && b.hostBytes >= bytes) return true;
            releaseHost(b);
            auto& api = ClApi::get();
            cl_int err = 0;
            cl_mem m = api.clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bytes, nullptr, &err);
            if (!check(err, "clCreateBuffer (pinned)")) return false;
            void* p = api.clEnqueueMapBuffer(queue, m, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, bytes, 0, nullptr, nullptr, &err);
            if (!check(err, "clEnqueueMapBuffer"))
            {
                api.clReleaseMemObject(m);
                return false;
            }
            b.hostHandle = reinterpret_cast<uint64_t>(m);
            b.host = p;
            b.hostBytes = bytes;
            return true;
        }

        void release(Buffer& b) override
        {
            if (b.handle) ClApi::get().clReleaseMemObject(reinterpret_cast<cl_mem>(b.handle));
            releaseHost(b);
            b = Buffer();
        }

        bool uploadMirror(Buffer& b, size_t bytes) override
        {
            return bytes == 0
                || check(ClApi::get().clEnqueueWriteBuffer(queue, reinterpret_cast<cl_mem>(b.handle), CL_FALSE, 0, bytes, b.host, 0, nullptr, nullptr),
                         "upload");
        }

        bool download(Buffer& b, size_t bytes, size_t offset) override
        {
            if (bytes == 0) return true;
            if (!reserveHost(b, offset + bytes)) return false;
            return check(ClApi::get().clEnqueueReadBuffer(queue, reinterpret_cast<cl_mem>(b.handle), CL_FALSE, offset, bytes,
                                                          static_cast<uint8_t*>(b.host) + offset, 0, nullptr, nullptr),
                         "download");
        }

        bool launch(Kernel k, int width, int height, std::initializer_list<Arg> args) override
        {
            if (width <= 0 || height < 0) return true;
            size_t global[2], local[2];
            if (height == 0)
            {
                local[0] = 64;
                global[0] = (width + 63) / 64 * 64;
                return run(k, 1, global, local, args);
            }
            local[0] = local[1] = 16;
            global[0] = (width + 15) / 16 * 16;
            global[1] = (height + 15) / 16 * 16;
            return run(k, 2, global, local, args);
        }

        bool launchGroups(Kernel k, int groups, std::initializer_list<Arg> args) override
        {
            if (groups <= 0) return true;
            size_t global[1] = { static_cast<size_t>(groups) * groupSize }, local[1] = { static_cast<size_t>(groupSize) };
            return run(k, 1, global, local, args);
        }

        bool finish() override { return check(ClApi::get().clFinish(queue), "clFinish"); }

    private:
        void releaseHost(Buffer& b)
        {
            if (!b.hostHandle) return;
            auto& api = ClApi::get();
            cl_mem m = reinterpret_cast<cl_mem>(b.hostHandle);
            api.clEnqueueUnmapMemObject(queue, m, b.host, 0, nullptr, nullptr);
            api.clFinish(queue);
            api.clReleaseMemObject(m);
            b.hostHandle = 0;
            b.host = nullptr;
            b.hostBytes = 0;
        }

        bool run(Kernel k, cl_uint dims, const size_t* global, const size_t* local, std::initializer_list<Arg> args)
        {
            auto& api = ClApi::get();
            cl_uint n = 0;
            for (const auto& a : args)
            {
                cl_int err = 0;
                switch (a.type)
                {
                case Arg::BUFFER:
                {
                    cl_mem m = reinterpret_cast<cl_mem>(a.buffer->handle);
                    err = api.clSetKernelArg(kernels[k], n, sizeof(cl_mem), &m);
                    break;
                }
                case Arg::INT: err = api.clSetKernelArg(kernels[k], n, sizeof(int32_t), &a.i); break;
                case Arg::UINT: err = api.clSetKernelArg(kernels[k], n, sizeof(uint32_t), &a.u); break;
                case Arg::FLOAT: err = api.clSetKernelArg(kernels[k], n, sizeof(float), &a.f); break;
                }
                if (!check(err, kernelNames[k])) return false;
                ++n;
            }
            return check(api.clEnqueueNDRangeKernel(queue, kernels[k], dims, nullptr, global, local, 0, nullptr, nullptr), kernelNames[k]);
        }

        bool init(const ClApi::Gpu& gpu)
        {
            auto& api = ClApi::get();
            name = gpu.info.name;
            cl_int err = 0;
            const cl_context_properties props[] = { CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(gpu.platform), 0 };
            context = api.clCreateContext(props, 1, &gpu.device, nullptr, nullptr, &err);
            if (!check(err, "clCreateContext")) return false;
            queue = api.clCreateCommandQueue(context, gpu.device, 0, &err);
            if (!check(err, "clCreateCommandQueue")) return false;

            const char* src = kernelSource;
            const size_t length = std::strlen(kernelSource);
            program = api.clCreateProgramWithSource(context, 1, &src, &length, &err);
            if (!check(err, "clCreateProgramWithSource")) return false;
            // Correctly rounded division and sqrt, as the CPU, where the GPU has them.
            cl_bitfield fp = 0;
            api.clGetDeviceInfo(gpu.device, CL_DEVICE_SINGLE_FP_CONFIG, sizeof(fp), &fp, nullptr);
            const bool exact = (fp & CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT) != 0;
            if (!exact) ofLogNotice("ofxLibRealSense2") << "OpenCL: " << name << " has no correctly rounded division: results may differ from the CPU in the last bit";
            err = api.clBuildProgram(program, 1, &gpu.device, exact ? "-cl-fp32-correctly-rounded-divide-sqrt" : "", nullptr, nullptr);
            if (err != 0)
            {
                size_t size = 0;
                api.clGetProgramBuildInfo(program, gpu.device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &size);
                std::string log(size, '\0');
                if (size) api.clGetProgramBuildInfo(program, gpu.device, CL_PROGRAM_BUILD_LOG, size, &log[0], nullptr);
                error = "building the kernels (OpenCL error " + ofToString(err) + "): " + log;
                return false;
            }
            for (int k = 0; k < K_COUNT; ++k)
            {
                kernels[k] = api.clCreateKernel(program, kernelNames[k], &err);
                if (!check(err, kernelNames[k])) return false;
            }
            return true;
        }

        bool check(cl_int err, const char* what)
        {
            if (err == 0) return true;
            error = std::string(what) + ": OpenCL error " + ofToString(err);
            return false;
        }

        cl_context context = nullptr;
        cl_command_queue queue = nullptr;
        cl_program program = nullptr;
        cl_kernel kernels[K_COUNT] = {};
    };

    // librealsense's temporal_filter::recalc_persistence_map: for each 8-frame history, the
    // phases (frame indices) at which it counts as credible.
    std::array<uint8_t, 256> persistenceMap(int persistence)
    {
        std::array<uint8_t, 256> map{};
        for (size_t i = 0; i < map.size(); i++)
        {
            const int last7 = !!(i & 1), last6 = !!(i & 2), last5 = !!(i & 4), last4 = !!(i & 8);
            const int last3 = !!(i & 16), last2 = !!(i & 32), last1 = !!(i & 64), lastFrame = !!(i & 128);
            const int all8 = lastFrame + last1 + last2 + last3 + last4 + last5 + last6 + last7;
            switch (persistence)
            {
            case 1: map[i] = all8 >= 8; break;
            case 2: map[i] = lastFrame + last1 + last2 >= 2; break;
            case 3: map[i] = lastFrame + last1 + last2 + last3 >= 2; break;
            case 4: map[i] = all8 >= 2; break;
            case 5: map[i] = lastFrame + last1 >= 1; break;
            case 6: map[i] = lastFrame + last1 + last2 + last3 + last4 >= 1; break;
            case 7: map[i] = all8 >= 1; break;
            case 8: map[i] = 1; break;
            default: break;
            }
        }
        std::array<uint8_t, 256> credible{};
        for (int phase = 0; phase < 8; phase++)
        {
            const unsigned char mask = static_cast<unsigned char>(1 << phase);
            for (int i = 0; i < 256; i++)
            {
                const unsigned char pos = static_cast<unsigned char>((i << (8 - phase)) | (i >> phase));
                if (map[pos]) credible[i] |= mask;
            }
        }
        return credible;
    }

    void packIntrinsics(const rs2_intrinsics& in, float* out)
    {
        out[0] = static_cast<float>(in.width);
        out[1] = static_cast<float>(in.height);
        out[2] = in.ppx;
        out[3] = in.ppy;
        out[4] = in.fx;
        out[5] = in.fy;
        out[6] = static_cast<float>(in.model);
        for (int i = 0; i < 5; ++i) out[7 + i] = in.coeffs[i];
    }

    void packExtrinsics(const rs2_extrinsics& e, float* out)
    {
        for (int i = 0; i < 9; ++i) out[i] = e.rotation[i];
        for (int i = 0; i < 3; ++i) out[9 + i] = e.translation[i];
    }

    bool sameIntrinsics(const rs2_intrinsics& a, const rs2_intrinsics& b) { return std::memcmp(&a, &b, sizeof(a)) == 0; }
}

namespace ofxLibRealSense2Gpu
{
    std::vector<DeviceInfo> listDevices(Api api)
    {
        std::vector<DeviceInfo> out;
        if (api == Api::CUDA)
        {
            auto& cuda = CudaApi::get();
            if (!cuda.loaded) return out;
            int count = 0;
            cuda.cuDeviceGetCount(&count);
            for (int i = 0; i < count; ++i)
            {
                CUdevice d = 0;
                char name[256] = {};
                if (cuda.cuDeviceGet(&d, i) == 0) cuda.cuDeviceGetName(name, sizeof(name) - 1, d);
                out.push_back({ name, true });
            }
        }
        else
        {
            for (const auto& g : ClApi::get().gpus()) out.push_back(g.info);
        }
        return out;
    }

    struct Processor::Impl
    {
        Api api;
        std::unique_ptr<Backend> gpu;

        // depth
        Buffer depth, disparity, transposed, temporalLast, temporalHistory, persistence, mm;
        int depthW = 0, depthH = 0;
        // temporal filter state
        int temporalW = 0, temporalH = 0, temporalPersistence = -1, frameIndex = 0;
        float temporalAlpha = -1.f, temporalDelta = -1.f;

        // align
        Buffer corners, alignCalib, aligned, bigMM;
        rs2_intrinsics cornersFor{};
        int bigW = 0, bigH = 0;

        // colour map
        Buffer depthRays, mapCalib, map;
        rs2_intrinsics depthRaysFor{};

        // colour
        Buffer color;

        // cloud
        Buffer bigRays, opts, pos, valid, colorIndex, texCoords, colors, normals, counts, start, outPos, outTex, outCol, indices, matrix, dummy;
        rs2_intrinsics bigRaysFor{};

        // downloads to hand over in finishFrame()
        struct Pending
        {
            Buffer* buffer;
            void* destination;
            size_t bytes;
        };
        std::vector<Pending> pending;

        ~Impl()
        {
            if (!gpu) return;
            gpu->makeCurrent();
            gpu->finish();
            for (Buffer* b : { &depth, &disparity, &transposed, &temporalLast, &temporalHistory, &persistence, &mm, &corners, &alignCalib,
                               &aligned, &bigMM, &depthRays, &mapCalib, &map, &color, &bigRays, &opts, &pos, &valid, &colorIndex,
                               &texCoords, &colors, &normals, &counts, &start, &outPos, &outTex, &outCol, &indices, &matrix, &dummy })
            {
                gpu->release(*b);
            }
        }

        bool downloadTo(Buffer& b, void* destination, size_t bytes)
        {
            if (!gpu->download(b, bytes)) return false;
            pending.push_back({ &b, destination, bytes });
            return true;
        }

        // Rays (x / z, y / z) for a calibration, uploaded when it changes.
        bool uploadRays(Buffer& buffer, rs2_intrinsics& cachedFor, const rs2_intrinsics& in, const glm::vec2* rays)
        {
            if (buffer.handle && sameIntrinsics(cachedFor, in)) return true;
            if (!gpu->upload(buffer, rays, sizeof(glm::vec2) * in.width * in.height)) return false;
            cachedFor = in;
            return true;
        }

        bool resetTemporal(int count, const DepthFilters& f)
        {
            temporalW = depthW;
            temporalH = depthH;
            temporalAlpha = f.temporalAlpha;
            temporalDelta = f.temporalDelta;
            temporalPersistence = f.temporalPersistence;
            frameIndex = 0;
            const auto lut = persistenceMap(f.temporalPersistence);
            const std::vector<float> zeros(count, 0.f);
            const std::vector<uint8_t> zeroBytes(count, 0);
            return gpu->upload(persistence, lut.data(), lut.size()) && gpu->upload(temporalLast, zeros.data(), zeros.size() * sizeof(float))
                && gpu->upload(temporalHistory, zeroBytes.data(), zeroBytes.size());
        }

        // One pass of the spatial filter along rows (horizontal) or columns (vertical) of the
        // disparity image. Lines that fit in local memory go through it, columns by way of a
        // transpose; longer ones are filtered in global memory.
        bool spatialPass(bool rows, int width, int height, float alpha, float delta)
        {
            const int line = rows ? width : height;
            if (line > maxLine)
            {
                return rows ? gpu->launch(K_SPATIAL_ROWS, height, 0, { disparity, width, height, alpha, delta })
                            : gpu->launch(K_SPATIAL_COLUMNS, width, 0, { disparity, width, height, alpha, delta });
            }
            if (rows) return gpu->launchGroups(K_SPATIAL_ROWS_SHARED, height, { disparity, width, height, alpha, delta });
            if (!gpu->reserve(transposed, sizeof(float) * width * height)) return false;
            return gpu->launch(K_TRANSPOSE, width, height, { disparity, transposed, width, height })
                && gpu->launchGroups(K_SPATIAL_ROWS_SHARED, width, { transposed, height, width, alpha, delta })
                && gpu->launch(K_TRANSPOSE, height, width, { transposed, disparity, height, width });
        }

        bool uploadMatrix(const CloudParams& p)
        {
            float mat[25];
            std::memcpy(mat, p.matrix, sizeof(p.matrix));
            std::memcpy(mat + 16, p.normalMatrix, sizeof(p.normalMatrix));
            return gpu->upload(matrix, mat, sizeof(mat));
        }
    };

    std::unique_ptr<Processor> Processor::create(Api api, int deviceIndex, std::string& error)
    {
        std::unique_ptr<Backend> backend;
        if (api == Api::CUDA)
        {
            backend = CudaBackend::create(deviceIndex, error);
        }
        else
        {
            backend = ClBackend::create(deviceIndex, error);
        }
        if (!backend) return nullptr;
        std::unique_ptr<Processor> p(new Processor());
        p->impl.reset(new Impl());
        p->impl->api = api;
        p->impl->gpu = std::move(backend);
        if (!p->impl->gpu->reserve(p->impl->dummy, 4))
        {
            error = p->impl->gpu->error;
            return nullptr;
        }
        return p;
    }

    Processor::~Processor() = default;

    Api Processor::getApi() const { return impl->api; }
    const std::string& Processor::getDeviceName() const { return impl->gpu->name; }
    const std::string& Processor::getError() const { return impl->gpu->error; }

    bool Processor::processDepth(const uint16_t* depth, int width, int height, int strideBytes, const DepthFilters& f, float toMM,
                                 bool flip, float* mm)
    {
        auto& m = *impl;
        auto& gpu = *m.gpu;
        gpu.makeCurrent();
        const int count = width * height;
        const size_t depthBytes = sizeof(uint16_t) * count;
        m.depthW = width;
        m.depthH = height;
        m.pending.clear();

        // Upload, packed (librealsense frames normally are already), through the pinned mirror.
        if (!gpu.reserve(m.depth, depthBytes) || !gpu.reserveHost(m.depth, depthBytes)) return false;
        for (int y = 0; y < height; ++y)
        {
            std::memcpy(static_cast<uint8_t*>(m.depth.host) + sizeof(uint16_t) * width * y,
                        reinterpret_cast<const uint8_t*>(depth) + static_cast<size_t>(strideBytes) * y, sizeof(uint16_t) * width);
        }
        if (!gpu.uploadMirror(m.depth, depthBytes)) return false;

        if ((f.spatial || f.temporal) && f.disparityFactor > 0.f)
        {
            if (!gpu.reserve(m.disparity, sizeof(float) * count)) return false;
            if (!gpu.launch(K_DEPTH_TO_DISPARITY, count, 0, { m.depth, m.disparity, count, f.disparityFactor })) return false;
            if (f.spatial)
            {
                for (int i = 0; i < f.spatialIterations; ++i)
                {
                    if (!m.spatialPass(true, width, height, f.spatialAlpha, f.spatialDelta)) return false;
                    if (!m.spatialPass(false, width, height, f.spatialAlpha, f.spatialDelta)) return false;
                }
            }
            if (f.temporal)
            {
                // librealsense resets the history when the options or the stream change
                if (m.temporalW != width || m.temporalH != height || m.temporalAlpha != f.temporalAlpha || m.temporalDelta != f.temporalDelta
                    || m.temporalPersistence != f.temporalPersistence)
                {
                    if (!m.resetTemporal(count, f)) return false;
                }
                const int mask = 1 << m.frameIndex;
                if (!gpu.launch(K_TEMPORAL, count, 0,
                                { m.disparity, m.temporalLast, m.temporalHistory, m.persistence, count, mask, f.temporalAlpha,
                                  static_cast<float>(static_cast<uint8_t>(f.temporalDelta)) }))
                {
                    return false;
                }
                m.frameIndex = (m.frameIndex + 1) % 8;
            }
            if (!gpu.launch(K_DISPARITY_TO_DEPTH, count, 0, { m.disparity, m.depth, count, f.disparityFactor })) return false;
        }

        if (f.holeFillMode >= 0)
        {
            // sequential: on the CPU, between a download and an upload
            if (!gpu.download(m.depth, depthBytes) || !gpu.finish()) return false;
            ofxLibRealSense2Processing::holeFill(static_cast<uint16_t*>(m.depth.host), width, height, f.holeFillMode);
            if (!gpu.uploadMirror(m.depth, depthBytes)) return false;
        }

        if (!gpu.reserve(m.mm, sizeof(float) * count)) return false;
        if (!gpu.launch(K_DEPTH_TO_MM, width, height, { m.depth, m.mm, width, height, toMM, flip ? 1 : 0 })) return false;
        return m.downloadTo(m.mm, mm, sizeof(float) * count);
    }

    bool Processor::alignDepth(const rs2_intrinsics& depthIn, const rs2_extrinsics& depthToOther, const rs2_intrinsics& otherIn,
                               float depthUnits, float toMM, bool flip, float* mm)
    {
        auto& m = *impl;
        auto& gpu = *m.gpu;
        gpu.makeCurrent();
        const int dw = depthIn.width, dh = depthIn.height;
        if (dw != m.depthW || dh != m.depthH) return true;
        const int ow = otherIn.width, oh = otherIn.height;
        const int count = ow * oh;

        // Rays through the depth pixels' corners, (x - 0.5, y - 0.5) for x in 0..w, y in 0..h.
        if (!m.corners.handle || !sameIntrinsics(m.cornersFor, depthIn))
        {
            std::vector<float> corners(static_cast<size_t>(dw + 1) * (dh + 1) * 2);
            for (int y = 0; y <= dh; ++y)
            {
                for (int x = 0; x <= dw; ++x)
                {
                    const float pixel[2] = { x - 0.5f, y - 0.5f };
                    float p[3];
                    rs2_deproject_pixel_to_point(p, &depthIn, pixel, 1.f);
                    const size_t i = (static_cast<size_t>(y) * (dw + 1) + x) * 2;
                    corners[i] = p[0];
                    corners[i + 1] = p[1];
                }
            }
            if (!gpu.upload(m.corners, corners.data(), sizeof(float) * corners.size())) return false;
            m.cornersFor = depthIn;
        }
        float calib[24];
        packIntrinsics(otherIn, calib);
        packExtrinsics(depthToOther, calib + 12);
        if (!gpu.upload(m.alignCalib, calib, sizeof(calib))) return false;

        if (!gpu.reserve(m.aligned, sizeof(uint32_t) * count) || !gpu.reserve(m.bigMM, sizeof(float) * count)) return false;
        if (!gpu.launch(K_FILL32, count, 0, { m.aligned, count, 0xffffffffu })) return false;
        if (!gpu.launch(K_ALIGN_SPLAT, dw, dh, { m.depth, m.corners, m.alignCalib, m.aligned, dw, dh, depthUnits })) return false;
        if (!gpu.launch(K_ALIGNED_TO_MM, ow, oh, { m.aligned, m.bigMM, ow, oh, toMM, flip ? 1 : 0 })) return false;
        m.bigW = ow;
        m.bigH = oh;
        return m.downloadTo(m.bigMM, mm, sizeof(float) * count);
    }

    bool Processor::colorMap(const rs2_intrinsics& depthIn, const glm::vec2* rays, const rs2_intrinsics& colorIn,
                             const rs2_extrinsics& depthToColor, bool flip, int* map)
    {
        auto& m = *impl;
        auto& gpu = *m.gpu;
        gpu.makeCurrent();
        const int w = depthIn.width, h = depthIn.height;
        if (w != m.depthW || h != m.depthH) return true;
        if (!m.uploadRays(m.depthRays, m.depthRaysFor, depthIn, rays)) return false;
        float calib[24];
        packIntrinsics(colorIn, calib);
        packExtrinsics(depthToColor, calib + 12);
        if (!gpu.upload(m.mapCalib, calib, sizeof(calib))) return false;
        if (!gpu.reserve(m.map, sizeof(int) * w * h)) return false;
        if (!gpu.launch(K_COLOR_MAP, w, h, { m.mm, m.depthRays, m.mapCalib, m.map, w, h, flip ? 1 : 0 })) return false;
        return m.downloadTo(m.map, map, sizeof(int) * w * h);
    }

    bool Processor::finishFrame(int threads)
    {
        auto& m = *impl;
        m.gpu->makeCurrent();
        if (!m.gpu->finish()) return false;
        for (const auto& p : m.pending) copyParallel(p.destination, p.buffer->host, p.bytes, threads);
        m.pending.clear();
        return true;
    }

    bool Processor::uploadColor(const uint8_t* pixels, int width, int height, int channels)
    {
        impl->gpu->makeCurrent();
        return impl->gpu->upload(impl->color, pixels, static_cast<size_t>(width) * height * channels);
    }

    bool Processor::buildCloud(const CloudParams& p, const rs2_intrinsics& gridIn, const glm::vec2* rays, CloudOutput& out, int threads)
    {
        auto& m = *impl;
        auto& gpu = *m.gpu;
        gpu.makeCurrent();

        const int w = p.width, h = p.height;
        Buffer& depthImage = p.big ? m.bigMM : m.mm;
        if (p.big ? (w != m.bigW || h != m.bigH) : (w != m.depthW || h != m.depthH))
        {
            out.positions.clear();
            out.normals.clear();
            out.texCoords.clear();
            out.colors.clear();
            out.indices.clear();
            return true;
        }
        Buffer& rayBuffer = p.big ? m.bigRays : m.depthRays;
        rs2_intrinsics& raysFor = p.big ? m.bigRaysFor : m.depthRaysFor;
        if (!m.uploadRays(rayBuffer, raysFor, gridIn, rays)) return false;

        const int step = std::max(1, p.step);
        const int gw = (w - 1) / step + 1;
        const int gh = (h - 1) / step + 1;
        const size_t count = static_cast<size_t>(gw) * gh;
        const bool useTexCoords = p.texCoords && p.hasColor;
        const bool useColors = !p.texCoords && p.hasColor;
        const bool meshTex = p.faces && useTexCoords;
        const bool meshCol = p.faces && useColors;

        const int opts[12] = { p.flip ? 1 : 0, p.denoise ? 1 : 0, p.denoiseSpacing, step, gw, gh, p.hasColor ? 1 : 0, p.big ? 1 : 0,
                               p.colorWidth, meshTex ? 1 : 0, meshCol ? 1 : 0, p.channels };
        if (!gpu.upload(m.opts, opts, sizeof(opts))) return false;
        if (!gpu.reserve(m.pos, sizeof(float) * 3 * count) || !gpu.reserve(m.valid, count) || !gpu.reserve(m.colorIndex, sizeof(int) * count))
        {
            return false;
        }
        if (meshTex && !gpu.reserve(m.texCoords, sizeof(float) * 2 * count)) return false;
        if (meshCol && !gpu.reserve(m.colors, sizeof(float) * 4 * count)) return false;
        Buffer& colorBuffer = useColors ? m.color : m.dummy;
        Buffer& mapBuffer = (p.hasColor && !p.big) ? m.map : m.dummy;
        if (!gpu.launch(K_CLOUD_SAMPLES, gw, gh,
                        { depthImage, rayBuffer, mapBuffer, colorBuffer, m.opts, w, h, p.minZ, p.maxZ, p.alpha, m.pos, m.valid, m.colorIndex,
                          meshTex ? m.texCoords : m.dummy, meshCol ? m.colors : m.dummy }))
        {
            return false;
        }

        // Rows' counts -> prefix sum -> total (one wait), then the compacted output (another).
        const int rows = p.faces ? std::max(0, gh - 1) : gh;
        if (!gpu.reserve(m.counts, sizeof(uint32_t) * std::max(1, rows)) || !gpu.reserve(m.start, sizeof(uint32_t) * (rows + 1))) return false;
        if (p.faces)
        {
            if (!gpu.reserve(m.normals, sizeof(float) * 3 * count)) return false;
            if (!gpu.launch(K_NORMALS, gw, gh, { m.pos, m.valid, m.normals, gw, gh, p.maxEdge2, p.maxJump })) return false;
            if (!gpu.launchGroups(K_TRIANGLES_ROW_COUNT, rows, { m.pos, m.valid, m.counts, gw, gh, p.maxEdge2, p.maxJump })) return false;
        }
        else
        {
            if (!gpu.launchGroups(K_POINTS_ROW_COUNT, rows, { m.valid, m.counts, gw, gh })) return false;
        }
        if (!gpu.launch(K_SCAN, 1, 0, { m.counts, m.start, rows })) return false;
        if (!gpu.download(m.start, sizeof(uint32_t), sizeof(uint32_t) * rows) || !gpu.finish()) return false;
        uint32_t total = 0;
        std::memcpy(&total, static_cast<uint8_t*>(m.start.host) + sizeof(uint32_t) * rows, sizeof(total));

        if (!p.faces)
        {
            // Points: compacted, row-major.
            if (!gpu.reserve(m.outPos, sizeof(float) * 3 * total)) return false;
            if (useTexCoords && !gpu.reserve(m.outTex, sizeof(float) * 2 * total)) return false;
            if (useColors && !gpu.reserve(m.outCol, sizeof(float) * 4 * total)) return false;
            if (!gpu.launchGroups(K_POINTS_COMPACT, gh,
                                  { m.pos, m.valid, m.colorIndex, m.start, colorBuffer, gw, gh, useTexCoords ? 1 : 0, useColors ? 1 : 0,
                                    p.colorWidth, p.channels, p.alpha, m.outPos, useTexCoords ? m.outTex : m.dummy,
                                    useColors ? m.outCol : m.dummy }))
            {
                return false;
            }
            if (p.transform)
            {
                if (!m.uploadMatrix(p)) return false;
                if (!gpu.launch(K_TRANSFORM, static_cast<int>(total), 0, { m.outPos, m.dummy, m.matrix, static_cast<int>(total), 0 })) return false;
            }
            if (!gpu.download(m.outPos, sizeof(float) * 3 * total)) return false;
            if (useTexCoords && !gpu.download(m.outTex, sizeof(float) * 2 * total)) return false;
            if (useColors && !gpu.download(m.outCol, sizeof(float) * 4 * total)) return false;
            if (!gpu.finish()) return false;

            out.normals.clear();
            out.indices.clear();
            out.positions.resize(total);
            copyParallel(out.positions.data(), m.outPos.host, sizeof(float) * 3 * total, threads);
            out.texCoords.resize(useTexCoords ? total : 0);
            if (useTexCoords) copyParallel(out.texCoords.data(), m.outTex.host, sizeof(float) * 2 * total, threads);
            out.colors.resize(useColors ? total : 0);
            if (useColors) copyParallel(out.colors.data(), m.outCol.host, sizeof(float) * 4 * total, threads);
            return true;
        }

        // Mesh: every grid sample a vertex, triangles row-major.
        if (!gpu.reserve(m.indices, sizeof(uint32_t) * std::max<uint32_t>(1, total))) return false;
        if (!gpu.launchGroups(K_TRIANGLES_EMIT, rows, { m.pos, m.valid, m.start, m.indices, gw, gh, p.maxEdge2, p.maxJump })) return false;
        if (p.transform)
        {
            if (!m.uploadMatrix(p)) return false;
            if (!gpu.launch(K_TRANSFORM, static_cast<int>(count), 0, { m.pos, m.normals, m.matrix, static_cast<int>(count), 1 })) return false;
        }
        if (!gpu.download(m.pos, sizeof(float) * 3 * count) || !gpu.download(m.normals, sizeof(float) * 3 * count)) return false;
        if (meshTex && !gpu.download(m.texCoords, sizeof(float) * 2 * count)) return false;
        if (meshCol && !gpu.download(m.colors, sizeof(float) * 4 * count)) return false;
        if (!gpu.download(m.indices, sizeof(uint32_t) * total)) return false;
        if (!gpu.finish()) return false;

        static_assert(sizeof(ofIndexType) == sizeof(uint32_t), "ofIndexType must be 32-bit for the GPU mesh");
        out.positions.resize(count);
        copyParallel(out.positions.data(), m.pos.host, sizeof(float) * 3 * count, threads);
        out.normals.resize(count);
        copyParallel(out.normals.data(), m.normals.host, sizeof(float) * 3 * count, threads);
        out.texCoords.resize(meshTex ? count : 0);
        if (meshTex) copyParallel(out.texCoords.data(), m.texCoords.host, sizeof(float) * 2 * count, threads);
        out.colors.resize(meshCol ? count : 0);
        if (meshCol) copyParallel(out.colors.data(), m.colors.host, sizeof(float) * 4 * count, threads);
        out.indices.resize(total);
        copyParallel(out.indices.data(), m.indices.host, sizeof(uint32_t) * total, threads);
        return true;
    }
}
