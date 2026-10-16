//
//  ofxLibRealSense2Gpu.h
//
//  GPU (CUDA or OpenCL) versions of the capture thread's processing: librealsense's disparity
//  transform, spatial and temporal filters, the depth images, depth -> colour align, the
//  depth -> colour map and the point cloud / mesh. Results match the CPU path (same float
//  operations, no FMA contraction).
//
//  Both APIs are loaded at run time - CUDA's driver API from the NVIDIA driver (nvcuda.dll /
//  libcuda.so.1), OpenCL from the system's ICD loader (OpenCL.dll / libOpenCL.so.1 / the macOS
//  framework) - so nothing is linked, nothing extra ships, and an app runs on machines with
//  neither: Processor::create() fails and the camera uses the CPU.
//
//  CUDA runs the kernels as PTX built ahead of time (ofxLibRealSense2KernelsPtx.h, see
//  scripts/build_gpu_kernels); OpenCL compiles the same kernel source at run time.
//
//  Internal to the addon; not part of the API.
//

#pragma once

#include "ofMain.h"

#include <librealsense2/rs.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ofxLibRealSense2Gpu
{
    enum class Api { CUDA, OPENCL };

    struct DeviceInfo
    {
        std::string name;
        bool discrete = true;
    };

    /// The GPUs an API can use, in its own order (the index for Processor::create()). Empty
    /// when the API isn't installed (no driver / ICD) or has no GPU.
    std::vector<DeviceInfo> listDevices(Api api);

    struct DepthFilters
    {
        bool spatial = false;
        float spatialAlpha = 0.5f;
        float spatialDelta = 20.f;
        int spatialIterations = 2;
        bool temporal = false;
        float temporalAlpha = 0.4f;
        float temporalDelta = 20.f;
        int temporalPersistence = 3;
        /// librealsense's disparity <-> depth factor for this depth stream
        float disparityFactor = 0.f;
        /// librealsense's hole filling mode (RS2_OPTION_HOLES_FILL), -1 = off. Runs on the CPU
        /// (it's sequential), between a download and an upload.
        int holeFillMode = -1;
    };

    struct CloudParams
    {
        bool big = false;           // grid = the aligned depth (colour camera) instead of depth
        int width = 0, height = 0;  // grid image size
        bool flip = false;
        bool denoise = false;
        int denoiseSpacing = 1;
        int step = 1;
        float minZ = 0.f, maxZ = 0.f;
        bool hasColor = false;
        int colorWidth = 0;
        int channels = 4;
        bool texCoords = false;     // texture coordinates (else colours, when hasColor)
        float alpha = 1.f;
        bool faces = false;
        float maxEdge2 = 0.f;
        float maxJump = 0.f;
        bool transform = false;
        float matrix[16] = {};       // column-major
        float normalMatrix[9] = {};  // column-major
    };

    struct CloudOutput
    {
        std::vector<glm::vec3>& positions;
        std::vector<glm::vec3>& normals;
        std::vector<glm::vec2>& texCoords;
        std::vector<ofFloatColor>& colors;
        std::vector<ofIndexType>& indices;
    };

    class Backend;

    /// One camera's processing on one GPU, with its buffers and the temporal filter's state.
    /// Every call returns false on a GPU error (getError() says what): the caller then drops
    /// the processor and uses the CPU.
    ///
    /// processDepth(), alignDepth() and colorMap() only queue their work: their results are
    /// in the caller's buffers once finishFrame() returns. The GPU then waits for nothing in
    /// between, and the copies go through pinned memory.
    class Processor
    {
    public:
        /// deviceIndex: into listDevices(api), -1 for the default (CUDA: device 0; OpenCL: the
        /// discrete GPU with the most compute units).
        static std::unique_ptr<Processor> create(Api api, int deviceIndex, std::string& error);
        ~Processor();

        Api getApi() const;
        const std::string& getDeviceName() const;
        const std::string& getError() const;

        /// Uploads 16-bit depth, runs the filters on it and writes it in mm (flipped when
        /// asked) to `mm` (width * height floats). The filtered depth stays on the GPU for the
        /// calls below, until the next processDepth().
        bool processDepth(const uint16_t* depth, int width, int height, int strideBytes, const DepthFilters& filters,
                          float toMM, bool flip, float* mm);

        /// The filtered depth re-projected into another camera (librealsense's align, nearest
        /// wins), in mm, flipped when asked: `mm` is otherIn.width * otherIn.height floats.
        /// Stays on the GPU as the colour-resolution cloud's grid.
        bool alignDepth(const rs2_intrinsics& depthIn, const rs2_extrinsics& depthToOther, const rs2_intrinsics& otherIn,
                        float depthUnits, float toMM, bool flip, float* mm);

        /// For each depth pixel, the index of its colour pixel or -1 (depth size). `rays`: the
        /// depth camera's rays (x / z, y / z) per pixel, for depthIn.
        bool colorMap(const rs2_intrinsics& depthIn, const glm::vec2* rays, const rs2_intrinsics& colorIn,
                      const rs2_extrinsics& depthToColor, bool flip, int* map);

        /// Waits for the queued work and copies its results out (`threads` share the copies).
        bool finishFrame(int threads);

        /// The colour image, for per-vertex colours.
        bool uploadColor(const uint8_t* pixels, int width, int height, int channels);

        /// Point cloud / mesh from this frame's depth (or aligned depth, p.big), as
        /// ofxLibRealSense2::buildPointCloud. `rays` for gridIn. Complete when it returns.
        bool buildCloud(const CloudParams& p, const rs2_intrinsics& gridIn, const glm::vec2* rays, CloudOutput& out, int threads);

    private:
        Processor() = default;
        struct Impl;
        std::unique_ptr<Impl> impl;
    };
}
