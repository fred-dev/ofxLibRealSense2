//
//  ofxLibRealSense2Kernels.h
//
//  GPU kernels for ofxLibRealSense2's CUDA and OpenCL processing: one source, compiled as
//  OpenCL C at run time (the text of this file, see ofxLibRealSense2Gpu.cpp) and as CUDA into
//  PTX ahead of time (scripts/build_gpu_kernels: strips the R"OFXRS2( / )OFXRS2" lines and
//  writes ofxLibRealSense2KernelsPtx.h). Edit here, then rerun that script.
//
//  Each kernel is a transcription of the CPU code it replaces (librealsense's filters, the
//  addon's align, colour map and cloud builder), with the same float operations in the same
//  order and no FMA contraction (-fmad=false / FP_CONTRACT OFF), so results match the CPU.
//
//  Internal to the addon; not part of the API.
//
//  (Split into several raw string literals: MSVC limits each to 16 KB.)
//
R"OFXRS2(
#ifdef __OPENCL_VERSION__
#pragma OPENCL FP_CONTRACT OFF
#define KERNEL __kernel void
#define DEVICE inline
#define GLOBAL __global
#define GX ((int)get_global_id(0))
#define GY ((int)get_global_id(1))
#define AS_INT(x) as_int(x)
#define ATOMIC_MIN(p, v) atomic_min((p), (v))
#define LOCAL __local
#define LOCAL_ARRAY(type, name, size) __local type name[size]
#define BARRIER() barrier(CLK_LOCAL_MEM_FENCE)
#define LID ((int)get_local_id(0))
#define LX ((int)get_local_id(0))
#define LY ((int)get_local_id(1))
#define GROUP ((int)get_group_id(0))
typedef uchar u8;
typedef ushort u16;
typedef uint u32;
#else
#define KERNEL extern "C" __global__ void
#define DEVICE __device__ __forceinline__
#define GLOBAL
#define GX ((int)(blockIdx.x * blockDim.x + threadIdx.x))
#define GY ((int)(blockIdx.y * blockDim.y + threadIdx.y))
#define AS_INT(x) __float_as_int(x)
#define ATOMIC_MIN(p, v) atomicMin((p), (v))
#define LOCAL
#define LOCAL_ARRAY(type, name, size) __shared__ type name[size]
#define BARRIER() __syncthreads()
#define LID ((int)threadIdx.x)
#define LX ((int)threadIdx.x)
#define LY ((int)threadIdx.y)
#define GROUP ((int)blockIdx.x)
typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
#endif

// Work-group kernels (one group per line / row) run GROUP_SIZE threads; the host uses the same.
#define GROUP_SIZE 128
// Longest line the shared-memory spatial filter holds (floats); longer ones use the global one.
#define MAX_LINE 2048

// Calibration blocks, as floats: intrinsics [width, height, ppx, ppy, fx, fy, model, c0..c4],
// extrinsics [rotation 0..8 (column-major, as rs2_extrinsics), translation 0..2].
#define INTRINSICS_SIZE 12
#define EXTRINSICS_SIZE 12

// rs2_distortion values
#define MODEL_MODIFIED_BROWN_CONRADY 1
#define MODEL_INVERSE_BROWN_CONRADY 2
#define MODEL_FTHETA 3
#define MODEL_BROWN_CONRADY 4
#define MODEL_KANNALA_BRANDT4 5

// librealsense's validity test for disparity: the float's bits as an int are > 0.
DEVICE int validDisparity(float v) { return AS_INT(v) > 0; }

// std::isnormal
DEVICE int isNormal(float v)
{
    const int e = AS_INT(v) & 0x7f800000;
    return e != 0 && e != 0x7f800000;
}

// rs2_transform_point_to_point
DEVICE void transformPoint(GLOBAL const float* e, float x, float y, float z, float* o)
{
    o[0] = e[0] * x + e[3] * y + e[6] * z + e[9];
    o[1] = e[1] * x + e[4] * y + e[7] * z + e[10];
    o[2] = e[2] * x + e[5] * y + e[8] * z + e[11];
}

// rs2_project_point_to_pixel (librealsense 2.59's rs.cpp). F-theta and Kannala-Brandt (T265
// fisheyes, not D400 / D500) use float maths here, librealsense double for parts of them.
DEVICE void projectPoint(GLOBAL const float* in, const float* point, float* pixel)
{
    float x = point[0] / point[2], y = point[1] / point[2];
    const int model = (int)in[6];
    GLOBAL const float* c = in + 7;

    if (model == MODEL_MODIFIED_BROWN_CONRADY || model == MODEL_INVERSE_BROWN_CONRADY)
    {
        float r2 = x * x + y * y;
        float f = 1 + c[0] * r2 + c[1] * r2 * r2 + c[4] * r2 * r2 * r2;
        x *= f;
        y *= f;
        float dx = x + 2 * c[2] * x * y + c[3] * (r2 + 2 * x * x);
        float dy = y + 2 * c[3] * x * y + c[2] * (r2 + 2 * y * y);
        x = dx;
        y = dy;
    }
    if (model == MODEL_BROWN_CONRADY)
    {
        float r2 = x * x + y * y;
        float f = 1 + c[0] * r2 + c[1] * r2 * r2 + c[4] * r2 * r2 * r2;
        float xf = x * f;
        float yf = y * f;
        float dx = xf + 2 * c[2] * x * y + c[3] * (r2 + 2 * x * x);
        float dy = yf + 2 * c[3] * x * y + c[2] * (r2 + 2 * y * y);
        x = dx;
        y = dy;
    }
    if (model == MODEL_FTHETA)
    {
        float r = sqrt(x * x + y * y);
        if (r < 1.192092896e-07f) r = 1.192092896e-07f;
        float rd = 1.0f / c[0] * atan(2 * r * tan(c[0] / 2.0f));
        x *= rd / r;
        y *= rd / r;
    }
    if (model == MODEL_KANNALA_BRANDT4)
    {
        float r = sqrt(x * x + y * y);
        if (r < 1.192092896e-07f) r = 1.192092896e-07f;
        float theta = atan(r);
        float theta2 = theta * theta;
        float series = 1 + theta2 * (c[0] + theta2 * (c[1] + theta2 * (c[2] + theta2 * c[3])));
        float rd = theta * series;
        x *= rd / r;
        y *= rd / r;
    }
    pixel[0] = x * in[4] + in[2];
    pixel[1] = y * in[5] + in[3];
}

// ---------------------------------------------------------------------------------------
// Depth filters (librealsense's disparity transform, spatial and temporal filters)
// ---------------------------------------------------------------------------------------

// disparity_transform, depth -> disparity: factor / depth (0 stays 0)
KERNEL k_depthToDisparity(GLOBAL const u16* depth, GLOBAL float* disparity, int count, float factor)
{
    const int i = GX;
    if (i >= count) return;
    const float input = (float)depth[i];
    disparity[i] = isNormal(input) ? (factor / input) + 0.f : 0.f;
}

// disparity_transform, disparity -> depth, rounded
KERNEL k_disparityToDepth(GLOBAL const float* disparity, GLOBAL u16* depth, int count, float factor)
{
    const int i = GX;
    if (i >= count) return;
    const float input = disparity[i];
    depth[i] = isNormal(input) ? (u16)(int)((factor / input) + 0.5f) : (u16)0;
}

// One pass of the spatial filter's recursive edge-preserving filter along a line of n
// samples `step` apart (spatial_filter::recursive_filter_horizontal_fp / _vertical_fp: the
// CPU's goto state machine as a flag).
// Defined for global and for local (shared) memory: OpenCL needs one per address space.
#define FILTER_LINE(NAME, SPACE) \
DEVICE void NAME(SPACE float* line, int n, int step, float alpha, float deltaZ) \
{ \
    int idx = 0; \
    float state = line[idx]; \
    float previousInnovation = state; \
    idx += step; \
    float innovation = line[idx]; \
    int u = n - 1; \
    int currentlyValid = validDisparity(previousInnovation); \
    for (;;) \
    { \
        if (currentlyValid) \
        { \
            if (validDisparity(innovation)) \
            { \
                const float delta = previousInnovation - innovation; \
                if (delta < deltaZ && delta > -deltaZ) \
                { \
                    const float filtered = innovation * alpha + state * (1.0f - alpha); \
                    line[idx] = state = filtered; \
                } \
                else \
                { \
                    state = innovation; \
                } \
                if (--u <= 0) return; \
                previousInnovation = innovation; \
                idx += step; \
                innovation = line[idx]; \
            } \
            else \
            { \
                if (--u <= 0) return; \
                previousInnovation = innovation; \
                idx += step; \
                innovation = line[idx]; \
                currentlyValid = 0; \
            } \
        } \
        else \
        { \
            if (--u <= 0) return; \
            if (validDisparity(innovation)) \
            { \
                previousInnovation = state = innovation; \
                idx += step; \
                innovation = line[idx]; \
                currentlyValid = 1; \
            } \
            else \
            { \
                idx += step; \
                innovation = line[idx]; \
            } \
        } \
    } \
}
FILTER_LINE(filterLine, GLOBAL)
FILTER_LINE(filterLineLocal, LOCAL)

// Spatial filter, horizontal pass: one thread per row, forward then backward.
KERNEL k_spatialRows(GLOBAL float* disparity, int width, int height, float alpha, float deltaZ)
{
    const int y = GX;
    if (y >= height || width < 2) return;
    GLOBAL float* row = disparity + (size_t)y * width;
    filterLine(row, width, 1, alpha, deltaZ);
    filterLine(row + (width - 1), width, -1, alpha, deltaZ);
}

// Spatial filter, vertical pass: one thread per column, down then up.
KERNEL k_spatialColumns(GLOBAL float* disparity, int width, int height, float alpha, float deltaZ)
{
    const int x = GX;
    if (x >= width || height < 2) return;
    GLOBAL float* column = disparity + x;
    filterLine(column, height, width, alpha, deltaZ);
    filterLine(column + (size_t)(height - 1) * width, height, -width, alpha, deltaZ);
}

// temporal_filter::temp_jw_smooth<float>: last / history are the filter's state, lut its
// persistence map, mask 1 << the frame's index in the 8-frame history.
KERNEL k_temporal(GLOBAL float* frame, GLOBAL float* last, GLOBAL u8* history, GLOBAL const u8* lut,
                  int count, int mask, float alpha, float deltaZ)
{
    const int i = GX;
    if (i >= count) return;
    const float oneMinusAlpha = 1.f - alpha;
    const float cur = frame[i];
    const float prev = last[i];
    if (cur != 0.f)
    {
        if (prev == 0.f)
        {
            last[i] = cur;
            history[i] = (u8)mask;
        }
        else
        {
            const float diff = fabs(cur - prev);
            if (diff < deltaZ)
            {
                history[i] = (u8)(history[i] | mask);
                const float filtered = alpha * cur + oneMinusAlpha * prev;
                frame[i] = filtered;
                last[i] = filtered;
            }
            else
            {
                last[i] = cur;
                history[i] = (u8)mask;
            }
        }
    }
    else
    {
        if (prev != 0.f)
        {
            if (lut[history[i]] & mask) frame[i] = prev;
        }
        history[i] = (u8)(history[i] & ~mask);
    }
}
)OFXRS2"
R"OFXRS2(
// ---------------------------------------------------------------------------------------
// Images
// ---------------------------------------------------------------------------------------

// 16-bit depth -> float mm, flipped horizontally when asked.
KERNEL k_depthToMM(GLOBAL const u16* depth, GLOBAL float* mm, int width, int height, float toMM, int flip)
{
    const int x = GX, y = GY;
    if (x >= width || y >= height) return;
    mm[(size_t)y * width + x] = depth[(size_t)y * width + (flip ? width - 1 - x : x)] * toMM;
}

// Depth -> colour map: each depth pixel (mm, as stored, maybe flipped) through the depth
// camera's rays, the extrinsics and the colour intrinsics; -1 where it misses the colour image.
// calib: colour intrinsics, then depth -> colour extrinsics.
KERNEL k_colorMap(GLOBAL const float* mm, GLOBAL const float* rays, GLOBAL const float* calib, GLOBAL int* map,
                  int width, int height, int flip)
{
    const int x = GX, y = GY;
    if (x >= width || y >= height) return;
    const size_t i = (size_t)y * width + x;
    GLOBAL const float* colorIn = calib;
    GLOBAL const float* extr = calib + INTRINSICS_SIZE;
    const int cw = (int)colorIn[0], ch = (int)colorIn[1];
    const float z = mm[i] * 0.001f;
    int c = -1;
    if (z > 0.f)
    {
        const int sx = flip ? width - 1 - x : x;
        const size_t r = ((size_t)y * width + sx) * 2;
        float q[3], px[2];
        transformPoint(extr, rays[r] * z, rays[r + 1] * z, z, q);
        projectPoint(colorIn, q, px);
        const int cx = (int)(px[0] + 0.5f);
        const int cy = (int)(px[1] + 0.5f);
        if (cx >= 0 && cy >= 0 && cx < cw && cy < ch) c = cy * cw + (flip ? cw - 1 - cx : cx);
    }
    map[i] = c;
}

// Align depth into another camera, z-buffered (librealsense's align_z_to_other, as
// ofxLibRealSense2Processing::alignDepthToOther): each depth pixel's footprint, between its
// corners' projections, filled with its depth, the nearest winning. out starts at 0xffffffff.
// corners: rays through the pixel corners (x - 0.5, y - 0.5), (width + 1) x (height + 1).
// calib: other intrinsics, then depth -> other extrinsics.
KERNEL k_alignSplat(GLOBAL const u16* depth, GLOBAL const float* corners, GLOBAL const float* calib, GLOBAL u32* out,
                    int width, int height, float scale)
{
    const int x = GX, y = GY;
    if (x >= width || y >= height) return;
    const u16 raw = depth[(size_t)y * width + x];
    if (!raw) return;
    const float z = scale * raw;
    GLOBAL const float* otherIn = calib;
    GLOBAL const float* extr = calib + INTRINSICS_SIZE;
    const int ow = (int)otherIn[0], oh = (int)otherIn[1];

    float p[3], px[2];
    size_t r = ((size_t)y * (width + 1) + x) * 2;
    transformPoint(extr, z * corners[r], z * corners[r + 1], z, p);
    projectPoint(otherIn, p, px);
    const int x0 = (int)(px[0] + 0.5f);
    const int y0 = (int)(px[1] + 0.5f);

    r = ((size_t)(y + 1) * (width + 1) + x + 1) * 2;
    transformPoint(extr, z * corners[r], z * corners[r + 1], z, p);
    projectPoint(otherIn, p, px);
    const int x1 = (int)(px[0] + 0.5f);
    const int y1 = (int)(px[1] + 0.5f);

    if (x0 < 0 || y0 < 0 || x1 >= ow || y1 >= oh) return;
    for (int yy = y0; yy <= y1; ++yy)
    {
        for (int xx = x0; xx <= x1; ++xx) ATOMIC_MIN(out + (size_t)yy * ow + xx, (u32)raw);
    }
}

KERNEL k_fill32(GLOBAL u32* data, int count, u32 value)
{
    const int i = GX;
    if (i < count) data[i] = value;
}

// Aligned depth (0xffffffff = none) -> float mm, flipped when asked.
KERNEL k_alignedToMM(GLOBAL const u32* aligned, GLOBAL float* mm, int width, int height, float toMM, int flip)
{
    const int x = GX, y = GY;
    if (x >= width || y >= height) return;
    const u32 v = aligned[(size_t)y * width + (flip ? width - 1 - x : x)];
    mm[(size_t)y * width + x] = (v == 0xffffffffu ? 0u : v) * toMM;
}
)OFXRS2"
R"OFXRS2(
// ---------------------------------------------------------------------------------------
// Point cloud / mesh (ofxLibRealSense2::buildPointCloud)
// ---------------------------------------------------------------------------------------

// Edge-preserving 3x3 average of depth (mm) around one pixel (denoisedDepthAt).
DEVICE float denoisedDepthAt(GLOBAL const float* depth, int w, int h, int x, int y, int spacing)
{
    const float center = depth[y * w + x];
    if (!(center > 0.f && center < 1e6f)) return center;
    const float threshold = fmax(15.f, center * 0.015f);
    float sum = 0.f;
    int count = 0;
    for (int dy = -spacing; dy <= spacing; dy += spacing)
    {
        const int yy = y + dy;
        if (yy < 0 || yy >= h) continue;
        GLOBAL const float* row = depth + yy * w;
        for (int dx = -spacing; dx <= spacing; dx += spacing)
        {
            const int xx = x + dx;
            if (xx < 0 || xx >= w) continue;
            const float n = row[xx];
            if (n > 0.f && fabs(n - center) < threshold)
            {
                sum += n;
                ++count;
            }
        }
    }
    return sum / count;
}

// Per grid sample: position (camera space, metres; written whenever the depth is in range),
// valid (in range and, with colour, seen by the colour camera), colour index (-1: none).
// Mesh: also the per-vertex texture coordinates / colours (zero where invalid).
// opts: [0] flip, [1] denoise, [2] spacing, [3] step, [4] gw, [5] gh, [6] hasColor, [7] big,
//       [8] colorW, [9] meshTexCoords, [10] meshColors, [11] channels
KERNEL k_cloudSamples(GLOBAL const float* depth, GLOBAL const float* rays, GLOBAL const int* colorMap, GLOBAL const u8* color,
                      GLOBAL const int* opts, int w, int h, float minZ, float maxZ, float alpha,
                      GLOBAL float* pos, GLOBAL u8* valid, GLOBAL int* colorIndex, GLOBAL float* texCoords, GLOBAL float* colors)
{
    const int gx = GX, gy = GY;
    const int flip = opts[0], denoise = opts[1], spacing = opts[2], step = opts[3], gw = opts[4], gh = opts[5];
    const int hasColor = opts[6], big = opts[7], colorW = opts[8], meshTex = opts[9], meshCol = opts[10], channels = opts[11];
    if (gx >= gw || gy >= gh) return;
    const int u = gx * step, v = gy * step;
    const size_t i = (size_t)gy * gw + gx;
    valid[i] = 0;
    colorIndex[i] = -1;
    pos[i * 3 + 0] = 0.f;
    pos[i * 3 + 1] = 0.f;
    pos[i * 3 + 2] = 0.f;
    if (meshTex)
    {
        texCoords[i * 2 + 0] = 0.f;
        texCoords[i * 2 + 1] = 0.f;
    }
    if (meshCol)
    {
        colors[i * 4 + 0] = 0.f;
        colors[i * 4 + 1] = 0.f;
        colors[i * 4 + 2] = 0.f;
        colors[i * 4 + 3] = 0.f;
    }

    const float z = (denoise ? denoisedDepthAt(depth, w, h, u, v, spacing) : depth[v * w + u]) * 0.001f;
    if (!(z > minZ && z < maxZ)) return;
    const size_t r = ((size_t)v * w + (flip ? w - 1 - u : u)) * 2;
    pos[i * 3 + 0] = (flip ? -rays[r] : rays[r]) * z;
    pos[i * 3 + 1] = rays[r + 1] * z;
    pos[i * 3 + 2] = z;

    int c = -1;
    if (hasColor) c = big ? v * colorW + u : colorMap[v * w + u];
    if (hasColor && c < 0) return;
    valid[i] = 1;
    colorIndex[i] = c;
    if (meshTex)
    {
        texCoords[i * 2 + 0] = (float)(c % colorW);
        texCoords[i * 2 + 1] = (float)(c / colorW);
    }
    if (meshCol)
    {
        GLOBAL const u8* p = color + (size_t)c * channels;
        colors[i * 4 + 0] = p[0] / 255.f;
        colors[i * 4 + 1] = p[1] / 255.f;
        colors[i * 4 + 2] = p[2] / 255.f;
        colors[i * 4 + 3] = alpha;
    }
}

// Exclusive prefix sum of `count` values into start[0..count], start[count] = total. One
// thread: the counts are per row (hundreds).
KERNEL k_scan(GLOBAL const u32* counts, GLOBAL u32* start, int count)
{
    if (GX != 0) return;
    u32 sum = 0;
    for (int i = 0; i < count; ++i)
    {
        start[i] = sum;
        sum += counts[i];
    }
    start[count] = sum;
}

)OFXRS2"
R"OFXRS2(
// glm's dot (((x + y) + z)) and cross, on float[3]
DEVICE float dot3(const float* a, const float* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
DEVICE void cross3(const float* x, const float* y, float* o)
{
    o[0] = x[1] * y[2] - y[1] * x[2];
    o[1] = x[2] * y[0] - y[2] * x[0];
    o[2] = x[0] * y[1] - y[0] * x[1];
}
DEVICE void load3(GLOBAL const float* pos, size_t i, float* o)
{
    o[0] = pos[i * 3 + 0];
    o[1] = pos[i * 3 + 1];
    o[2] = pos[i * 3 + 2];
}
DEVICE void sub3(GLOBAL const float* pos, size_t a, size_t b, float* o)
{
    o[0] = pos[a * 3 + 0] - pos[b * 3 + 0];
    o[1] = pos[a * 3 + 1] - pos[b * 3 + 1];
    o[2] = pos[a * 3 + 2] - pos[b * 3 + 2];
}

// edgeOk: shorter than the max edge and within the relative depth jump
DEVICE int edgeOk(GLOBAL const float* pos, size_t a, size_t b, float maxEdge2, float maxJump)
{
    float d[3];
    sub3(pos, a, b, d);
    return dot3(d, d) < maxEdge2 && fabs(d[2]) < maxJump * fmin(pos[a * 3 + 2], pos[b * 3 + 2]);
}

DEVICE int triangleOk(GLOBAL const float* pos, GLOBAL const u8* valid, size_t a, size_t b, size_t c, float maxEdge2, float maxJump)
{
    return valid[a] && valid[b] && valid[c] && edgeOk(pos, a, b, maxEdge2, maxJump) && edgeOk(pos, b, c, maxEdge2, maxJump)
        && edgeOk(pos, c, a, maxEdge2, maxJump);
}

DEVICE float dist2(GLOBAL const float* pos, size_t a, size_t b)
{
    float d[3];
    sub3(pos, a, b, d);
    return dot3(d, d);
}

// Mesh normals: central differences over valid neighbours on the same surface, falling back
// to one-sided ones, facing the camera. Zero where invalid or degenerate.
KERNEL k_normals(GLOBAL const float* pos, GLOBAL const u8* valid, GLOBAL float* normals, int gw, int gh, float maxEdge2, float maxJump)
{
    const int gx = GX, gy = GY;
    if (gx >= gw || gy >= gh) return;
    const size_t i = (size_t)gy * gw + gx;
    normals[i * 3 + 0] = 0.f;
    normals[i * 3 + 1] = 0.f;
    normals[i * 3 + 2] = 0.f;
    if (!valid[i]) return;

    size_t nb[4];
    const int nx[4] = { gx + 1, gx - 1, gx, gx };
    const int ny[4] = { gy, gy, gy + 1, gy - 1 };
    for (int k = 0; k < 4; ++k)
    {
        nb[k] = i;
        if (nx[k] < 0 || ny[k] < 0 || nx[k] >= gw || ny[k] >= gh) continue;
        const size_t n = (size_t)ny[k] * gw + nx[k];
        if (valid[n] && edgeOk(pos, i, n, maxEdge2, maxJump)) nb[k] = n;
    }
    float dx[3], dy[3], n[3], p[3];
    sub3(pos, nb[0], nb[1], dx);
    sub3(pos, nb[2], nb[3], dy);
    cross3(dx, dy, n);
    const float len2 = dot3(n, n);
    if (len2 < 1e-20f) return;
    const float len = sqrt(len2);
    n[0] /= len;
    n[1] /= len;
    n[2] /= len;
    load3(pos, i, p);
    const float s = dot3(n, p) > 0.f ? -1.f : 1.f;
    normals[i * 3 + 0] = s * n[0];
    normals[i * 3 + 1] = s * n[1];
    normals[i * 3 + 2] = s * n[2];
}

// The triangles of one quad (tl, tr, bl, br) as the CPU picks them; writes up to 6 indices to
// out (when not null), returns how many. Every triangle is wound to face the camera.
DEVICE int pushTriangle(GLOBAL const float* pos, size_t a, size_t b, size_t c, GLOBAL u32* out, int n)
{
    if (out)
    {
        float ab[3], ac[3], nn[3], pa[3];
        sub3(pos, b, a, ab);
        sub3(pos, c, a, ac);
        cross3(ab, ac, nn);
        load3(pos, a, pa);
        if (dot3(nn, pa) > 0.f)
        {
            const size_t t = b;
            b = c;
            c = t;
        }
        out[n + 0] = (u32)a;
        out[n + 1] = (u32)b;
        out[n + 2] = (u32)c;
    }
    return n + 3;
}

DEVICE int quadTriangles(GLOBAL const float* pos, GLOBAL const u8* valid, int gw, int gx, int gy, float maxEdge2, float maxJump, GLOBAL u32* out)
{
    const size_t tl = (size_t)gy * gw + gx;
    const size_t tr = tl + 1;
    const size_t bl = tl + gw;
    const size_t br = bl + 1;
    const int validCount = valid[tl] + valid[tr] + valid[bl] + valid[br];
    int n = 0;
    if (validCount < 3) return 0;
    if (validCount == 4)
    {
        if (dist2(pos, tl, br) <= dist2(pos, tr, bl))
        {
            if (triangleOk(pos, valid, tl, bl, br, maxEdge2, maxJump)) n = pushTriangle(pos, tl, bl, br, out, n);
            if (triangleOk(pos, valid, tl, br, tr, maxEdge2, maxJump)) n = pushTriangle(pos, tl, br, tr, out, n);
        }
        else
        {
            if (triangleOk(pos, valid, tl, bl, tr, maxEdge2, maxJump)) n = pushTriangle(pos, tl, bl, tr, out, n);
            if (triangleOk(pos, valid, tr, bl, br, maxEdge2, maxJump)) n = pushTriangle(pos, tr, bl, br, out, n);
        }
    }
    else
    {
        if (!valid[tl]) { if (triangleOk(pos, valid, tr, bl, br, maxEdge2, maxJump)) n = pushTriangle(pos, tr, bl, br, out, n); }
        else if (!valid[tr]) { if (triangleOk(pos, valid, tl, bl, br, maxEdge2, maxJump)) n = pushTriangle(pos, tl, bl, br, out, n); }
        else if (!valid[bl]) { if (triangleOk(pos, valid, tl, br, tr, maxEdge2, maxJump)) n = pushTriangle(pos, tl, br, tr, out, n); }
        else { if (triangleOk(pos, valid, tl, bl, tr, maxEdge2, maxJump)) n = pushTriangle(pos, tl, bl, tr, out, n); }
    }
    return n;
}

// Camera -> world: positions by m (column-major mat4, glm's (m0 x + m1 y) + (m2 z + m3)),
// normals by the inverse transpose (column-major mat3), renormalised.
KERNEL k_transform(GLOBAL float* pos, GLOBAL float* normals, GLOBAL const float* m, int count, int hasNormals)
{
    const int i = GX;
    if (i >= count) return;
    const float x = pos[i * 3 + 0], y = pos[i * 3 + 1], z = pos[i * 3 + 2];
    for (int k = 0; k < 3; ++k)
    {
        pos[i * 3 + k] = (m[k] * x + m[4 + k] * y) + (m[8 + k] * z + m[12 + k] * 1.f);
    }
    if (hasNormals)
    {
        GLOBAL const float* nm = m + 16;
        const float nx = normals[i * 3 + 0], ny = normals[i * 3 + 1], nz = normals[i * 3 + 2];
        float t[3];
        for (int k = 0; k < 3; ++k) t[k] = nm[k] * nx + nm[3 + k] * ny + nm[6 + k] * nz;
        const float len = sqrt(dot3(t, t));
        if (len > 1e-12f)
        {
            normals[i * 3 + 0] = t[0] / len;
            normals[i * 3 + 1] = t[1] / len;
            normals[i * 3 + 2] = t[2] / len;
        }
    }
}
)OFXRS2"
R"OFXRS2(
// ---------------------------------------------------------------------------------------
// Work-group versions (GROUP_SIZE threads per group, one group per line / row): the line-walking
// and compaction steps are sequential per line, which is slow from global memory one thread per
// line. Same results: the same per-line code, or an order-preserving prefix sum.
// ---------------------------------------------------------------------------------------

// Spatial filter along the rows of a width x height image, one group per row: the row is copied
// to local memory, one thread filters it there (forward, then backward), and it is copied back.
// Columns: transpose, this, transpose back. width <= MAX_LINE.
KERNEL k_spatialRowsShared(GLOBAL float* image, int width, int height, float alpha, float deltaZ)
{
    LOCAL_ARRAY(float, line, MAX_LINE);
    const int y = GROUP;
    if (y >= height) return;
    GLOBAL float* row = image + (size_t)y * width;
    for (int x = LID; x < width; x += GROUP_SIZE) line[x] = row[x];
    BARRIER();
    if (LID == 0 && width >= 2)
    {
        filterLineLocal(line, width, 1, alpha, deltaZ);
        filterLineLocal(line + (width - 1), width, -1, alpha, deltaZ);
    }
    BARRIER();
    for (int x = LID; x < width; x += GROUP_SIZE) row[x] = line[x];
}

// out (height x width) = in (width x height) transposed, through a 16 x 16 tile (2D launch).
KERNEL k_transpose(GLOBAL const float* in, GLOBAL float* out, int width, int height)
{
    LOCAL_ARRAY(float, tile, 16 * 17);
    const int x0 = GX - LX, y0 = GY - LY;
    if (GX < width && GY < height) tile[LY * 17 + LX] = in[(size_t)GY * width + GX];
    BARRIER();
    const int ox = y0 + LX, oy = x0 + LY;
    if (ox < height && oy < width) out[(size_t)oy * height + ox] = tile[LX * 17 + LY];
}

// Exclusive prefix sum of one value per thread over the group; *total = the group's sum.
DEVICE int groupScan(LOCAL int* s, int v, int* total)
{
    const int l = LID;
    s[l] = v;
    BARRIER();
    for (int offset = 1; offset < GROUP_SIZE; offset <<= 1)
    {
        const int t = l >= offset ? s[l - offset] : 0;
        BARRIER();
        s[l] += t;
        BARRIER();
    }
    const int inclusive = s[l];
    *total = s[GROUP_SIZE - 1];
    BARRIER();
    return inclusive - v;
}

// Points: valid samples per grid row, one group per row.
KERNEL k_pointsRowCountGroup(GLOBAL const u8* valid, GLOBAL u32* counts, int gw, int gh)
{
    LOCAL_ARRAY(int, s, GROUP_SIZE);
    const int gy = GROUP;
    if (gy >= gh) return;
    int n = 0;
    for (int gx = LID; gx < gw; gx += GROUP_SIZE) n += valid[(size_t)gy * gw + gx];
    int total;
    groupScan(s, n, &total);
    if (LID == 0) counts[gy] = (u32)total;
}

// Points: valid samples compacted in row-major order, one group per row, GROUP_SIZE samples at
// a time.
KERNEL k_pointsCompactGroup(GLOBAL const float* pos, GLOBAL const u8* valid, GLOBAL const int* colorIndex, GLOBAL const u32* start,
                            GLOBAL const u8* color, int gw, int gh, int useTexCoords, int useColors, int colorW, int channels, float alpha,
                            GLOBAL float* outPos, GLOBAL float* outTexCoords, GLOBAL float* outColors)
{
    LOCAL_ARRAY(int, s, GROUP_SIZE);
    const int gy = GROUP;
    if (gy >= gh) return;
    size_t base = start[gy];
    for (int c = 0; c < gw; c += GROUP_SIZE)
    {
        const int gx = c + LID;
        const size_t i = (size_t)gy * gw + gx;
        const int v = gx < gw ? valid[i] : 0;
        int total;
        const int prefix = groupScan(s, v, &total);
        if (v)
        {
            const size_t o = base + prefix;
            outPos[o * 3 + 0] = pos[i * 3 + 0];
            outPos[o * 3 + 1] = pos[i * 3 + 1];
            outPos[o * 3 + 2] = pos[i * 3 + 2];
            const int ci = colorIndex[i];
            if (useTexCoords)
            {
                outTexCoords[o * 2 + 0] = (float)(ci % colorW);
                outTexCoords[o * 2 + 1] = (float)(ci / colorW);
            }
            if (useColors)
            {
                GLOBAL const u8* p = color + (size_t)ci * channels;
                outColors[o * 4 + 0] = p[0] / 255.f;
                outColors[o * 4 + 1] = p[1] / 255.f;
                outColors[o * 4 + 2] = p[2] / 255.f;
                outColors[o * 4 + 3] = alpha;
            }
        }
        base += total;
    }
}

// Mesh: indices per row of quads, one group per row.
KERNEL k_trianglesRowCountGroup(GLOBAL const float* pos, GLOBAL const u8* valid, GLOBAL u32* counts, int gw, int gh, float maxEdge2, float maxJump)
{
    LOCAL_ARRAY(int, s, GROUP_SIZE);
    const int gy = GROUP;
    if (gy >= gh - 1) return;
    int n = 0;
    for (int gx = LID; gx < gw - 1; gx += GROUP_SIZE) n += quadTriangles(pos, valid, gw, gx, gy, maxEdge2, maxJump, 0);
    int total;
    groupScan(s, n, &total);
    if (LID == 0) counts[gy] = (u32)total;
}

// Mesh: the indices, row-major, one group per row of quads, GROUP_SIZE quads at a time.
KERNEL k_trianglesEmitGroup(GLOBAL const float* pos, GLOBAL const u8* valid, GLOBAL const u32* start, GLOBAL u32* indices,
                            int gw, int gh, float maxEdge2, float maxJump)
{
    LOCAL_ARRAY(int, s, GROUP_SIZE);
    const int gy = GROUP;
    if (gy >= gh - 1) return;
    size_t base = start[gy];
    for (int c = 0; c < gw - 1; c += GROUP_SIZE)
    {
        const int gx = c + LID;
        const int n = gx < gw - 1 ? quadTriangles(pos, valid, gw, gx, gy, maxEdge2, maxJump, 0) : 0;
        int total;
        const int prefix = groupScan(s, n, &total);
        if (n) quadTriangles(pos, valid, gw, gx, gy, maxEdge2, maxJump, indices + base + prefix);
        base += total;
    }
}
)OFXRS2"
