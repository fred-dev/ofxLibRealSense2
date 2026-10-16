# ofxLibRealSense2

openFrameworks addon for Intel RealSense depth cameras (D400 / D500 series: D415, D435, D435i,
D455, D405...) through [librealsense 2](https://github.com/IntelRealSense/librealsense) (2.59).

> This is a rewrite of [shiyamon/ofxLibRealSense2](https://github.com/shiyamon/ofxLibRealSense2),
> kept as a fork. The API changed - see [Migrating](#migrating-from-the-previous-version).

- Several cameras at once, each on its own capture thread; one librealsense context is shared.
- Colour, depth, IR, colour registered to depth, all as `ofPixels` / `ofFloatPixels` (depth in mm).
- Point cloud or triangulated mesh with normals, built on the capture thread, at depth or colour
  resolution, textured from the colour image.
- librealsense's post-processing (decimation, spatial, temporal, hole filling), with the two
  expensive steps - the spatial filter and depth -> colour alignment - multi-threaded in the
  addon, with the same results.
- Optional processing on the GPU with **CUDA or OpenCL** (experimental, off by default):
  filters, alignment, registration and point clouds / meshes, bit-identical to the CPU path.

Point clouds are in metres, camera space with +X right, +Y down, +Z forward, then an optional
camera -> world matrix - the same as ofxKinectV2 and ofxK4a.

## Platforms

| | Library | Status |
|---|---|---|
| macOS (arm64 + x86_64) | static `librealsense2.a`, universal, macOS 11+ (prebuilt) | built and tested (M3 Ultra, D415) |
| Windows x64 | `realsense2.dll`, Media Foundation backend, CUDA colour conversion (prebuilt) | built and tested (i7-7700K + GTX 1080 Ti, D415; CPU, CUDA, OpenCL) |
| Linux x64 / aarch64 | static `librealsense2.a`, libusb backend | build script, not yet run |

Build or rebuild librealsense with the scripts in [`scripts/`](scripts/README.md). Notes for
whoever brings up Windows and Linux: [`scripts/HANDOFF_WINDOWS_LINUX.md`](scripts/HANDOFF_WINDOWS_LINUX.md).

### Windows

No driver changes or admin rights (librealsense uses Windows' own camera stack). Generate the
projects with the projectGenerator; `realsense2.dll` is copied next to the exe. Make the project's
platform toolset match `openframeworksLib`'s (Visual Studio 2026 builds openFrameworks with v145
while the `vs` template still says v143; a mismatch fails at link time with `C1900`). Close other
apps that hold the camera (RealSense Viewer, a browser using it as a webcam).

### macOS: run as root

**On macOS 12 and later librealsense needs root.** It talks to the camera through libusb, and
macOS's own camera driver keeps the camera's video interfaces from anything that isn't root (this
is librealsense's documented requirement, not the addon's). Without it the camera is listed but
`open()` fails with "cannot access depth sensor" (the addon's log says what to do).

- From a terminal: `sudo ./bin/example-streams.app/Contents/MacOS/example-streams`
- From Xcode: Product > Scheme > Edit Scheme > Run > Info > Debug Process As: **root**
- Xcode 26+: set `MACOSX_DEPLOYMENT_TARGET = 12.0` in the project's `Project.xcconfig`
  (openFrameworks' template says 11.5, which Xcode 26+ refuses).

## Usage

```cpp
// ofApp.h
std::vector<std::shared_ptr<ofxLibRealSense2>> cameras; // a thread: keep it by pointer
ofTexture texColour;

// setup()
ofEnableArbTex(); // texture coordinates are pixels of the colour image
ofxLibRealSense2::Settings settings;
settings.pointCloudHasFaces = true;
for (const auto& device : ofxLibRealSense2::listDevices()) // sorted by serial
{
    auto camera = std::make_shared<ofxLibRealSense2>();
    if (camera->open(device.serial, settings)) cameras.push_back(camera);
}

// update()
for (auto& camera : cameras)
{
    camera->update();
    if (camera->isFrameNew()) texColour.loadData(camera->getPixels());
}

// draw(), inside an ofEasyCam
ofScale(1000.f, -1000.f, -1000.f); // metres -> mm, camera space -> openFrameworks
texColour.bind();
cameras[0]->drawPointCloud();
texColour.unbind();
```

Useful `Settings`: stream sizes and rate (`depthWidth`, `colorWidth`, `fps`; 0 = the camera's
default), `enableRGB` / `enableIR` / `enableDepth` / `registerImages` / `usePointCloud`,
`decimation` / `spatialFilter` / `temporalFilter` / `holeFilling`, `processing` /
`processingDevice` (CPU or GPU, below), `syncMode` (hardware sync), `playbackFile` / `recordFile`.
Most can also be changed live through `camera->params`.

`getCaptureStats()` reports fps, the capture thread's time per frame (`processMs`, of which
`buildMs` is the point cloud), the processing in use, connection, dropped frames and the colour /
depth time offset.

### CPU, CUDA or OpenCL (experimental)

Everything the capture thread does after librealsense hands over a frame - the depth filters
(disparity transform, spatial, temporal), the depth image, depth seen from the colour camera, the
depth -> colour map and the point cloud or mesh (normals, triangles, transform) - can also run on
the GPU, with exactly the same result as the CPU path: every value was checked bit for bit against
it on live frames, in every mode (build with `OFXLIBREALSENSE2_VERIFY_GPU` defined to repeat it).

**It's off by default because it hasn't been faster yet.** On the one machine it was tested on (GTX
1080 Ti, i7-7700K, Windows) the GPU paths were 20% faster for colour-resolution clouds and slower
for the default depth-resolution points (see [Performance](#performance)): the kernels take a few
ms, but the transfers and waits between the capture thread and a GPU that is also drawing the app
cost more. A newer GPU, a faster bus or less GPU traffic may change that - try it on yours:

```cpp
settings.processing = ofxLibRealSense2::Processing::CUDA;   // then OpenCL, then the CPU
// or OPENCL (then the CPU); DEFAULT / CPU = the CPU
settings.processingDevice = -1; // GPU index, see ofxLibRealSense2::listProcessingDevices(api)
...
camera->getProcessing();              // what was picked
camera->getCaptureStats().processing; // e.g. "CUDA NVIDIA GeForce GTX 1080 Ti (sm_61)"
```

- **Nothing to install or ship.** CUDA comes from the NVIDIA driver (530 or newer) and OpenCL from
  the GPU vendor's driver; both are loaded at run time, so the same app runs on a machine with
  neither and uses the CPU. The kernels are built into the addon (PTX for CUDA, source for OpenCL;
  see `scripts/README.md` to rebuild them).
- If a GPU fails while running, the camera logs it and carries on on the CPU.
- `OFXLIBREALSENSE2_PROCESSING=cuda` / `opencl` / `cpu` in the environment overrides the setting,
  to compare on a machine without rebuilding (`getCaptureStats().processMs` / `buildMs`).
- Decimation stays in librealsense; hole filling (off by default) is sequential and runs on the CPU
  between the GPU filters and the rest.
- Separately, the Windows build of librealsense uses CUDA for its own colour conversion when there
  is an NVIDIA GPU (see `scripts/README.md`).

### Examples

| | |
|---|---|
| `example-streams` | every stream side by side, several cameras, stats |
| `example-pointcloud` | textured point cloud / mesh |
| `example-shader` | lit mesh and shader effects on the colour image |
| `example-pointCloudTransform` | placing the cloud in the world with a transform |
| `example-playback` | record to `.bag`, play back, seek, pause; drop a RealSense Viewer recording on it |

Generate the projects with the projectGenerator (`ofxGui` + `ofxLibRealSense2`).

## Performance

Capture thread time per frame (`getCaptureStats().processMs`; the point cloud's share,
`buildMs`, in brackets), D415 at 1280 x 720 depth + 1920 x 1080 colour, 30 fps.

macOS, M3 Ultra:

| | ms |
|---|---|
| filters on, point cloud | 7.8 |
| same, depth-resolution mesh | 11.5 |
| same, colour-resolution points / mesh | 12.4 / 22 |
| filters off | 4.3 |
| 848 x 480 depth + 640 x 480 colour at 60 fps | 3.6 |

These were measured before a fix: the spatial filter wasn't actually running (the disparity frame
wasn't found), so expect them to grow by roughly the filter's 1.5-1.9 ms. To be measured again.

Windows, i7-7700K (4 cores / 8 threads) + GTX 1080 Ti, `example-streams` (all streams shown):

| | CPU | CUDA | OpenCL |
|---|---|---|---|
| filters on, point cloud | 36.6 (10.8) | 46.8 (20.2) | 40.3 (21.4) |
| depth-resolution mesh | 60.4 (35.7) | 58.7 (39.6) | 56.4 (39.4) |
| colour-resolution points | 67.0 (25.7) | 52.9 (27.5) | 51.6 (27.2) |
| colour-resolution mesh | 132 (90) | 110 (81) | 131 (104) |
| filters off, colour-resolution points | 54.1 (25.1) | 53.7 (29.4) | 56.3 (35.0) |

A 4-core CPU doesn't keep up with the full streams at 30 fps: only the default point cloud comes
close. librealsense's spatial filter takes 9.8 ms on the M3 alone; the addon's multi-threaded
version of it gives identical output in 1.5-1.9 ms, and its depth -> colour alignment takes 2-3.6
ms instead of 6.9 ms.

To go faster: `decimation` 2 (a quarter of the depth pixels), a smaller depth size, `steps` > 1
for the cloud, or turning off what you don't use (IR, registration, the cloud).

### Several cameras

Each camera streams on its own thread and point clouds share the cores between cameras. USB is
usually the limit: at full resolution give each camera its own USB 3 controller (on Apple Silicon
every Thunderbolt / USB-C port has its own). For frames exposed together, connect the cameras'
sync ports and set `syncMode` to `MASTER` on one and `SLAVE` on the others.

## Differences from ofxKinectV2

- No `PacketPipelineType`: depth comes decoded from the camera.
- `getColorIntrinsics()` / `getDepthIntrinsics()` / `getDepthToColorExtrinsics()` replace
  `getIrCameraParams()` / `getColorCameraParams()`.
- Colour is RGBA (the Kinect's is BGRA); `getRawIRPixels()` is 0-255.
- `unmirror` defaults to true like ofxKinectV2, but RealSense images aren't mirrored to start
  with: false mirrors them (and the cloud).
- Camera controls are RealSense ones (laser, emitter, exposure / gain, white balance in Kelvin).
- Extra: recording and playback, hardware sync, librealsense's filters, `getDevice()` for anything
  else librealsense offers.

## Migrating from the previous version

| before | now |
|---|---|
| `setupDevice(id)`, `setupColor/IR/Depth(w, h, fps)`, `startPipeline(true)` | `Settings` + `open(id, settings)` |
| `getColorTex()`, `getIrTex()`, `getDepthTex()` | `getPixels()`, `getIRPixels()`, `getDepthPixels()` (load into your own `ofTexture`) |
| `getRawDepthTex()` | `getRawDepthPixels()` (float mm) |
| `getGui()` | `params` (add to an `ofxPanel`) |
| `exit()` | `close()` (also on destruction) |

## Credits

Original addon by YOSHIDA Shinya ([shiyamon](https://github.com/shiyamon/ofxLibRealSense2)).
Rewrite by Fred Rodrigues, sharing its design with ofxKinectV2 (originally by Theo Watson).
librealsense is Apache 2.0, by Intel / RealSense.
