# Building librealsense for ofxLibRealSense2

> Bringing the addon up on Windows or Linux? Read [HANDOFF_WINDOWS_LINUX.md](HANDOFF_WINDOWS_LINUX.md) first.

ofxLibRealSense2 links [librealsense](https://github.com/IntelRealSense/librealsense) built from
source with the scripts here, with the same options on every platform. Built and tested at
**v2.59.1** (`c84fe6438`). Clone it anywhere and pass its path to the script.

| | macOS | Windows | Linux |
|---|---|---|---|
| Library | static `librealsense2.a`, universal (arm64 + x86_64), macOS 11+ | `realsense2.dll` + `.lib` | static `librealsense2.a` |
| Backend | libusb (RSUSB, required on macOS) | Media Foundation (native) | libusb (`--v4l2` for the kernel backend) |
| SIMD | NEON (arm64), SSSE3 / AVX (x86_64) | SSSE3 / AVX | SSSE3 / AVX or NEON |
| CUDA (librealsense's own) | n/a | on when a CUDA Toolkit is found (`-NoCuda`), static runtime | off |
| Recording | `.bag` (rosbag 1) | same | same |
| Off | examples, tools, tests, OpenMP, update checks, usage statistics, DDS, rosbag2 | same | same |
| Status | built and tested | built and tested (VS 2026, CUDA 12.1) | not yet run |

The addon's own GPU processing (CUDA / OpenCL, see the main README) needs nothing from these
builds: its kernels are compiled separately ([GPU kernels](#gpu-kernels)) and the GPU APIs are
loaded at run time.

The C++ API (`rs.hpp`) is header-only over librealsense's C API, so:

- the static libraries pre-link librealsense with all it bundles (libusb, json, easylogging, lz4...)
  into one object that exports only `rs2_*`: nothing inside can clash with openFrameworks (which has
  its own libusb, json...) or other addons such as ofxKinectV2;
- one Release DLL serves Debug and Release apps on Windows.

Output:

```
libs/librealsense2/include/librealsense2/       headers (the same everywhere)
libs/librealsense2/lib/osx/librealsense2.a
libs/librealsense2/lib/vs/x64/realsense2.lib    + realsense2.dll (copied next to the exe)
libs/librealsense2/lib/linux64/librealsense2.a  (linuxaarch64 on ARM)
```

Then regenerate your project with the projectGenerator.

## macOS

```bash
brew install cmake
./build_librealsense_macos.sh /path/to/librealsense            # universal (what's in the repo)
./build_librealsense_macos.sh /path/to/librealsense --native   # this Mac's architecture only
```

Needs full Xcode. What the script takes care of:

- **Universal:** librealsense picks its SIMD code from `$CC -dumpmachine`, so each architecture is a
  separate build (the other one with compiler wrappers that target it), joined with `lipo`.
- **libusb** is librealsense's own static copy (`USE_EXTERNAL_USB=ON`), not Homebrew's dylib, built
  without `pipe2()`: the current SDK declares it but only macOS 27+ has it, and an unguarded call
  would crash older systems. Common symbols are turned off (`-fno-common`) so libusb's globals can
  be hidden too. Both through a toolchain file, which librealsense hands on to its libusb build.
- **One object:** `ld -r` over all of librealsense's static libraries plus libusb, exporting only
  `_rs2_*` (rsutils' duplicate copy of easylogging++ is dropped). librealsense's own
  `realsense2-all` bundle needs GNU `ar`, which macOS doesn't have.
- **rosbag2 off:** its yaml-cpp only arrives as headers in a static build. Recording and playback use
  the classic `.bag` format, which RealSense Viewer reads and writes too.

At run time librealsense needs **root** on macOS 12+ (see the main README).

## Windows

```powershell
.\build_librealsense_windows.ps1 -Source C:\path\to\librealsense
.\build_librealsense_windows.ps1 -Source C:\path\to\librealsense -NoCuda
.\build_librealsense_windows.ps1 -Source C:\path\to\librealsense -CudaPath "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
```

Needs Visual Studio 2022 or newer with the C++ workload (its CMake and Ninja are used) and, for
CUDA, a CUDA Toolkit (`CUDA_PATH` or `-CudaPath`). The native Media Foundation backend needs no
driver changes or admin rights (unlike the Kinect v2, no Zadig). The C runtime is linked statically,
so `realsense2.dll` is all there is to ship. One Release DLL serves Debug and Release apps.

**CUDA** is on when a Toolkit is found. librealsense uses it for colour format conversion (every
colour frame, on librealsense's own thread) and its own align / point cloud, and checks for an
NVIDIA driver at run time, falling back to SSE without one. The script keeps the DLL shippable:

- the CUDA runtime is linked statically, and cuBLAS / cuSPARSE, which librealsense links but never
  calls, end up with no import: `realsense2.dll` depends on system DLLs only (the script warns if
  that changes) and loads on any PC;
- sm_52 and sm_61 (GTX 9xx / 10xx, e.g. the 1080 Ti) are added: librealsense's list leaves them out;
- nvcc only accepts host compilers it knows. When the newest MSVC is too new for the Toolkit (CUDA
  12.1's front end crashes on VS 2026's 14.5x headers), the script finds the newest older toolset
  nvcc accepts (VS 2022's 14.44 - "MSVC v143 build tools" in the VS 2026 installer) and builds the
  library with it. That toolset may come without ATL, which the Media Foundation backend needs, so
  the newest toolset's ATL headers are used. The app can use any toolset: only the C API crosses
  the DLL boundary. `-VcVars 14.44` picks the toolset by hand.

Tested: VS 2026 (MSVC 14.51, and 14.44 for the CUDA build), CUDA 12.1, GTX 1080 Ti, D415.

## GPU kernels

The addon's CUDA and OpenCL processing share one kernel source, `src/ofxLibRealSense2Kernels.h`.
OpenCL compiles it at run time. CUDA runs it as PTX compiled ahead of time and embedded in
`src/ofxLibRealSense2KernelsPtx.h` (committed), so apps need no CUDA Toolkit and ship no CUDA DLLs.
After editing the kernels, regenerate it:

```powershell
.\build_gpu_kernels.ps1                   # Windows (CUDA Toolkit + Visual Studio)
```
```bash
./build_gpu_kernels.sh [path/to/nvcc]      # Linux
```

The PTX targets compute_52 (Maxwell / GTX 9xx and everything newer: the driver compiles it for the
GPU it finds, once, and caches the result) and is built with `-fmad=false`, so the kernels round
exactly like the CPU code (OpenCL: `FP_CONTRACT OFF` and correctly rounded division). Apps need an
NVIDIA driver at least as new as the Toolkit that made the PTX - CUDA 12.1: driver 530 or newer.

## Linux

```bash
sudo apt install build-essential cmake git pkg-config libudev-dev libssl-dev
./build_librealsense_linux.sh /path/to/librealsense
sudo cp /path/to/librealsense/config/99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```

The libusb backend needs no kernel patches; the udev rules let users open the cameras without root.
`--v4l2` builds the kernel backend instead (frame metadata then needs librealsense's kernel
patches). Same pre-linking as macOS with GNU tools (`ld -r -d`, `objcopy --keep-global-symbol`).
The app links `-ludev -lpthread` (`addon_config.mk`).

## Testing a build

librealsense's own tools aren't built. A quick check without openFrameworks: build any program
against the library (see the frameworks in `addon_config.mk`) and list the devices - or just run
`example-streams` (with `sudo` on macOS).
