# Hand-off: bringing ofxLibRealSense2 up on Windows and Linux

Notes for whoever (person or agent) builds and tests this addon on Windows and Linux. macOS is done
and tested; this file says what exists, what still has to happen per platform, and how to check it.

## Where things stand

- **librealsense v2.59.1** (`c84fe6438`), built from source with `scripts/build_librealsense_*`.
  Use the same version on the other platforms unless there's a reason to move, and note it.
- **macOS is finished and hardware-tested**: M3 Ultra, macOS 27, Xcode 27, openFrameworks git
  master, RealSense D415 (firmware updated to the current release during testing). All five
  examples build from projectGenerator projects with no edits and stream. Static universal
  library (arm64 + x86_64) in `libs/librealsense2/lib/osx`.
- **Windows is built and tested** (2026-10-16): Windows 10, VS 2026 (MSVC 14.51; 14.44 for the
  CUDA build of librealsense), CUDA 12.1, i7-7700K + GTX 1080 Ti, D415 on USB 3.2. `realsense2.dll`
  with librealsense's CUDA colour conversion is in `lib/vs/x64` (system DLLs only). The addon
  compiles warning-free with MSVC; `example-streams` streams at 30 fps and exits cleanly. Timings
  are in the main README. What the script needed is in `scripts/README.md` (Windows).
- **Linux has not been built yet.** `build_librealsense_linux.sh` matches the macOS options but
  has never run: expect small fixes. There is no `lib/linux64` yet.
- **Bug found on Windows, affects every platform:** the spatial filter never ran. After
  `toDisparity` the frame is `DISPARITY32` and `frameset::get_depth_frame()` only returns `Z16`,
  so the addon's filter got no frame. Fixed (`first_or_default(RS2_STREAM_DEPTH,
  RS2_FORMAT_DISPARITY32)`). The macOS timings in the README predate the fix: **measure macOS
  again** (expect +1.5-1.9 ms).
- **New: optional GPU processing** (`Settings::processing` = CUDA / OPENCL; default CPU),
  `src/ofxLibRealSense2Gpu.*` and `src/ofxLibRealSense2Kernels.h`. It reproduces the CPU path bit
  for bit (filters, depth images, align, colour map, cloud / mesh, transform) - checked on live
  frames in every mode with `OFXLIBREALSENSE2_VERIFY_GPU` defined, which runs both and logs
  differing values every 30 frames. On the 1080 Ti it was no faster than the CPU (transfers and
  waits dominate while the app draws on the same GPU), so it's off by default. On Linux: try
  `OFXLIBREALSENSE2_PROCESSING=cuda` / `opencl` and compare `processMs`; GCC hasn't compiled it
  yet (`addon_config.mk` links `-ldl` for the run-time loading).

## Rules

- **Don't touch the macOS side:** `libs/librealsense2/lib/osx`, `build_librealsense_macos.sh`, the
  `osx:` section of `addon_config.mk`. The headers in `libs/librealsense2/include` are shared:
  every script rewrites them from the same librealsense checkout, so use the same version.
- **Keep the CMake options identical across platforms** (table in `scripts/README.md`). If one has
  to change on one platform, write down why there.
- Keep the API the same as ofxKinectV2's where it overlaps (see the main README); the examples are
  shared with it apart from the class name.
- Prefer fixing build problems in the scripts over patching librealsense.
- Generated projects, `bin/` (except `bin/data`), `obj/`, `dll/` are git-ignored: commit sources,
  scripts, `addons.make`, `bin/data` and the libraries for your platform.

## How the addon works (what to look at when something's off)

- `ofxLibRealSense2` is an `ofThread` per camera. The capture thread waits on an `rs2::pipeline`,
  runs the filters, converts to `ofPixels`, computes the depth -> colour map, builds the point
  cloud and hands everything to `update()` under the lock (double-buffered, as ofxKinectV2).
- `src/ofxLibRealSense2Processing.h` holds the multi-threaded spatial filter and depth -> colour
  align. Both are transcriptions of librealsense's own code and were checked against it on live
  frames: the spatial filter is bit-identical; align differs on ~0.03% of pixels (depth edges)
  from librealsense's NEON version on ARM. On x86 librealsense uses an SSE align: worth comparing
  again. The check on macOS was a small program running `rs2::spatial_filter` / `rs2::align` and
  the addon's versions on the same live frames and counting differing pixels.
- One shared `rs2::context` (`getContext()`); disconnection comes from its devices-changed
  callback, which must only call `was_removed()` - listing new devices inside it makes
  librealsense open them and race a camera being opened (that broke opens on macOS).
- Camera options are applied from the parameters at open (parameters are the source of truth):
  a camera can be left with the laser off by another app.

## Windows

### What you need

- Windows 10 / 11 x64.
- **Visual Studio 2022 or 2026** with the "Desktop development with C++" workload. It brings the
  MSVC compiler, the Windows SDK, CMake and Ninja; the script uses VS's own CMake and Ninja.
  Use the same Visual Studio (platform toolset) as openFrameworks is built with: a mismatch fails
  at link time with `C1900: Il mismatch between 'P1' version ... and 'P2' version ...` (whole
  program optimisation). Generate with the `vs2026` template, or set Project > Properties >
  General > Platform Toolset to match `openframeworksLib`. (Found with ofxKinectV2.)
- Git, and PowerShell (5.1 or 7). If scripts are blocked:
  `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
- openFrameworks (tested with git master on macOS; ofxKinectV2 is tested on Windows with it).
- Not needed: the RealSense SDK installer, CUDA, Zadig / driver changes, admin rights for the camera.

### Build librealsense

```powershell
git clone https://github.com/IntelRealSense/librealsense C:\dev\librealsense
cd C:\dev\librealsense; git checkout v2.59.1
cd C:\path\to\openFrameworks\addons\ofxLibRealSense2\scripts
.\build_librealsense_windows.ps1 -Source C:\dev\librealsense
```

What the script does: enters the VS developer shell (x64), configures with Ninja (single config,
Release) and the options in `scripts/README.md`, builds only the `realsense2` target, then copies

```
libs/librealsense2/include/librealsense2/      (rewritten from the checkout - same as macOS)
libs/librealsense2/lib/vs/x64/realsense2.lib   import library
libs/librealsense2/lib/vs/x64/realsense2.dll   static C runtime: nothing else to ship
```

The configure step downloads librealsense's own dependencies (nlohmann json, etc.), so it needs
network access the first time. One Release DLL serves Debug and Release apps (C API underneath).

### Build the examples

1. projectGenerator: import `example-streams` (addons `ofxGui`, `ofxLibRealSense2`), template
   `vs` / `vs2026` to match openFrameworks, Update.
2. Open the `.sln`, build **Release x64**, then **Debug x64**.
3. Check `realsense2.dll` ended up next to the exe in `bin/` (see below), plug in the camera
   (USB 3 port and cable), run. The status line should show the camera, `USB 3.x`,
   librealsense 2.59.1 and ~30 fps.
4. Repeat for the other four examples, then go through "How to check a platform is done".
5. Update this file, `scripts/README.md` and the main README (the Windows rows), commit
   `libs/librealsense2/lib/vs` and any script / addon fixes.

### Checking the library

- `dumpbin /exports libs\librealsense2\lib\vs\x64\realsense2.dll` should list the C API (`rs2_*`).
- `dumpbin /dependents ...\realsense2.dll`: system DLLs only (no MSVC runtime, no CUDA).
- A camera that's listed but won't open: another app has it (RealSense Viewer, a browser using the
  RGB camera as a webcam, Logitech G HUB...).

What came up (2026-10-16):

- **The DLL next to the exe:** works. The projectGenerator puts `realsense2.dll` in the project's
  `dll/x64` and the template's post-build step copies it next to the exe.
- **projectGenerator:** its command line takes long options only (`--ofPath=... --platforms=vs
  --template=vs2026 --addons=ofxGui,ofxLibRealSense2 <project>`); it reported "cannot find vs2026"
  and used the `vs` template (v143). Build with the toolset openframeworksLib uses
  (`msbuild /p:PlatformToolset=v145`, or set it in the project).
- **Frame metadata:** not set up (Intel's registry script wasn't run); `colorDepthOffsetMs` showed
  0.3 ms anyway. Not investigated further.
- **CUDA:** librealsense's own is now on in the Windows build (colour conversion), shippable - see
  `scripts/README.md` for what the script does about cuBLAS / cuSPARSE, sm_61 and nvcc's host
  compiler. The addon's own GPU processing is separate and off by default (see above).
- **MSVC vs the addon code:** compiled without warnings in the addon's files; no changes needed.
- The `control_transfer ... timed out` warnings seen on macOS: not checked (they only show with OF_LOG_VERBOSE); no errors were logged.
- Several cameras: not tested (one D415).

## Linux

### What you need

- Ubuntu 22.04 / 24.04 (x86_64), or aarch64 (Jetson, Raspberry Pi 5) for `linuxaarch64`.
- `sudo apt install build-essential cmake git pkg-config libudev-dev libssl-dev`
- openFrameworks for Linux with its dependencies (`scripts/linux/<distro>/install_dependencies.sh`).

### Build librealsense

```bash
git clone https://github.com/IntelRealSense/librealsense ~/dev/librealsense
git -C ~/dev/librealsense checkout v2.59.1
./scripts/build_librealsense_linux.sh ~/dev/librealsense      # x86_64 -> lib/linux64, aarch64 -> lib/linuxaarch64
```

Then let users open the cameras without root (replug afterwards):

```bash
sudo cp ~/dev/librealsense/config/99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```

### Build the examples

`cd example-streams && make -j$(nproc) && make RunRelease` (or the projectGenerator for another
IDE). The addon links `-ludev -lpthread` (`addon_config.mk`); the makefiles pick up
`lib/linux64/librealsense2.a` themselves.

Things likely to need attention:

- **The pre-link step** (`ld -r -d --whole-archive ...` then `objcopy --keep-global-symbol='rs2_*'`)
  is the untested part. Check with `nm -g --defined-only librealsense2.a` that only `rs2_*` is
  left, and that an app links (missing symbols: add the library to the `ld -r` list; duplicate
  symbols: as on macOS, rsutils' copy of easylogging++ is dropped - there may be others).
- **Link flags:** `addon_config.mk` adds `-ludev -lpthread` (libusb uses udev). Add anything else the
  linker asks for and record it.
- **Backend:** libusb (RSUSB) by default - no kernel patches. `--v4l2` builds the kernel backend
  Intel's packages use; it needs librealsense's kernel patches for metadata. Try RSUSB first.
- **Several cameras** on Linux with libusb may need a larger USB buffer:
  `echo 1000 | sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb` (permanent: kernel
  parameter `usbcore.usbfs_memory_mb=1000`).
- Jetson / Raspberry Pi (aarch64): librealsense builds with NEON there. CUDA on Jetson
  (`BUILD_WITH_CUDA_ZEROCOPY`) is a separate project - don't enable it without measuring.

## Findings from macOS (apply everywhere)

- **Apps that grab USB devices block the camera.** Logitech G HUB's `lghub_agent` opens every USB
  device; with it running, opens failed at random with `failed to claim usb interface ... ACCESS`.
  Quit such apps. (Same problem with the Kinect v2.)
- **Update the camera's firmware** (RealSense Viewer or `rs-fw-update`) before chasing problems;
  the test D415 was on 5.12.6.
- librealsense logs `control_transfer returned error ... timed out` warnings on macOS at stream
  start and sometimes while streaming, between 2 and ~100 per 10 s in identical runs, with or
  without the addon and with no frame loss. They're hidden by default now (librealsense logs
  errors only unless `OF_LOG_VERBOSE`). Check whether Windows / Linux show them at all.
- At stream start a few frames are counted as dropped (frame counter gaps): not real loss.
- macOS needs root (libusb vs the system camera driver). Windows (Media Foundation) and Linux (udev
  rules) shouldn't - confirm.
- RealSense Viewer's "Post-Processing" toggle is librealsense's filter chain; the addon's
  spatial + temporal filters are the same filters, on by default (the viewer also decimates by 2).

## How to check a platform is done

Build all five examples from freshly generated projects, then with a camera connected:

1. `example-streams`: colour, depth, IR, colour-in-depth and the cloud live; the status line shows
   the camera, USB 3.x, librealsense 2.59.1, ~30 fps and the ms per frame.
2. `example-pointcloud`: textured cloud, the right way round (text in the scene reads correctly);
   "Point cloud faces" gives a mesh, "Colour-res point cloud" the colour-resolution grid.
3. `example-shader`: lit mesh (black mesh = normals or winding wrong).
4. `example-pointCloudTransform`: the cloud moves with the transform gizmo.
5. `example-playback`: `r` records, `r` again stops, `p` plays it back with the position moving,
   space pauses, left / right seek; a RealSense Viewer `.bag` dropped on the window plays.
6. Quit each with the window close button: clean exit, and the camera reopens straight away.

macOS reference (M3 Ultra, D415, `getCaptureStats().processMs`, all at the full frame rate):
points 7.8 ms (1280 x 720 depth, 1080p colour, spatial + temporal), depth-res mesh 11.5-13 ms,
colour-res points 12-14 ms, colour-res mesh 22-29 ms, filters off 4.3-5 ms, 848 x 480 @ 60 fps
3.2-3.6 ms. librealsense's own spatial filter alone: 9.8 ms; the addon's: 1.5-1.9 ms.

When a platform works, update its row in the main README and `scripts/README.md` ("not yet run"),
and commit the new `libs/librealsense2/lib/<platform>` files.
