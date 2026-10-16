# Builds librealsense for ofxLibRealSense2 on Windows: x64, Release, realsense2.dll with the
# native Windows (Media Foundation) backend - no libusb, no admin rights. Installs the headers
# and realsense2.lib / realsense2.dll into libs/librealsense2.
#
# One Release DLL serves Debug and Release apps: librealsense's C++ API (rs.hpp) is header-only
# on top of its C API, so no C++ types cross the DLL boundary. It also links the C runtime
# statically, so nothing else needs shipping.
#
# CUDA is on when a CUDA Toolkit is found (-NoCuda to leave it out). librealsense uses it for
# colour format conversion (YUY2 -> RGBA of every colour frame) and its own align / point
# cloud, and checks for an NVIDIA driver at run time (LoadLibrary("nvcuda.dll")), falling back
# to its SSE code without one. The CUDA runtime is linked statically and cuBLAS / cuSPARSE,
# which librealsense links but never calls, add no import: realsense2.dll still depends on
# system DLLs only and runs on any PC.
#
# nvcc only accepts the host compilers it knows: when the newest MSVC is too new for the
# installed CUDA Toolkit (e.g. CUDA 12.1 with VS 2026's 14.5x, whose STL crashes cudafe++), the
# library is built with the newest older MSVC toolset nvcc accepts (e.g. VS 2022's 14.44, "MSVC
# v143 build tools" in the VS 2026 installer). Only the C API crosses the DLL boundary, so the app
# can use any toolset. If that toolset has no ATL (librealsense's Media Foundation backend uses
# CComPtr), the newest toolset's ATL headers are used.
#
# Keep the CMake options in sync with build_librealsense_macos.sh / _linux.sh.
#
# Requirements: Visual Studio 2022 or newer with the C++ workload (its CMake is used); the CUDA
# Toolkit (11.x / 12.x / 13.x) for the CUDA build.
#
# Usage (PowerShell):
#   .\build_librealsense_windows.ps1 -Source C:\path\to\librealsense
#   .\build_librealsense_windows.ps1 -Source C:\path\to\librealsense -NoCuda
#   .\build_librealsense_windows.ps1 -Source C:\path\to\librealsense -CudaPath "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"

param(
    [Parameter(Mandatory = $true)][string]$Source,
    [switch]$NoCuda,
    # default: $env:CUDA_PATH
    [string]$CudaPath = $env:CUDA_PATH,
    # MSVC toolset for the build, e.g. "14.44"; default: the newest one nvcc accepts
    [string]$VcVars = "",
    # librealsense's own list leaves out desktop Maxwell / Pascal (e.g. GTX 1080 Ti, sm_61)
    [string[]]$ExtraCudaArchs = @("52", "61")
)

$ErrorActionPreference = "Continue"
$Source = (Resolve-Path $Source).Path
$Addon = (Resolve-Path "$PSScriptRoot\..").Path
$build = "$Source\build_ofx"

$installer = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer"
$env:PATH = "$installer;$env:PATH" # VsDevCmd calls vswhere by name
$vs = & "$installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw "Visual Studio with the C++ workload not found" }
$cmake = "$vs\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
Write-Host "Visual Studio: $vs"

$nvcc = if ($CudaPath) { "$CudaPath\bin\nvcc.exe" } else { "" }
$cuda = -not $NoCuda
if ($cuda -and -not ($nvcc -and (Test-Path $nvcc))) {
    Write-Warning "CUDA Toolkit not found (set -CudaPath or CUDA_PATH): building without CUDA"
    $cuda = $false
}
$nvccFlags = "-allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"

# Can nvcc build with this MSVC toolset ("" = the default one)? Checked in a child shell, as
# entering a developer shell can't be undone.
function Test-NvccHost([string]$ver) {
    $dir = Join-Path ([IO.Path]::GetTempPath()) "ofxrs_nvcc_probe"
    New-Item -ItemType Directory -Force $dir | Out-Null
    Set-Content "$dir\probe.cu" "#include <vector>`n#include <string>`n__global__ void k(float* a) { a[threadIdx.x] *= 2.f; }`nint main() { std::vector<std::string> v; k<<<1, 1>>>(nullptr); return 0; }" -Encoding ascii
    $vcv = if ($ver) { " -vcvars_ver=$ver" } else { "" }
    $probe = "`$env:PATH = '$installer;' + `$env:PATH; Import-Module '$vs\Common7\Tools\Microsoft.VisualStudio.DevShell.dll'; " +
        "Enter-VsDevShell -VsInstallPath '$vs' -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64$vcv' | Out-Null; " +
        "& '$nvcc' $nvccFlags -c -o '$dir\probe.obj' '$dir\probe.cu' *> `$null; exit `$LASTEXITCODE"
    powershell -NoProfile -Command $probe | Out-Null
    $ok = $LASTEXITCODE -eq 0
    Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
    return $ok
}

if ($cuda -and -not $VcVars) {
    $candidates = @("") + (Get-ChildItem "$vs\VC\Tools\MSVC" -Directory | Sort-Object { [version]$_.Name } -Descending |
        ForEach-Object { ($_.Name -split '\.')[0..1] -join '.' } | Select-Object -Skip 1 -Unique)
    $found = $false
    foreach ($ver in $candidates) {
        if (Test-NvccHost $ver) { $VcVars = $ver; $found = $true; break }
        Write-Host "nvcc can't use MSVC $(if ($ver) { $ver } else { '(default)' })"
    }
    if (-not $found) {
        Write-Warning "nvcc ($nvcc) accepts none of the installed MSVC toolsets: building without CUDA. Install a newer CUDA Toolkit, or an older MSVC toolset from the Visual Studio installer."
        $cuda = $false
    }
}

$vcvArg = if ($VcVars) { " -vcvars_ver=$VcVars" } else { "" }
Import-Module "$vs\Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
Enter-VsDevShell -VsInstallPath $vs -SkipAutomaticLocation -DevCmdArguments "-arch=x64 -host_arch=x64$vcvArg" | Out-Null
Write-Host "MSVC: $env:VCToolsInstallDir"
if (-not (Test-Path "$env:VCToolsInstallDir\atlmfc\include\atlcomcli.h")) {
    $atl = Get-ChildItem "$vs\VC\Tools\MSVC" -Directory | Sort-Object { [version]$_.Name } -Descending |
        Where-Object { Test-Path "$($_.FullName)\atlmfc\include\atlcomcli.h" } | Select-Object -First 1
    if (-not $atl) { throw "No MSVC toolset with ATL found: install 'C++ ATL' in the Visual Studio installer" }
    Write-Host "ATL from: $($atl.FullName)"
    $env:INCLUDE = "$env:INCLUDE;$($atl.FullName)\atlmfc\include"
    $env:LIB = "$env:LIB;$($atl.FullName)\atlmfc\lib\x64"
}

$cmakeArgs = @(
    "-S", $Source, "-B", $build,
    "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DBUILD_SHARED_LIBS=ON",
    "-DBUILD_WITH_STATIC_CRT=ON",
    "-DFORCE_RSUSB_BACKEND=OFF",
    "-DBUILD_EXAMPLES=OFF",
    "-DBUILD_GRAPHICAL_EXAMPLES=OFF",
    "-DBUILD_GLSL_EXTENSIONS=OFF",
    "-DBUILD_TOOLS=OFF",
    "-DBUILD_UNIT_TESTS=OFF",
    "-DBUILD_WITH_OPENMP=OFF",
    "-DCHECK_FOR_UPDATES=OFF",
    "-DENABLE_STATS=OFF",
    "-DENABLE_CCACHE=OFF",
    "-DBUILD_WITH_DDS=OFF",
    "-DBUILD_ROSBAG2=OFF"
)
if ($cuda) {
    Write-Host "CUDA: $CudaPath"
    $gencode = ($ExtraCudaArchs | ForEach-Object { "-gencode arch=compute_$_,code=sm_$_" }) -join " "
    $cmakeArgs += @(
        "-DBUILD_WITH_CUDA=ON",
        "-DCMAKE_CUDA_COMPILER=$($nvcc -replace '\\', '/')",
        "-DCUDA_TOOLKIT_ROOT_DIR=$($CudaPath -replace '\\', '/')",
        "-DCUDA_USE_STATIC_CUDA_RUNTIME=ON",
        "-DCMAKE_CUDA_RUNTIME_LIBRARY=Static",
        "-DCMAKE_CUDA_FLAGS=$nvccFlags $gencode",
        # librealsense switches /MD to /MT for C / C++ only
        "-DCMAKE_CUDA_FLAGS_RELEASE=-Xcompiler=-MT -O3 -DNDEBUG"
    )
} else {
    $cmakeArgs += "-DBUILD_WITH_CUDA=OFF"
}

Remove-Item -Recurse -Force $build -ErrorAction SilentlyContinue
& $cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed" }
& $cmake --build $build --target realsense2 --parallel
if ($LASTEXITCODE -ne 0) { throw "build failed" }

$incOut = "$Addon\libs\librealsense2\include"
$libOut = "$Addon\libs\librealsense2\lib\vs\x64"
Remove-Item -Recurse -Force $incOut, $libOut -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $incOut, $libOut | Out-Null
Copy-Item -Recurse "$Source\include\librealsense2" "$incOut\"
$dll = Get-ChildItem -Recurse "$build" -Filter realsense2.dll | Select-Object -First 1
$lib = Get-ChildItem -Recurse "$build" -Filter realsense2.lib | Select-Object -First 1
Copy-Item $dll.FullName, $lib.FullName $libOut

$deps = dumpbin /dependents "$libOut\realsense2.dll" | Select-String "^\s+\S+\.dll\s*$" | ForEach-Object { $_.Line.Trim() }
$nonSystem = $deps | Where-Object { -not (Test-Path "$env:SystemRoot\System32\$_") }
if ($nonSystem) { Write-Warning "realsense2.dll depends on non-system DLLs: $($nonSystem -join ', ')" }

Write-Host ""
Write-Host "librealsense $(git -C $Source describe --tags --always) installed into $Addon\libs\librealsense2 (vs x64$(if ($cuda) { ", CUDA $(Split-Path $CudaPath -Leaf), MSVC $((Get-Command cl).Source -replace '.*MSVC\\([^\\]+).*', '$1')" }))"
Get-ChildItem $libOut | ForEach-Object { Write-Host "  $($_.Name)  $([math]::Round($_.Length / 1MB, 1)) MB" }
