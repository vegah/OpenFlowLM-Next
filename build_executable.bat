@echo off
setlocal
::
:: Build flm.exe on Windows.
::
::     build_executable.bat              configure + build via vcpkg + MSVC
::     build_executable.bat <vcpkg-root> use a vcpkg other than C:\dev\vcpkg
::
:: THE CONTRACT IS THE SAME AS build_executable.sh; THE INSIDE IS NOT, AND
:: PRETENDING OTHERWISE WOULD BE A LIE. Same name, same place, same output --
:: but this leg is MSVC plus a vcpkg toolchain, and the Linux leg is CMake plus
:: the system compiler. They cannot be one script, because this build cannot
:: run there: XRT's Windows import library exports 2,395 MSVC-mangled C++
:: symbols with std::string in the signatures, the NPU path (hw_context,
:: ext::bo, elf, module) has ZERO plain-C entry points, and hrx.dll is a closed
:: prebuilt nobody can rebuild.
::
:: The xclbins are the opposite case and DO build identically everywhere:
:: build_xclbin.bat, one implementation, both hosts.
::
:: This forwards to src\build-windows-vcpkg.cmd, which is the path that worked
:: on a bare box: no standalone Boost b2 build, no hand-copied import libs.
:: One-time setup is documented in that file's header. If a configure has
:: already failed once in this tree, use clean_build.bat instead -- CMake does
:: NOT re-apply a toolchain file to an existing cache, so a half-configured
:: build directory silently links a different Boost forever.

if /I "%~1"=="--help" goto :usage
if /I "%~1"=="-h"     goto :usage

where cl >nul 2>&1
if errorlevel 1 (
    echo No 'cl' on this shell's path -- this needs the MSVC environment.
    echo Open "x64 Native Tools Command Prompt for VS" and re-run.
    echo A script that re-derives that environment is liability, not convenience.
    exit /b 1
)

call "%~dp0src\build-windows-vcpkg.cmd" %*
exit /b %ERRORLEVEL%

:usage
echo Build flm.exe on Windows (MSVC + vcpkg).
echo.
echo     build_executable.bat              configure + build
echo     build_executable.bat ^<vcpkg-root^> use a vcpkg other than C:\dev\vcpkg
echo.
echo For the NPU design sets instead: build_xclbin.bat
exit /b 0
