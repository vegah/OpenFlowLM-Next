#!/usr/bin/env sh
# Build the flm executable on Linux.
#
#   ./build_executable.sh                 configure + build (preset linux-default)
#   ./build_executable.sh --preset linux-portable
#   ./build_executable.sh --install       then install into ~/flm_exe, no sudo
#
# THE CONTRACT IS THE SAME AS build_executable.bat; THE INSIDE IS NOT, AND
# PRETENDING OTHERWISE WOULD BE A LIE. Same name, same place, same flags, same
# output directory -- but this leg is CMake + your system compiler, and the
# Windows leg is MSVC plus a vcpkg toolchain and a WiX installer. They cannot
# be one script, because the Windows build cannot run here:
#
#   * XRT's Windows import library exports 2,395 MSVC-MANGLED C++ symbols with
#     std::string in the signatures. The NPU path is C++-only -- hw_context,
#     ext::bo, elf and module have ZERO plain-C entry points -- so mingw cannot
#     link it at all. clang-cl could (it keeps the MSVC ABI), but it needs the
#     Windows SDK and MSVC CRT, which are Microsoft's to redistribute, not ours.
#   * hrx.dll is fetched as a closed prebuilt binary. Nobody can rebuild it.
#
# The xclbins are the opposite case and DO build identically everywhere:
# ./build_xclbin.sh, one implementation, both hosts.
set -eu

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
preset=linux-default
install=0
args=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --preset) preset=$2; shift 2 ;;
        --install) install=1; shift ;;
        -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
        *) args="$args $1"; shift ;;
    esac
done

for t in cmake; do
    command -v "$t" >/dev/null 2>&1 || { echo "$t is not on PATH." >&2; exit 1; }
done

cmake --preset "$preset" -S "$here/src"
cmake --build --preset "${preset%%-*}-default" --parallel ${args:-}

if [ "$install" -eq 1 ]; then
    # Replicates `cmake --install` into a user-writable prefix; no sudo.
    PRESET="$preset" "$here/src/home_install.sh" --no-build
fi
