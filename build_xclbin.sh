#!/usr/bin/env sh
# Build the NPU design sets (xclbins). Same command, same behaviour, on every
# host -- this file and build_xclbin.bat are FORWARDERS, not two implementations.
#
#   ./build_xclbin.sh                 build whatever is not built yet
#   ./build_xclbin.sh doctor          can this shell build at all?
#   ./build_xclbin.sh list            what sets exist, and are they built?
#   ./build_xclbin.sh check           do the built sets match their spec?
#   ./build_xclbin.sh build --force   rebuild everything
#
# Everything real is in tools/build_designs.py, which drives BOTH producers
# (npu_offload/gemm_rtp/ and open_kernels/) and holds one lock across them.
# Nothing here is OS-specific: an xclbin is a device artifact -- AIE core ELFs,
# CDOs, a PDI -- with no host code in it at all. The BERT families were built
# on Windows and the Qwen sets in WSL through this same path.
#
# Needs the IRON toolchain on the interpreter it finds. `doctor` says what is
# missing instead of failing four minutes in; docs/design-sets.md has the
# from-scratch recipe.
set -eu

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON=${PYTHON:-}
if [ -z "$PYTHON" ]; then
    for c in python3 python; do
        if command -v "$c" >/dev/null 2>&1; then PYTHON=$c; break; fi
    done
fi
if [ -z "$PYTHON" ]; then
    echo "No python on PATH. Activate the IRON venv first:" >&2
    echo "    . <venv>/bin/activate     # built from ironvenv-requirements.txt" >&2
    echo "See docs/design-sets.md." >&2
    exit 1
fi

# No arguments means the thing you came here for.
if [ "$#" -eq 0 ]; then set -- build; fi

exec "$PYTHON" "$here/tools/build_designs.py" "$@"
