# SPDX-License-Identifier: MIT
# Toolchain provenance sidecar for the gemm_rtp design sets.
#
# THE IMPLEMENTATION MOVED to tools/npu_designs.py. It used to live here, and
# open_kernels/export_qwen36_kernels.py grew its own copy that wrote the same
# FILENAME with a different SCHEMA -- three fields here, eight there, both
# called toolchain.json, both landing under src/xclbins/. src/open_npue/
# npu_device.cpp reads the file and reports "unavailable" for any field it does
# not find, so the disagreement degrades silently instead of failing. One
# schema, written by one function, is the fix; this file stays as the name
# export_gemm_rtp.py imports.
#
# What it records: the mlir_aie and Peano (llvm-aie) package versions, the git
# HEAD of the mlir-aie checkout and of this repository. Nothing here is
# interpreted by the runtime -- it only reports it.
#
# Never fails a build: an export must not die because provenance could not be
# established. A field that cannot be read is recorded as "unavailable", never
# guessed at and never a raised exception.
#
# DELIBERATE DEPARTURE from the original proposal: the packer does NOT copy a
# design's toolchain.json into the .npue container header. A container is packed
# from HuggingFace weights and never invokes mlir-aie, so a design's toolchain
# is not a property of the container -- the relationship between designs and
# containers is many-to-many in both directions. toolchain.json lives ONLY next
# to design.json, in the artifacts directory.

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from npu_designs import mlir_aie_root, write_toolchain_json as _write  # noqa: E402

MLIR_AIE_ROOT = mlir_aie_root()


def write_toolchain_json(out_dir, mlir_aie_root=None) -> dict:
    """Write toolchain.json next to design.json in `out_dir`.

    `mlir_aie_root` is accepted for compatibility and ignored: the shared module
    resolves the checkout the same way for every producer, so a build no longer
    depends on which of three documented locations the caller happened to know
    about (C:\\dev\\mlir-aie here, ~/ironenv142 in open_kernels, ironvenv +
    utilities/mlir-aie in AGENTS.md).
    """
    return _write(Path(out_dir), producer="gemm_rtp")
