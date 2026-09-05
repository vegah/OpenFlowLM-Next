r"""Build the six open kernel sets the Qwen3.6-MoE engine loads, from source.

    source ~/ironenv142/bin/activate          # mlir-aie 1.4.2 + Peano, xclbinutil/aiebu-asm on PATH
    python open_kernels/export_qwen36_kernels.py [--out DIR] [--only lx0,ln] [--no-build] [--check FILE]

The compiled kernels are NOT checked in (see .gitignore): this script is how
they come to exist. It runs build_design.py once per set with the design's
compile-time knobs, then copies final.xclbin + insts.bin into
<out>/<name>/ -- by default src/xclbins/Qwen3.6-35B-A3B-NPU2/open_kernels/,
which is where src/open_qwen36/engine.cpp looks for them (and what
`install(DIRECTORY xclbins ...)` ships in the package). toolchain.json beside
them records the mlir-aie / Peano versions, the git commit of this tree and
the sha256 of every file, so a binary can always be traced to its source.

--check DIR compares the fresh build against a previous one (a directory with
the same <name>/final.xclbin, <name>/insts.bin layout) and exits non-zero on
any real difference: the "is the source really the source?" test. insts.bin
must be byte-identical. final.xclbin is compared after masking the fields
xclbinutil stamps per build -- the axlf header's unique id, timestamp and
UUID, the PDI UUID inside the AIE_PARTITION section, and the JSON
"XCLBIN_MIRROR_DATA" tail that repeats them -- because every other byte (the
AIE core ELFs, the CDOs, the metadata sections) is expected to match. A
rebuild on the same toolchain here differed in exactly those ~80 bytes per
xclbin and nowhere else (see src/open_qwen36/README.md).
"""
from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent                 # open_kernels/
REPO = HERE.parent
DESIGNS = HERE / "designs"
DEFAULT_OUT = REPO / "src" / "xclbins" / "Qwen3.6-35B-A3B-NPU2" / "open_kernels"

# The build-side facts this producer shares with npu_offload/gemm_rtp/: the
# toolchain.json schema, the ~/.npu/cache lock (both producers build through
# that cache, and its entries are purged by content), and the xclbin comparison
# that used to live in this file. That comparison moved because "did this source
# really produce these bytes?" is the only honest test of a repository that
# ships source instead of binaries, and BOTH producers make that claim.
# Flags stay local: SETS below is still the only place the Qwen sets'
# compile-time knobs are written down.
sys.path.insert(0, str(REPO / "tools"))
from npu_designs import (  # noqa: E402
    CacheLock, artifacts_equivalent, sha256_file, write_toolchain_json,
)

# name -> (design source, build dir, compile-time environment)
# The build dirs are the ones the batch harness (model/make_decode.py) and the
# per-design make_test scripts already name, so a run of this script also
# leaves every design testable in place.
SETS = {
    "lx0":        (DESIGNS / "layer_x" / "lx.py",              DESIGNS / "layer_x" / "build_lx0",     {"LX_PART": "0"}),
    "lx1":        (DESIGNS / "layer_x" / "lx.py",              DESIGNS / "layer_x" / "build_lx1",     {"LX_PART": "1"}),
    "ax0":        (DESIGNS / "layer_x" / "ax.py",              DESIGNS / "layer_x" / "build_ax0",     {"AX_PART": "0"}),
    "ax1":        (DESIGNS / "layer_x" / "ax.py",              DESIGNS / "layer_x" / "build_ax1",     {"AX_PART": "1"}),
    "ln":         (DESIGNS / "ln" / "ln.py",                   DESIGNS / "ln" / "build",              {}),
    "lm_head_q8": (DESIGNS / "lm_head_q8" / "lm_head_q8.py",   DESIGNS / "lm_head_q8" / "build_full", {"LMHEAD_N": "248320", "LMHEAD_CORES": "8"}),
}
# knobs that must NOT leak in from the caller's shell
CLEAR = ("LX_PART", "LX_STOP", "AX_PART", "LMHEAD_N", "LMHEAD_CORES")
FILES = ("final.xclbin", "insts.bin")


def build(name: str) -> Path:
    src, out, knobs = SETS[name]
    env = {k: v for k, v in os.environ.items() if k not in CLEAR}
    env.update(knobs)
    t0 = time.time()
    knob_str = " ".join(f"{k}={v}" for k, v in knobs.items())
    print(f"[{name}] {knob_str} python build_design.py {src.relative_to(HERE).as_posix()} "
          f"{out.relative_to(HERE).as_posix()}", flush=True)
    r = subprocess.run([sys.executable, str(HERE / "build_design.py"), str(src), str(out)], env=env, cwd=str(HERE))
    if r.returncode != 0:
        sys.exit(f"[{name}] build FAILED ({r.returncode})")
    print(f"[{name}] built in {time.time() - t0:.0f}s", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help=f"destination (default {DEFAULT_OUT.relative_to(REPO).as_posix()})")
    ap.add_argument("--only", default=",".join(SETS), help="comma-separated subset of " + ",".join(SETS))
    ap.add_argument("--no-build", action="store_true", help="copy/hash the existing build dirs without rebuilding")
    ap.add_argument("--check", metavar="DIR",
                    help="a previous export (or a shipped xclbins/<model>/open_kernels dir) to compare "
                         "against; non-zero exit on any difference beyond the per-build UUID/timestamp stamps")
    ap.add_argument("--force-unlock", action="store_true",
                    help="break a stale ~/.npu/cache lock left by a crashed build")
    a = ap.parse_args()
    names = [n.strip() for n in a.only.split(",") if n.strip()]
    bad = [n for n in names if n not in SETS]
    if bad:
        sys.exit(f"unknown set(s) {bad}; choose from {list(SETS)}")

    out_root = Path(a.out).resolve()
    hashes: dict[str, str] = {}
    # One build at a time through the shared IRON cache. --no-build only copies
    # and hashes what is already there, so it needs no lock and must not take
    # one -- it is the operation you run while a build is going.
    with CacheLock(a.force_unlock, what="export_qwen36_kernels.py") if not a.no_build \
            else contextlib.nullcontext():
        for n in names:
            bdir = SETS[n][1] if a.no_build else build(n)
            dst = out_root / n
            dst.mkdir(parents=True, exist_ok=True)
            for f in FILES:
                srcf = bdir / f
                if not srcf.is_file():
                    sys.exit(f"[{n}] {srcf} missing after build")
                shutil.copyfile(srcf, dst / f)
                key = f"{n}/{f}"
                hashes[key] = sha256_file(dst / f)
                print(f"  {hashes[key]}  {key}  ({(dst / f).stat().st_size} B)")

    # One toolchain.json schema for every design set in the repo -- see
    # tools/npu_designs.py. `merge` keeps what an earlier run recorded when this
    # one built a subset, or ran outside the toolchain venv and would otherwise
    # overwrite real versions with "unavailable".
    info = write_toolchain_json(
        out_root, producer="open_kernels", model="Qwen3.6-35B-A3B-NPU2",
        sets={n: {"design": SETS[n][0].relative_to(REPO).as_posix(),
                  "env": SETS[n][2]} for n in names},
        sha256=hashes,
        merge=(a.no_build or len(names) < len(SETS)))
    print(f"-> {out_root}  (toolchain.json: mlir-aie {info['mlir_aie_version']}, Peano {info['peano_version']})")

    if a.check:
        ref = Path(a.check)
        bad = 0
        for key in hashes:
            mine, theirs = out_root / key, ref / key
            if not theirs.is_file():
                print(f"MISSING  {key}: not in {ref}")
                bad += 1
                continue
            ok, note = artifacts_equivalent(mine, theirs)
            print(f"{'same    ' if ok else 'MISMATCH'} {key}: {note}")
            bad += not ok
        if bad:
            print(f"CHECK FAILED: {bad} of {len(hashes)} files differ from {ref}")
            return 1
        print(f"CHECK OK: all {len(hashes)} files match {ref} (instruction streams byte-identical, "
              f"xclbins identical apart from build stamps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
