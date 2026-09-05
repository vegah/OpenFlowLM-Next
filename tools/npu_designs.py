#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
r"""Shared build-side facts for every NPU design set in this repository.

WHY THIS EXISTS. Two independent producers build xclbins here:

    npu_offload/gemm_rtp/   -> src/xclbins/BERT-h*/gemm_rtp/
    open_kernels/           -> src/xclbins/Qwen3.6-35B-A3B-NPU2/open_kernels/

They arrived through different pull requests and each invented the same
conventions separately -- build from source, do not check the binaries in, drop
a toolchain.json beside them, offer a --check that proves the source really is
the source. Independently invented conventions drift, and three had already
drifted apart by the time this file was written:

  * `toolchain.json` names the same file in both trees with DIFFERENT schemas.
    gemm_rtp writes three fields (mlir_aie_version, peano_version,
    mlir_aie_git_head); open_kernels writes eight. src/open_npue/npu_device.cpp
    reads the file and reports "unavailable" for anything missing, so a set
    built by the wrong producer degrades silently rather than failing.
  * The IRON build cache (~/.npu/cache) is GLOBAL and gemm_rtp's purge() deletes
    entries by content markers. npu_offload/gemm_rtp/build.ps1 states in its own
    header that "export_gemm_rtp.py holds a lock now and refuses in under a
    second" -- and in this tree it does not: the lock was added upstream after
    the copy was taken. Two producers now share that cache, so the guard matters
    more than it did when it went missing.
  * mlir-aie is looked for in three places by three files: C:\dev\mlir-aie
    (toolchain_provenance.py), ~/ironenv142 (open_kernels), ironvenv +
    utilities/mlir-aie (AGENTS.md). A from-scratch build follows whichever
    document it found first.

So the rule is: anything two producers must agree about lives here, and nowhere
else. Flags do NOT live here -- npu_offload/gemm_rtp/families.json and
open_kernels/export_qwen36_kernels.py's SETS remain the only places their own
knobs are written down. This module is the contract between them, not a third
copy of their content.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CACHE = Path.home() / ".npu" / "cache"
XCLBINS = REPO / "src" / "xclbins"

# Bumped only when a field changes meaning. Readers (npu_device.cpp) treat an
# absent field as "unavailable", so ADDING one is safe and renaming one is not.
SCHEMA = "npu-design-set/1"

ARTIFACTS = ("final.xclbin", "insts.bin")


# --------------------------------------------------------------------------
# toolchain: where it is, what version, one answer for both producers
# --------------------------------------------------------------------------

def mlir_aie_root() -> Path:
    """Where the mlir-aie checkout lives, in the order a build should try.

    Three files each hardcoded a different answer. This resolves the same way
    everywhere and, crucially, RETURNS SOMETHING even when nothing exists, so
    provenance degrades to "unavailable" instead of raising in an export."""
    env = os.environ.get("MLIR_AIE_ROOT")
    candidates = [Path(env)] if env else []
    candidates += [
        REPO / "utilities" / "mlir-aie",   # AGENTS.md / this repo's submodule
        Path.home() / "mlir-aie",          # open_kernels' WSL layout
        Path(r"C:\dev\mlir-aie"),          # the Windows layout gemm_rtp was written on
    ]
    for c in candidates:
        if (c / ".git").exists() or (c / "utils").is_dir():
            return c
    return candidates[0]


def pkg_version(name: str) -> str:
    try:
        import importlib.metadata as m
        return m.version(name)
    except Exception:
        return "unavailable"


def git_head(root: Path) -> str:
    try:
        r = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or "unavailable"
    except Exception:
        return "unavailable"


def required_pins() -> dict:
    """The versions ironvenv-requirements.txt pins, as {package: version}.

    One pinned toolchain for the whole repository is the point: PR #6 moved it
    to mlir_aie 1.4.2 without touching npu_offload/gemm_rtp/, so the BERT sets'
    documented build now runs on a toolchain nothing had rebuilt them against.
    `build_designs.py doctor` compares this against what is installed."""
    pins = {}
    req = REPO / "ironvenv-requirements.txt"
    if not req.is_file():
        return pins
    for line in req.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        m = re.match(r"^([A-Za-z0-9_.\-]+)\s*==\s*([^\s;]+)$", line)
        if m:
            pins[m.group(1).replace("-", "_")] = m.group(2)
    return pins


def set_current_device() -> None:
    """Pin IRON to npu2 BEFORE any kernel is created.

    Without it _detect_arch() falls back to aie2 (NPU1) with no error and no
    change to what iron.get_current_device() reports: bf16 mac_dims become
    (4,8,4) instead of (4,8,8), emulate_bf16_mmul_with_bfp16 turns into a no-op
    worth 5.5x, and the maximum shim DMA burst halves from 512 B to 256 B.
    n_cols=None matters too -- from_name("npu2") alone defaults to ONE column."""
    import aie.iron as iron
    from aie.iron.device import from_name
    iron.set_current_device(from_name("npu2", n_cols=None))


# --------------------------------------------------------------------------
# the shared build cache, and the lock that keeps two producers out of it
# --------------------------------------------------------------------------

class CacheLock:
    """Refuse to start while another export owns ~/.npu/cache.

    gemm_rtp's purge() rmtree's every cache entry whose markers match, and the
    markers are CONTENT -- M*K, K*N, M*N, the dtypes. Two builds can therefore
    own the same marker set, and they do: qkv and attn_out depend on neither
    --gated-ffn nor --intermediate, so BERT-h768-bfp16 and BERT-h768-gated-bfp16
    share 8 of their 16 entries exactly. Running them together is not a race
    that needs bad luck; it is a guaranteed collision, and it surfaces minutes
    later as a FileNotFoundError on a cache hash that names nothing.

    It is in the shared module rather than in one producer because open_kernels
    builds through the same cache. A lock only export_gemm_rtp.py takes does not
    protect a gemm_rtp build from a concurrent Qwen build.

    It is re-entrant ACROSS PROCESSES through NPU_DESIGN_BUILD_LOCK, which
    build_designs.py sets for its children. Without that, the one entry command
    holding the lock and then invoking a producer that also takes it would
    deadlock against itself -- the failure a naive shared lock introduces the
    day someone wires the second caller up."""

    ENV = "NPU_DESIGN_BUILD_LOCK"

    def __init__(self, force: bool = False, what: str = ""):
        self.dir = CACHE / ".export.lock"
        self.force = force
        self.what = what
        self.held = False
        self.inherited = False

    def __enter__(self):
        if os.environ.get(self.ENV):
            self.inherited = True
            return self
        CACHE.mkdir(parents=True, exist_ok=True)
        if self.force and self.dir.exists():
            shutil.rmtree(self.dir, ignore_errors=True)
        try:
            self.dir.mkdir()                      # atomic on Windows and POSIX
        except FileExistsError:
            who = ""
            try:
                who = (self.dir / "owner").read_text(encoding="utf-8").strip()
            except OSError:
                pass
            raise SystemExit(
                "another NPU design build already owns " + str(CACHE)
                + (" (" + who + ")" if who else "")
                + "\n\nRun them ONE AT A TIME. The IRON build cache is shared"
                  " and entries are purged by\ncontent markers, so two builds"
                  " delete each other's work -- guaranteed, not unlucky.\n"
                  "If nothing is running, the lock is stale: pass"
                  " --force-unlock.")
        (self.dir / "owner").write_text(
            "pid {}, {}, started {:%Y-%m-%d %H:%M:%S}".format(
                os.getpid(), self.what or "npu design build",
                datetime.datetime.now()),
            encoding="utf-8")
        self.held = True
        return self

    def __exit__(self, *exc):
        if self.held:
            shutil.rmtree(self.dir, ignore_errors=True)
            self.held = False
        return False

    @staticmethod
    def owner():
        try:
            return ((CACHE / ".export.lock" / "owner")
                    .read_text(encoding="utf-8").strip())
        except OSError:
            return None


# --------------------------------------------------------------------------
# toolchain.json -- ONE schema, written by both producers
# --------------------------------------------------------------------------

def write_toolchain_json(out_dir, producer, model=None, sets=None,
                         sha256=None, merge=False) -> dict:
    """Write toolchain.json into out_dir. Never raises: an export must not die
    because provenance could not be established, so a field that cannot be read
    is recorded as "unavailable" -- never guessed, never omitted.

    `merge` keeps what an earlier run recorded when this one built a subset (or
    ran outside the toolchain venv and would otherwise overwrite real versions
    with "unavailable")."""
    info = {
        "schema": SCHEMA,
        "producer": producer,
        "built": datetime.datetime.now().astimezone().strftime(
            "%Y-%m-%dT%H:%M:%S%z"),
        "mlir_aie_version": pkg_version("mlir_aie"),
        "peano_version": pkg_version("llvm-aie"),
        "mlir_aie_root": str(mlir_aie_root()),
        "mlir_aie_git_head": git_head(mlir_aie_root()),
        "source_git_head": git_head(REPO),
    }
    if model:
        info["model"] = model
    if sets is not None:
        info["sets"] = sets
    if sha256 is not None:
        info["sha256"] = sha256

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tj = out_dir / "toolchain.json"
    if merge and tj.is_file():
        try:
            old = json.loads(tj.read_text(encoding="utf-8"))
            for k in ("sets", "sha256"):
                if k in old:
                    merged = dict(old[k])
                    merged.update(info.get(k) or {})
                    info[k] = merged
            for k in ("built", "mlir_aie_version", "peano_version",
                      "mlir_aie_git_head", "source_git_head"):
                if info[k] == "unavailable":
                    info[k] = old.get(k, info[k])
        except Exception:
            pass
    tj.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return info


def read_toolchain_json(d):
    try:
        return json.loads((Path(d) / "toolchain.json").read_text(encoding="utf-8"))
    except Exception:
        return None


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# "is the source really the source?" -- xclbin comparison
# --------------------------------------------------------------------------

def xclbin_volatile_ranges(x: bytes) -> list:
    """Byte ranges of an xclbin that legitimately change from build to build.

    axlf layout (xrt/include/xclbin.h): m_uniqueId @296, header.m_timeStamp
    @312, header.uuid @416, m_numSections @448, section headers @456 (40 B each:
    kind u32, name[16], pad, offset u64, size u64). Kind 32 is AIE_PARTITION:
    its aie_pdi array (array_offset @+120, 96 B entries) holds a per-build UUID
    per PDI, and each PDI image is a bootgen boot image whose image header
    carries a per-build unique id (@+0x98) and the header checksum that covers
    it (@+0xCC). The bytes after the last section are xclbinutil's
    XCLBIN_MIRROR_DATA JSON, a copy of the header. Nothing else -- the CDOs, the
    AIE core ELFs, the metadata sections -- may differ.

    Written for open_kernels/export_qwen36_kernels.py --check; it lives here now
    because the same question ("did this source produce these bytes?") is the
    only honest test of a repository that ships source and not binaries, and
    both producers make that claim."""
    if x[:7] != b"xclbin2":
        return []
    ranges = [(296, 304), (312, 320), (416, 432)]
    n = struct.unpack_from("<I", x, 448)[0]
    end = 456 + n * 40
    for i in range(n):
        off = 456 + i * 40
        kind = struct.unpack_from("<I", x, off)[0]
        so, sz = struct.unpack_from("<QQ", x, off + 24)
        end = max(end, so + sz)
        if kind == 32:
            n_pdi, off_pdi = struct.unpack_from("<II", x, so + 120)
            for k in range(n_pdi):
                e = so + off_pdi + k * 96
                ranges.append((e, e + 16))                               # aie_pdi.uuid
                img_size, img_off = struct.unpack_from("<II", x, e + 16)  # aie_pdi.pdi_image
                if img_size >= 0xD0:
                    ranges.append((so + img_off + 0x98, so + img_off + 0x9C))
                    ranges.append((so + img_off + 0xCC, so + img_off + 0xD0))
    ranges.append((end, len(x)))
    return ranges


def xclbin_equivalent(a: bytes, b: bytes):
    """True when a and b differ only in per-build stamps (UUIDs/timestamps/ids)."""
    if len(a) != len(b):
        return False, "sizes differ ({} vs {} B)".format(len(a), len(b))
    ranges = xclbin_volatile_ranges(a)
    diff = [i for i in range(len(a)) if a[i] != b[i]]
    if not diff:
        return True, "byte-identical"
    stray = [i for i in diff if not any(lo <= i < hi for lo, hi in ranges)]
    if stray:
        return False, ("{} bytes differ outside the build-stamp fields "
                       "(first @{})".format(len(stray), stray[0]))
    return True, "{} bytes differ, all build stamps".format(len(diff))


def artifacts_equivalent(mine, theirs):
    """Compare one built artifact against a reference. insts.bin must be
    byte-identical; an xclbin may differ only in the fields xclbinutil and
    bootgen stamp per build."""
    x, y = Path(mine).read_bytes(), Path(theirs).read_bytes()
    if Path(mine).suffix == ".xclbin":
        return xclbin_equivalent(x, y)
    if x == y:
        return True, "byte-identical"
    if len(x) != len(y):
        return False, "sizes differ ({} vs {} B)".format(len(x), len(y))
    n = sum(1 for a, b in zip(x, y) if a != b)
    return False, "{} of {} bytes differ".format(n, len(x))


# --------------------------------------------------------------------------
# the registry: every design set in the repo, from each producer's OWN spec
# --------------------------------------------------------------------------

@dataclass
class DesignSet:
    producer: str
    name: str
    dest: Path
    serves: list = field(default_factory=list)
    note: str = ""

    @property
    def built(self) -> bool:
        d = self.dest
        if self.producer == "gemm_rtp":
            return (d / "design.json").is_file()
        return all((d / f).is_file() for f in ARTIFACTS)


def gemm_rtp_spec() -> dict:
    return json.loads((REPO / "npu_offload" / "gemm_rtp" / "families.json")
                      .read_text(encoding="utf-8"))


def gemm_rtp_sets(xclbins=None) -> list:
    root = Path(xclbins or XCLBINS)
    return [DesignSet("gemm_rtp", f["name"], root / f["name"] / "gemm_rtp",
                      list(f.get("serves") or []), f.get("note", ""))
            for f in gemm_rtp_spec()["families"]]


def open_kernels_sets(xclbins=None) -> list:
    """Read open_kernels' own SETS table rather than restating it."""
    sys.path.insert(0, str(REPO / "open_kernels"))
    try:
        import export_qwen36_kernels as q
        names = list(q.SETS)
    except Exception:
        return []
    finally:
        sys.path.pop(0)
    root = Path(xclbins or XCLBINS) / "Qwen3.6-35B-A3B-NPU2" / "open_kernels"
    return [DesignSet("open_kernels", n, root / n, ["qwen3.6-moe:35b-a3b"], "")
            for n in names]


def all_sets(xclbins=None) -> list:
    return gemm_rtp_sets(xclbins) + open_kernels_sets(xclbins)
