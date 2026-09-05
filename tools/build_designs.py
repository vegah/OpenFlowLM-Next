#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
r"""Build and verify every NPU design set in this repository, from one command.

    python tools/build_designs.py doctor     # is this shell able to build at all?
    python tools/build_designs.py list       # what sets exist, and are they built?
    python tools/build_designs.py build      # build the ones that are missing
    python tools/build_designs.py check      # do the built sets match their spec?

WHY THIS EXISTS. The xclbins in src/xclbins/ are deliberately NOT checked in --
the source is the source, and a binary sitting beside the code that allegedly
produces it is a claim nobody checks. That policy only works if rebuilding is
easy, and until this file there was no single way in: npu_offload/gemm_rtp/ had
build.ps1 (Windows, PowerShell, five BERT families) and open_kernels/ had
export_qwen36_kernels.py (WSL venv, six Qwen sets), with three different
documented toolchain locations between them and AGENTS.md.

This does not replace either producer and does not restate a single one of
their flags. It finds them, runs them, holds one lock across both (they share
~/.npu/cache, and entries there are purged by content), and checks what came
out. `build.ps1` still works and now calls through here.

WHAT IT CANNOT DO. Building needs the IRON toolchain on the current
interpreter -- `doctor` says so plainly rather than failing four minutes in.
The Qwen sets additionally need xclbinutil/aiebu-asm on PATH, which is why they
are normally built in WSL.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import npu_designs as nd  # noqa: E402

GEMM_RTP = nd.REPO / "npu_offload" / "gemm_rtp"
OPEN_KERNELS = nd.REPO / "open_kernels"
PRODUCERS = ("gemm_rtp", "open_kernels")


def rel(p) -> str:
    try:
        return Path(p).resolve().relative_to(nd.REPO).as_posix()
    except Exception:
        return str(p)


# --------------------------------------------------------------------------
# doctor
# --------------------------------------------------------------------------

def cmd_doctor(args) -> int:
    """Everything a build needs, checked before the build rather than during."""
    bad = warn = 0

    def ok(label, detail=""):
        print("  ok       {:<26} {}".format(label, detail))

    def fail(label, detail=""):
        nonlocal bad
        bad += 1
        print("  MISSING  {:<26} {}".format(label, detail))

    def note(label, detail=""):
        nonlocal warn
        warn += 1
        print("  warn     {:<26} {}".format(label, detail))

    print("interpreter")
    ok("python", "{} ({})".format(
        ".".join(str(x) for x in sys.version_info[:3]), sys.executable))

    print("\nIRON toolchain")
    try:
        import aie.iron  # noqa: F401
        ok("import aie.iron")
        iron_ok = True
    except Exception as e:
        fail("import aie.iron", "{}: {}".format(type(e).__name__, e))
        print("           This shell cannot build. On Windows:")
        print("             cd C:\\dev\\mlir-aie; . .\\iron_env.ps1   "
              "# MUST be dot-sourced")
        print("           On Linux/WSL, activate the venv built from "
              "ironvenv-requirements.txt.")
        iron_ok = False

    pins = nd.required_pins()
    for pkg, want in sorted(pins.items()):
        got = nd.pkg_version(pkg)
        if got == "unavailable":
            (note if not iron_ok else fail)(pkg, "pinned {}, not installed".format(want))
        elif got.split("+")[0] != want.split("+")[0]:
            note(pkg, "pinned {}, installed {}".format(want, got))
        else:
            ok(pkg, got)
    if not pins:
        note("ironvenv-requirements.txt", "no == pins found")

    root = nd.mlir_aie_root()
    if (root / "utils").is_dir() or (root / ".git").exists():
        ok("mlir-aie checkout", "{}  (head {})".format(root, nd.git_head(root)[:12]))
    else:
        note("mlir-aie checkout", "{} does not exist -- provenance will read "
                                  "'unavailable'".format(root))

    print("\nenvironment")
    if os.environ.get("XILINX_XRT"):
        fail("XILINX_XRT unset", "it is set to {!r}; it poisons Windows builds. "
                                 "Use XRT_ROOT.".format(os.environ["XILINX_XRT"]))
    else:
        ok("XILINX_XRT unset")
    for tool in ("xclbinutil", "aiebu-asm"):
        p = shutil.which(tool)
        (ok if p else note)(tool, p or "not on PATH (needed for the Qwen sets)")

    print("\nbuild cache")
    owner = nd.CacheLock.owner()
    if owner:
        note("~/.npu/cache free", "locked by {} -- another build is running, or "
                                  "the lock is stale".format(owner))
    else:
        ok("~/.npu/cache free", str(nd.CACHE))

    print("\n{} problem(s), {} warning(s)".format(bad, warn))
    return 1 if bad else 0


# --------------------------------------------------------------------------
# list
# --------------------------------------------------------------------------

def cmd_list(args) -> int:
    sets = nd.all_sets(args.xclbins)
    width = max(len(s.name) for s in sets)
    last = None
    for s in sets:
        if s.producer != last:
            src = GEMM_RTP if s.producer == "gemm_rtp" else OPEN_KERNELS
            print("\n{}  (source: {}/)".format(s.producer, rel(src)))
            last = s.producer
        tj = nd.read_toolchain_json(s.dest if s.producer == "gemm_rtp"
                                    else s.dest.parent)
        stamp = ""
        if s.built and tj:
            stamp = "  mlir-aie {}".format(tj.get("mlir_aie_version", "?"))
        print("  {:<3} {:<{w}}  {:<38} {}{}".format(
            "[x]" if s.built else "[ ]", s.name, ", ".join(s.serves) or "-",
            rel(s.dest), stamp, w=width))
    n = sum(1 for s in sets if s.built)
    print("\n{} of {} built.  [ ] = not built here; "
          "`build_designs.py build` makes it.".format(n, len(sets)))
    return 0


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

def build_gemm_rtp(names, out_root, force) -> list:
    """Run export_gemm_rtp.py once per family, with families.json's own flags.

    This is build.ps1's loop, in Python so there is one implementation for both
    platforms. The flags come from families.json and are not repeated here --
    that file says, correctly, that it is the only place they are written."""
    spec = nd.gemm_rtp_spec()
    common = list(spec["common"])
    failed = []
    for fam in spec["families"]:
        if fam["name"] not in names:
            continue
        out = Path(out_root) / fam["name"]
        if (out / "gemm_rtp" / "design.json").is_file() and not force:
            print("  {:<24} already built (--force rebuilds)".format(fam["name"]))
            continue
        print("  {:<24} {}".format(fam["name"], ", ".join(fam.get("serves") or [])))
        if out.exists():
            shutil.rmtree(out, ignore_errors=True)
        t0 = time.time()
        r = subprocess.run(
            [sys.executable, "export_gemm_rtp.py"] + list(fam["args"]) + common
            + ["--out", str(out)],
            cwd=str(GEMM_RTP))
        dt = int(time.time() - t0)
        if r.returncode != 0:
            print("    FAILED (exit {}) after {}s".format(r.returncode, dt))
            failed.append(fam["name"])
        else:
            n = len(list((out / "gemm_rtp").glob("*"))) if (out / "gemm_rtp").is_dir() else 0
            print("    ok  {} files, {}s".format(n, dt))
    return failed


def build_open_kernels(names, out_root, force) -> list:
    dst = Path(out_root) / "Qwen3.6-35B-A3B-NPU2" / "open_kernels"
    todo = [n for n in names
            if force or not all((dst / n / f).is_file() for f in nd.ARTIFACTS)]
    if not todo:
        print("  all requested sets already built (--force rebuilds)")
        return []
    print("  {}".format(", ".join(todo)))
    r = subprocess.run(
        [sys.executable, "export_qwen36_kernels.py", "--out", str(dst),
         "--only", ",".join(todo)],
        cwd=str(OPEN_KERNELS))
    return [] if r.returncode == 0 else ["open_kernels"]


def cmd_build(args) -> int:
    out_root = Path(args.xclbins or nd.XCLBINS)
    sets = nd.all_sets(out_root)
    wanted = set(n.strip() for n in args.only.split(",") if n.strip()) if args.only else None
    if wanted:
        known = {s.name for s in sets}
        unknown = wanted - known
        if unknown:
            sys.exit("unknown set(s) {}; try `build_designs.py list`".format(sorted(unknown)))

    plan = {}
    for s in sets:
        if args.producer and s.producer != args.producer:
            continue
        if wanted and s.name not in wanted:
            continue
        plan.setdefault(s.producer, []).append(s.name)
    if not plan:
        sys.exit("nothing selected")

    try:
        import aie.iron  # noqa: F401
    except Exception:
        sys.exit("the IRON toolchain is not importable on this interpreter.\n"
                 "Run `python tools/build_designs.py doctor` for what is missing.")

    print("Building into {}".format(out_root))
    failed = []
    t0 = time.time()
    # ONE lock across both producers: they share ~/.npu/cache, where entries are
    # purged by content markers, so a Qwen build and a BERT build running
    # together delete each other's work exactly as two BERT builds would.
    with nd.CacheLock(force=args.force_unlock, what="build_designs.py"):
        # Set only AFTER the lock is held: a producer this script invokes sees
        # the variable and inherits the lock instead of deadlocking on it.
        os.environ["NPU_DESIGN_BUILD_LOCK"] = str(os.getpid())
        for producer, names in plan.items():
            print("\n{}:".format(producer))
            if producer == "gemm_rtp":
                failed += build_gemm_rtp(names, out_root, args.force)
            else:
                failed += build_open_kernels(names, out_root, args.force)
        os.environ.pop("NPU_DESIGN_BUILD_LOCK", None)

    print("\n{} failed, {:.0f}s total".format(len(failed), time.time() - t0))
    if failed:
        print("failed: {}".format(", ".join(failed)))
        return 1
    print("\nChecking what was just built:")
    return check(out_root, args.producer, require_built=True)


# --------------------------------------------------------------------------
# check
# --------------------------------------------------------------------------

def check_gemm_rtp_spec(s, problems) -> None:
    """families.json (the flags) against design.json (what got built).

    Every wrong flag here produces a perfectly valid design set, and a set is
    selected at load time by geometry AND datapath -- so a wrong flag is not a
    slower design, it is one the wrong model loads or none does. Two such
    defects have shipped, both found by accident from outside."""
    sys.path.insert(0, str(GEMM_RTP))
    try:
        import check_design_sets as cds
    finally:
        sys.path.pop(0)
    spec = nd.gemm_rtp_spec()
    fam = next(f for f in spec["families"] if f["name"] == s.name)
    want = cds.expected(list(fam["args"]) + list(spec["common"]))
    got = cds.actual(json.loads((s.dest / "design.json").read_text(encoding="utf-8")))
    for k in want:
        if want[k] is not None and want[k] != got.get(k):
            problems.append("{}: {} -- families.json says {!r}, design.json "
                            "says {!r}".format(s.name, k, want[k], got.get(k)))


def check_provenance(s, problems, warnings) -> None:
    """Every built set carries a toolchain.json this repo can read, and the
    files still hash to what that file recorded."""
    d = s.dest if s.producer == "gemm_rtp" else s.dest.parent
    tj = nd.read_toolchain_json(d)
    if tj is None:
        problems.append("{}: no readable toolchain.json in {}".format(s.name, rel(d)))
        return
    if tj.get("schema") != nd.SCHEMA:
        warnings.append("{}: toolchain.json predates schema {} (rebuild to "
                        "refresh)".format(s.name, nd.SCHEMA))
    if tj.get("mlir_aie_version", "unavailable") == "unavailable":
        warnings.append("{}: toolchain.json records no mlir-aie version".format(s.name))
    for key, want in (tj.get("sha256") or {}).items():
        if not key.startswith(s.name + "/"):
            continue
        f = d / key
        if not f.is_file():
            problems.append("{}: {} recorded in toolchain.json but missing".format(
                s.name, key))
        elif nd.sha256_file(f) != want:
            problems.append("{}: {} does not match the sha256 toolchain.json "
                            "recorded".format(s.name, key))


def check(xclbins, producer, require_built=False) -> int:
    sets = [s for s in nd.all_sets(xclbins)
            if not producer or s.producer == producer]
    problems, warnings, missing = [], [], []
    for s in sets:
        if not s.built:
            missing.append(s)
            continue
        before = len(problems)
        if s.producer == "gemm_rtp":
            check_gemm_rtp_spec(s, problems)
        check_provenance(s, problems, warnings)
        print("  {:<8} {}".format("MISMATCH" if len(problems) > before else "ok", s.name))
    for s in missing:
        print("  {:<8} {}  ({})".format(
            "MISSING" if require_built else "not built", s.name, rel(s.dest)))

    for w in warnings:
        print("\nwarn     {}".format(w))
    for p in problems:
        print("\nMISMATCH {}".format(p))

    bad = len(problems) + (len(missing) if require_built else 0)
    print("\n{} of {} set(s) built, {} problem(s), {} warning(s)".format(
        len(sets) - len(missing), len(sets), len(problems), len(warnings)))
    if problems:
        print("\nEither the set is stale -- rebuild it with `build_designs.py "
              "build --force` --\nor the spec is wrong, in which case fix it "
              "in families.json and nowhere else.")
    elif missing and not require_built:
        print("\nSets that are not built here are reported, not failed: nothing "
              "is built on a fresh\nclone, and a check that always fails is one "
              "nobody reads. --require-built makes it an error.")
    return 1 if bad else 0


def cmd_check(args) -> int:
    return check(args.xclbins, args.producer, args.require_built)


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, xclbins=True):
        if xclbins:
            p.add_argument("--xclbins", default=None,
                           help="destination root (default src/xclbins)")
        p.add_argument("--producer", choices=PRODUCERS, default=None)

    p = sub.add_parser("doctor", help="check this shell can build")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("list", help="every design set, and whether it is built")
    common(p)
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("build", help="build the sets that are missing")
    common(p)
    p.add_argument("--only", default="", help="comma-separated set names")
    p.add_argument("--force", action="store_true", help="rebuild built sets too")
    p.add_argument("--force-unlock", action="store_true",
                   help="break a stale ~/.npu/cache lock")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("check", help="built sets against their own spec")
    common(p)
    p.add_argument("--require-built", action="store_true",
                   help="treat a set that is not built as an error")
    p.set_defaults(func=cmd_check)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
