r"""Build an IRON design into xclbin + insts.bin + insts.elf (no NPU needed).

Runs in the WSL `ironenv` (mlir-aie wheel + Peano); the artifacts are then
loaded on Windows by phlegm's XRT shim exactly like FLM's kernels are:
xrt::elf(insts.elf) -> xrt::module -> xrt::ext::kernel(ctx, "MLIR_AIE"),
run(opcode=3, 0, 0, bo...).

    source ~/ironenv/bin/activate
    python build_design.py designs/rot13/rot13.py [out_dir]

The design module must expose DESIGN (an @iron.jit callable) and SPECIALIZE
(dict of CompileTime kwargs).

insts.elf is OPTIONAL and is skipped, loudly, when `aiebu-asm` is not
available. It is aiecc's last pipeline step (26 of 38 on the ln design) and it
is the only step that needs that tool -- so requiring it made the build of the
kernels the ENGINE loads depend on a tool whose output the engine never reads:
export_qwen36_kernels.py ships final.xclbin and insts.bin, and nothing else.
The ELF is for the C++ harness (`xrt::elf(insts.elf)`), which is a different
workflow with a different audience.

That distinction is worth a whole environment. With the ELF skipped, a
from-scratch WSL clone builds every set with pip, cmake and g++: mlir-aie and
Peano come from ironvenv-requirements.txt, and xclbinutil from
mlir-aie's own third_party/hrx-xclbinutil, which is deliberately self-contained
("No Boost, no system XRT, no submodule"). aiebu-asm is the one piece that is
not: Xilinx/aiebu needs Boost, liblzma and liblz4, i.e. root on the machine.
Making the ELF conditional is the difference between "needs a package manager
and an administrator" and "needs a checkout".
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from npu_designs import find_aiebu_asm, set_current_device  # noqa: E402


def main() -> int:
    src = Path(sys.argv[1]).resolve()
    out = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else src.parent / "build"
    # Always a fresh project: aiecc keeps copies of the kernel sources and their
    # objects in final.prj and skips recompiling them (header edits in the
    # design dir were silently ignored: identical results build after build).
    shutil.rmtree(out / "final.prj", ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)

    # Without this IRON silently targets aie2 / NPU1 -- see tools/npu_designs.py
    # for what that costs and why it is invisible.
    set_current_device()

    spec = importlib.util.spec_from_file_location(src.stem, src)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(src.parent))
    spec.loader.exec_module(mod)

    # aiecc emits insts.elf via aiebu-asm and fails the WHOLE pipeline when the
    # tool is absent -- after the AIE compile and insts.bin are already done.
    # Skip that one step rather than lose the build; say so, because the harness
    # does need the ELF and a silent omission would surface as a missing file
    # much later, somewhere else.
    aiebu = find_aiebu_asm()
    kwargs = dict(xclbin_path=str(out / "final.xclbin"),
                  inst_path=str(out / "insts.bin"))
    if aiebu:
        kwargs["elf_path"] = str(out / "insts.elf")
    else:
        (out / "insts.elf").unlink(missing_ok=True)   # never leave a stale one
        print("  WARNING: aiebu-asm not found -- building WITHOUT insts.elf.\n"
              "           final.xclbin and insts.bin are complete, which is all\n"
              "           export_qwen36_kernels.py ships and all the engine\n"
              "           loads. The C++ harness (xrt::elf(insts.elf)) will NOT\n"
              "           run against this build. Install XRT, or Xilinx/aiebu\n"
              "           (needs Boost, liblzma, liblz4), and rebuild.",
              flush=True)

    t0 = time.time()
    xclbin, insts = mod.DESIGN.specialize(**mod.SPECIALIZE).compile(**kwargs)
    dt = time.time() - t0
    for p in sorted(out.iterdir()):
        print(f"  {p.name:24s} {p.stat().st_size:>10d} B")
    print(f"BUILD_OK {src.name} -> {out}  ({dt:.1f}s)"
          + ("" if aiebu else "  [no insts.elf]"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
