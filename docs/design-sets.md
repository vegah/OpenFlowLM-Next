# NPU design sets: one convention, two producers

The `.xclbin` files this application loads are **built, not checked in**. The
source is the source, and a binary sitting in a repository beside the code that
allegedly produces it is a claim nobody checks. That policy only pays for itself
if rebuilding is easy — so this document is the contract, and
`tools/build_designs.py` is the one command that honours it.

```
python tools/build_designs.py doctor    # can this shell build at all?
python tools/build_designs.py list      # what sets exist, and are they built?
python tools/build_designs.py build     # build the ones that are missing
python tools/build_designs.py check     # do the built sets match their spec?
```

## The two producers

| producer | source | builds | destination |
|---|---|---|---|
| `gemm_rtp` | `npu_offload/gemm_rtp/` | 5 BERT/embedding GEMM families | `src/xclbins/BERT-*/gemm_rtp/` |
| `open_kernels` | `open_kernels/` | 6 Qwen3.6-MoE decode sets | `src/xclbins/Qwen3.6-35B-A3B-NPU2/open_kernels/` |

They are genuinely different things — RTP-parameterised GEMM families against
21 fused decode designs — and they are **not** being merged. What is shared is
the contract below, which lives in `tools/npu_designs.py` and nowhere else.

## The output convention

```
src/xclbins/<model-or-family>/<producer>/
    toolchain.json          <- provenance, one schema (see below)
    ...                     <- the producer's own artifacts
```

`gemm_rtp` writes `design.json` plus its instruction streams into that
directory; `open_kernels` writes one subdirectory per set, each holding
`final.xclbin` and `insts.bin`, with `toolchain.json` at the producer level.

**A set is identified by its contents, never by its modification time.** A JIT
cache hit does not restamp a directory, so "newest wins" silently returns a
design nobody asked for.

## `toolchain.json` — one schema

`src/open_npue/npu_device.cpp` reads this file and reports `unavailable` for
any field it does not find, so a missing field degrades **silently**. Adding a
field is safe; renaming one is not.

| field | meaning |
|---|---|
| `schema` | `npu-design-set/1` |
| `producer` | `gemm_rtp` or `open_kernels` |
| `built` | local ISO-8601 timestamp |
| `mlir_aie_version`, `peano_version` | installed package versions |
| `mlir_aie_root`, `mlir_aie_git_head` | which checkout, at which commit |
| `source_git_head` | this repository's HEAD at build time |
| `model`, `sets`, `sha256` | optional; `open_kernels` records all three |

Nothing here is interpreted by the runtime — it only reports it. A field that
cannot be read is written as `"unavailable"`, never guessed and never omitted,
because an export must not fail over provenance.

## The toolchain

**One pin, for the whole repository: `ironvenv-requirements.txt`.** It is the
only place mlir-aie and Peano versions are written down, and
`build_designs.py doctor` compares it against what is installed.

This matters more than it looks. PR #6 moved the pin to `mlir_aie==1.4.2`
without touching `npu_offload/gemm_rtp/`, so the BERT sets' documented build
began running on a toolchain nothing had rebuilt them against — no error, no
warning, and no way to tell from the artifacts, because the old
`toolchain.json` recorded only three fields and none of them was checked.

The mlir-aie checkout is resolved in one place, in this order:
`$MLIR_AIE_ROOT`, `utilities/mlir-aie`, `~/mlir-aie`, `C:\dev\mlir-aie`. Three
files used to hardcode three different answers, and a from-scratch build
followed whichever document its reader found first.

### From scratch in WSL, with no root

Verified on a clean Ubuntu 24.04 that had only an old mlir-aie 1.3.4 venv —
which `doctor` correctly refused, because 1.3.4's runtime API cannot build the
current designs at all (`rt.task_group()` vs `TaskGroup()`, `rt.fill(fifo, …)`
vs `fifo.fill(…)`).

```bash
python3 -m venv ~/ironenv142
~/ironenv142/bin/pip install -r ironvenv-requirements.txt      # mlir-aie 1.4.2 + Peano

# xclbinutil: mlir-aie ships the source, deliberately self-contained
cmake -B ~/hrx-build -S ~/mlir-aie/third_party/hrx-xclbinutil -DCMAKE_BUILD_TYPE=Release
cmake --build ~/hrx-build --target hrx-xclbinutil -j$(nproc)
mkdir -p ~/xrt-tools/bin
ln -sf ~/hrx-build/tools/hrx-xclbinutil ~/xrt-tools/bin/xclbinutil

export PATH=~/xrt-tools/bin:$PATH
source ~/ironenv142/bin/activate
python tools/build_designs.py doctor
python tools/build_designs.py build --producer open_kernels
```

`pip`, `cmake`, `g++`. **No XRT install, no Boost, no `sudo`.**

### `aiebu-asm` and `insts.elf`

The one tool that will *not* come from a checkout is `aiebu-asm`: Xilinx/aiebu
needs Boost, liblzma and liblz4, i.e. a package manager and an administrator.
It is used by exactly one aiecc step — the last one, emitting `insts.elf` — and
before this was made conditional, its absence failed the **whole** pipeline
after the AIE compile and `insts.bin` were already finished.

That step is not on the path to a shipped kernel. `export_qwen36_kernels.py`
copies `final.xclbin` and `insts.bin`, and nothing else; the ELF is for the C++
harness (`xrt::elf(insts.elf)`), a different workflow. So `build_design.py`
skips it when the tool is missing and **says so loudly**, `doctor` reports it as
a warning naming exactly what is degraded, and a stale `insts.elf` is deleted
rather than left to look current.

`xclbinutil`, by contrast, is a hard requirement — `doctor` reports its absence
as a problem, not a warning, because aiecc cannot package an xclbin without it.

## Why builds are serialised — one lock, two different collisions

Do not simplify this to "the shared cache". The two producers destroy a
concurrent build by different mechanisms, and only one of them involves the
cache at all.

**`gemm_rtp` collides in the IRON build cache.** It is global (`~/.npu/cache`)
and `purge()` deletes entries by **content markers** — `M*K`, `K*N`, `M*N`, the
dtypes. Two builds can own the same markers, and they do: `qkv` and `attn_out`
depend on neither `--gated-ffn` nor `--intermediate`, so `BERT-h768-bfp16` and
`BERT-h768-gated-bfp16` share 8 of their 16 entries exactly. Not a race that
needs bad luck — a guaranteed collision, surfacing minutes later as a
`FileNotFoundError` on a cache hash that names nothing.

**`open_kernels` never touches that cache.** Measured: six sets, 42 minutes,
**zero** entries left in `~/.npu/cache`, against 125 from a single `gemm_rtp`
run — `build_design.py` compiles with explicit output paths, and explicit paths
bypass the JIT cache. It collides somewhere else entirely: `SETS` names **fixed
build directories inside the working tree** (`designs/layer_x/build_lx0` and
friends), which are the same paths whatever `--out` says. That also means a
Windows build and a WSL build of the same set collide even though they have
different `$HOME`s and therefore different locks.

`build_designs.py` holds one lock across both producers for both reasons. The
lock directory stays inside `~/.npu/cache` anyway, deliberately: that is the
path upstream NpuEmbeddings' own `export_gemm_rtp.py` locks, so when a future
sync brings that copy in, the two meet on the same file instead of each
guarding a different door. `--force-unlock` clears a lock left by a crash.

> **One gap, stated rather than hidden.** `npu_offload/gemm_rtp/` is a *dumb
> copy* synced from the upstream NpuEmbeddings repository, so it is not patched
> here — patching a sync target is how the copy stops being a copy. Its
> `export_gemm_rtp.py` therefore takes no lock of its own in this tree, and
> invoking it **directly** is unguarded. Go through `build_designs.py` (or
> `build.ps1`, which now wraps it). Upstream's own copy does take a lock; when a
> future sync brings it in, the two will meet, and the shared lock is
> re-entrant across processes via `NPU_DESIGN_BUILD_LOCK` for that reason.

## Where the flags live

**Not here, and not in the shared module.** Each producer keeps its own:

* `npu_offload/gemm_rtp/families.json` — the five families' flags, with the
  reason each one is what it is. Every flag is load-bearing: a set is selected
  at load time by geometry **and** datapath, so a wrong flag is not a slower
  design, it is one the wrong model loads, or none does. Two such defects have
  shipped, both found by accident from outside.
* `open_kernels/export_qwen36_kernels.py`'s `SETS` — the six Qwen sets and
  their compile-time environment knobs.

`build_designs.py` reads both and restates neither.

## What `check` proves

* **`gemm_rtp`** — `families.json` against the built `design.json`, field by
  field. This catches a flag edited without a rebuild, and a set built by some
  other route.
* **both** — every built set carries a readable `toolchain.json` of the current
  schema, and where hashes were recorded, the files still match them.
* **`open_kernels --check DIR`** additionally answers *"did this source really
  produce these bytes?"*: `insts.bin` must be byte-identical, and an
  `.xclbin` may differ only in the fields `xclbinutil` and `bootgen` stamp per
  build (the axlf unique id, timestamp and UUID, the PDI UUIDs, the boot-image
  header id and its checksum, and the `XCLBIN_MIRROR_DATA` tail). Roughly 80
  bytes per xclbin; every other byte must match.

A set that is **not built** is reported, not failed — nothing is built on a
fresh clone, and a check that always fails is a check nobody reads.
`--require-built` makes it an error, which is what the build path passes for
what it has just built.
