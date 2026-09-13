# user-directory: where the application looks for registries, models and kernels

Prefix `USERDIR`. Home repo: openflowlm-next. Covers the lookups in `src/common/utils.cpp`
(`user_directories`, `find_model_lists`, `models_directories`, `find_xclbin_path_for`,
`find_model_info`), the registry merge in `src/include/model_registry.hpp`, how
`src/include/model_list.hpp` resolves a model's directory, and what `oflm-add` and
`q4nx-build` write and print (`utilities/oflm-add/oflm_add/__init__.py`,
`utilities/q4nx-build/q4nx/deploy.py`). Issue: #30.

Tests: `src/common/user_dirs_test.cpp` (ctest `user_dirs`; no device, no weights, no network),
`python -m pytest utilities/oflm-add/tests/test_user_registry.py utilities/q4nx-build/tests/test_deploy_registry.py`.

Every defect this file guards against is a **well-formed wrong answer**. A registry nobody
reads shows fewer models. A model looked up in the wrong directory shows as not downloaded
and is pulled again. A kernel root chosen per directory rather than per model serves
nothing for the model that was asked for. None of them crashes, which is why the
acceptance criteria below name paths and counts rather than "it runs".

There are two kinds of lookup, and they want different answers:

- **One thing per name** -- a model's weights, `xclbins/<name>`, a design family. The
  first directory that holds *that name* wins. A directory that merely exists must not
  win, or one model installed into a new directory hides every model in the old one.
- **A file that is many entries** -- `model_list.json`. First-wins would lose a user's
  whole registry the moment a newer directory gained one, so these files are merged.

## Requirements

### USERDIR-ORDER: the user directories, newest name first
**Applies to:** openflowlm-next (`src/common/utils.cpp`)
**Test category:** unit
**Tests:** `src/common/user_dirs_test.cpp` (`user_directories`), `utilities/oflm-add/tests/test_user_registry.py`

`utils::user_directories()` is the one list of user-level directories, and every lookup
below takes its user directories from it. On Windows, with `<user>` the profile directory:
`<user>\.oflm`, `<user>\.config\oflm`, `<user>\.flm`, `<user>\.config\flm`. On POSIX, with
`<user>` = `$HOME/.config`: `<user>/oflm`, `<user>/flm`. The `.config` form exists on
Windows because `oflm-add` writes to `Path.home()/.config/oflm` on every platform; the
`flm` forms because an install that predates the rename must keep working without moving
gigabytes of weights.

**Acceptance criteria:**
- The list has exactly those entries in exactly that order, unfiltered by existence.
- `oflm-add`'s `engine_user_directories()` returns the same list, so what it tells a user
  about the engine is true of the engine.

### USERDIR-REGISTRY-LAYERS: which registries are read
**Applies to:** openflowlm-next (`src/common/utils.cpp`, `src/src/main.cpp`)
**Test category:** unit
**Test:** `src/common/user_dirs_test.cpp` (`model_list_layers`, `find_xclbin_path_for`)

`utils::find_model_lists()` returns the registries the process reads, in merge order.

- When `OFLM_CONFIG_PATH` (or the pre-rename `FLM_CONFIG_PATH`) names an existing file
  that is not the built-in registry, that file alone: an explicit registry is the whole
  registry, exactly as before #30, so an install that exported it at a full copy keeps
  the behaviour it had.
- When it names **the built-in registry itself** -- `home_install.sh` writes an
  `oflm_env.sh` that exports exactly that -- it is not a custom registry, and the user
  files are merged over it as below. Otherwise every install made that way would never
  see a user model.
- When it is set but names nothing, one line says it is ignored, and the merge below applies.
- Otherwise the built-in registry, then `<dir>/model_list.json` for every user directory
  that has one, **oldest first**, so the newest is applied last.

The built-in registry is the first of `<exe>/model_list.json`, `./model_list.json`,
`<exe>/../share/oflm/model_list.json`, `<prefix>/share/oflm/model_list.json` that exists
**and is not a user registry**. The CWD candidate makes that exclusion necessary:
`cd ~/.config/oflm && oflm list` would otherwise take a user file -- which now holds only
what was added -- as the base, and lose every built-in model.

**Acceptance criteria:**
- With no user files: one layer, the built-in registry.
- With a file in `.oflm` and one in `.config/flm`: three layers -- built-in, `.config/flm`,
  `.oflm`.
- A user file that is the same file as the built-in registry is not read twice.
- An explicit custom registry: exactly that file. An explicit path equivalent to the
  built-in registry: the same three layers as no explicit path.
- A candidate list whose first existing entry is `<user dir>/model_list.json` yields the
  next candidate, not the user file.

### USERDIR-REGISTRY-MERGE: how the registries combine
**Applies to:** openflowlm-next (`src/include/model_registry.hpp`, `src/include/model_list.hpp`)
**Test category:** unit
**Test:** `src/common/user_dirs_test.cpp` (`model_registry::merge`, `model_list over merged registries`)

Only `models` is merged; every other top-level key, `model_path` included, comes from the
built-in registry. For each user entry:

- **A built-in tag is never replaced.** The built-ins are what the application ships and
  tests; a user entry under the same tag would make `oflm run qwen3:8b` mean something
  different on one machine. An entry identical to the built-in one is skipped in silence
  -- older `oflm-add` versions seeded user files with a full copy, and those copies are
  harmless. A *different* entry under a built-in tag is somebody's intention being
  ignored, and is reported: one line per file, with the count and up to three tags.
- **Between user files the later one wins**, and a replacement that changes the entry is
  reported, naming both files.
- A bare type tag (`qwen3`) is itself a built-in tag: it resolves to the size the
  built-in registry lists first, however a user size under that type sorts.
- A user file that cannot be parsed, is not an object, or holds a non-object entry is
  reported and skipped; the built-in models stay available.
- Each user file that contributed entries is reported once, with how many and up to
  three of their tags. The tags are named because an old full-copy registry also holds
  entries that have since *left* the built-in registry; those merge as user models, and
  a count alone would hide them.

**Acceptance criteria:**
- A user file whose `qwen3:8b` differs from the built-in leaves the built-in entry in
  place and yields a note naming the file and `1 entry`.
- Of two user files both defining `mine:1b`, the later one's entry is used, with a note
  naming both files.
- `rectify_model_tag("qwen3")` is `qwen3:4b` when the built-ins are `{4b, 8b}` and a user
  file adds `0.5b`; `qwen3:0.5b` is still listed.
- A user file containing `{ not json` leaves every built-in tag listed.
- Run against the two real user registries on the development machine (40 and 47
  entries, both old full copies) with no `OFLM_CONFIG_PATH`, `oflm list` prints the 44
  rows it prints for the built-in registry alone, byte-identical, plus 3 user models
  (47), and reports the 37 stale copies in one line.

### USERDIR-MODEL-PER-ENTRY: an installed model is found per model
**Applies to:** openflowlm-next (`src/include/model_list.hpp`, `src/common/utils.cpp`)
**Test category:** unit
**Test:** `src/common/user_dirs_test.cpp` (`model_list over merged registries`, `models_search_roots`)

`model_list::get_model_path(tag)` returns `<root>/<model_path>/<name>` for the first root
in `utils::models_directories()` holding a **complete** copy -- every file the entry's
`files` lists exists (an entry without `files`: the directory is not empty) -- and
otherwise `<get_models_directory()>/<model_path>/<name>`. `models_directories()` is
`{OFLM_MODEL_PATH}` when that is set (the one store, as it always was) and
`user_directories()` otherwise. `get_models_directory()` itself is unchanged: it still
picks the default directory for new downloads and for history.

Complete, not merely present, because the same path is where `oflm pull` writes and what
`oflm remove` deletes. A partial or abandoned copy in another store -- possibly one a
different installation still uses -- must never become that target; anything short of
complete belongs to the default directory. `oflm pull` prints the resolved path.

**Acceptance criteria:**
- With `Qwen3-4B-NPU2` complete under `.oflm/models`, and `Qwen3-8B-NPU2` complete under
  `.config/flm/models` but partial under `.oflm/models`: each tag resolves to its complete
  copy -- the pre-rename store is not hidden by the new one, nor by a partial copy in it.
- A user model under `.config/oflm/models` (where `oflm-add` installs by default) is found.
- A model that is only a partial copy under `.config/flm/models` resolves to the default
  directory.
- A model installed nowhere resolves to the default directory.
- With `OFLM_MODEL_PATH` set, that directory is the only root searched.

### USERDIR-XCLBIN-PER-NAME: a kernel root is chosen per model name
**Applies to:** openflowlm-next (`src/common/utils.cpp`, `src/include/lm_config.hpp`, `src/open_npue_adapter/npue_embedding.cpp`, `src/open_embedding/engine.cpp`)
**Test category:** unit (`find_xclbin_path_for`); the two embedding lookups are covered by the build only
**Test:** `src/common/user_dirs_test.cpp` (`find_xclbin_path_for`)

`utils::find_xclbin_path()` returns the first root with an `xclbins/` directory at all,
so a user root holding one linked model hid the install tree's other forty. That is the
whole reason `oflm-add` used to print `export OFLM_XCLBIN_PATH`. Every consumer that
looks for a *named* directory now searches `utils::xclbin_roots_install_first()` for it:
`$OFLM_XCLBIN_PATH`, then the install tree (the executable's directory, the CWD,
`<exe>/../share/oflm`, the configured prefix), then the directory holding
`$OFLM_CONFIG_PATH`, then `user_directories()`.

The install tree comes **before** the user directories, for the reason a user registry
cannot replace a built-in model: a user directory adds names, it does not replace shipped
ones. It is not hypothetical -- `q4nx-build` symlinks *every* official `xclbins/<name>`
into `~/.config/oflm/xclbins`, so a user directory searched first would make a second
installation or a development build load the first installation's kernels, in silence.
(The open engine's `find_kernels` keeps its own order, user roots first, per OPEN-MANIFEST:
there the unit is a set a user deliberately builds or links for one model.)

- the closed path: `LM_Config::exec_path` is `utils::find_xclbin_path_for(<model name>)`,
  the first root holding `xclbins/<model name>`, falling back to `find_xclbin_path()`;
- the npue embedding backend: the first root whose `xclbins/<npue_design_family>` holds
  a complete design set (`gemm_rtp/design.json`). A family is shared by several models
  and built rather than downloaded, so one family built into the user directory while
  the rest stay in the install tree is the expected state;
- the open embedding engine: the first root holding `npu_matmul_f32` kernels for its family.

`find_xclbin_path()` itself is unchanged.

**Acceptance criteria:**
- `xclbin_search_order({ENV, EXE, SHARE}, CFG, {U1, U2})` is `ENV, EXE, SHARE, CFG, U1, U2`.
- With `Qwen3-8B-NPU2` under both an install root and a user root, the install root serves
  it; a name only the user root holds is served from the user root.
- With `OFLM_XCLBIN_PATH` = A (holding `xclbins/Linked-NPU2`) and `OFLM_CONFIG_PATH` in B
  (holding `xclbins/Shipped-NPU2`): `find_xclbin_path()` is A;
  `find_xclbin_path_for("Linked-NPU2")` is A; `find_xclbin_path_for("Shipped-NPU2")` is B;
  a name under neither is A.

### USERDIR-MODEL-INFO: model_info.json is looked for in the same places on every platform
**Applies to:** openflowlm-next (`src/common/utils.cpp`)
**Test category:** unit
**Test:** `src/common/user_dirs_test.cpp` (`model_info_candidates`)

After `OFLM_MODELINFO_PATH` and the directory holding `OFLM_CONFIG_PATH`,
`find_model_info()` tries `<exe>/model_info.json`, `<exe>/../share/oflm/model_info.json`,
then `<prefix>/share/oflm/model_info.json`. The Windows branch used to stop after the
first.

It does **not** search the user directories. `model_info.json` holds download sizes and
hashes for the built-in models, nothing writes one into a user directory, and a copy
there would be a whole file that goes stale -- the trap this function's own comment
already warns about. There is nothing to merge and nothing to find, so there is no lookup.

**Acceptance criteria:**
- `model_info_candidates("E", "P")` is exactly those three paths, in that order, on every
  platform.

### USERDIR-HISTORY: saved history goes to the models directory on every platform
**Applies to:** openflowlm-next (`src/runner/runner.cpp`)
**Test category:** build only

`/save` writes to `<get_models_directory()>/history`. The POSIX branch used to read
`OFLM_MODEL_PATH` with a bare `getenv` and build `~/.config/oflm` by hand, so it saw
neither the pre-rename `FLM_MODEL_PATH` nor a pre-rename directory. History is only ever
written, so there is nothing to search for.

### USERDIR-ADD-SEED: a user registry holds what was added, unless it is read whole
**Applies to:** openflowlm-next (`utilities/oflm-add/oflm_add/__init__.py`, `utilities/q4nx-build/q4nx/deploy.py`)
**Test category:** unit
**Tests:** `utilities/oflm-add/tests/test_user_registry.py`, `utilities/q4nx-build/tests/test_deploy_registry.py`

Both tools mirror USERDIR-REGISTRY-LAYERS exactly, in `whole_registry_env()`: a
`OFLM_CONFIG_PATH`/`FLM_CONFIG_PATH` naming the built-in registry (next to `oflm`, in
`<oflm>/../share/oflm`, or a system prefix) is not a custom registry. The registry they
write is the variable's path only when it names a custom one; one naming the install's
built-in list is not theirs to edit, so they write the default user file instead.

When the registry file does not exist yet:

- if the engine **merges** it -- it is `<user dir>/model_list.json`, and no variable names
  an existing custom registry or this very file -- it starts as
  `{"model_path": "models", "models": {}}`. A copy of the built-in entries would go stale
  the day the application ships a new model, and the merge would ignore them anyway;
- otherwise it starts as a copy of the system registry, because the only way the engine
  reads it is as the whole registry, through `OFLM_CONFIG_PATH`. A variable naming the
  file about to be written counts: once written, the engine reads it whole.

An existing file is extended, never reseeded.

**Acceptance criteria:**
- Default path, no variable: the file's `models` is exactly the added entry.
- `OFLM_CONFIG_PATH` (or `FLM_CONFIG_PATH`) naming the file: the built-in entries are in it.
- A path in no user directory: the built-in entries are in it.
- A file that already holds `old:1b` holds `old:1b` and the new entry afterwards.
- `OFLM_CONFIG_PATH` naming the built-in registry: the tools write the default user file,
  seed it with the added entry only, leave the built-in file byte-identical, and advise
  no export.
- `FLM_CONFIG_PATH` naming a missing file elsewhere: the default user file still counts as
  merged, as it does for the engine.

### USERDIR-ADD-EXPORTS: advice to export a variable only when the engine needs it
**Applies to:** openflowlm-next (`utilities/oflm-add/oflm_add/__init__.py`, `utilities/q4nx-build/q4nx/deploy.py`)
**Test category:** unit
**Tests:** `utilities/oflm-add/tests/test_user_registry.py`, `utilities/q4nx-build/tests/test_deploy_registry.py`

`oflm-add` used to end every install with two `export` lines. It now prints
`export OFLM_CONFIG_PATH` only when the engine neither merges the registry nor has it
named, and `export OFLM_XCLBIN_PATH` only when the xclbin root is not among the roots
`utils::xclbin_roots()` searches (the user directories, `$OFLM_XCLBIN_PATH`, and the
directory holding `$OFLM_CONFIG_PATH` -- counting one the same run just advised
exporting). A models root the engine does not search is a warning, not an export:
`OFLM_MODEL_PATH` makes one directory the *only* store, which would hide every model
somewhere else.

**Acceptance criteria:**
- Default `--config`, `--xclbin-dir` and `--models-root`: no exports, no warnings.
- A custom registry path and a custom xclbin root: both exports.
- A custom registry path with its xclbins beside it: only `OFLM_CONFIG_PATH`.
- A custom models root: one warning and no `OFLM_MODEL_PATH` export.
