# user-directory: where the application finds the models it knows about

Prefix `USERDIR`. Home repo: openflowlm-next. Covers the lookups in `src/common/utils.cpp`
(`user_directories`, `registry_directories`, `find_model_lists`, `models_directories`,
`find_model_infos`), the merges in `src/include/model_registry.hpp`, and how
`src/include/model_list.hpp` resolves a model's directory. Issue: #30.

Tests: `src/common/user_dirs_test.cpp` (ctest `user_dirs`; no device, no weights, no network).

Kernel lookup is out of scope: where xclbins and open kernel sets are found is unchanged.

Every defect this file guards against is a **well-formed wrong answer**. A registry nobody
reads shows fewer models. A model looked up in the wrong directory shows as not downloaded
and is pulled again. A `model_info.json` nobody reads leaves an added model failing file
verification. None of them crashes, which is why the acceptance criteria below name paths
and counts rather than "it runs".

There are two kinds of lookup, and they want different answers:

- **One thing per name** -- a model's folder. The first directory holding a complete copy
  wins. A directory that merely exists must not win, or one model installed into a new
  directory hides every model in the old one.
- **A file that is many entries** -- `model_list.json`, `model_info.json`. First-wins would
  lose a user's whole registry the moment a newer directory gained one, so these files are
  merged.

## Requirements

### USERDIR-ORDER: the user directories, newest name first
**Applies to:** openflowlm-next (`src/common/utils.cpp`)
**Test category:** unit
**Test:** `src/common/user_dirs_test.cpp` (`user_directories`)

`utils::user_directories()` is the one list of user-level directories. On Windows, with
`<user>` the profile directory: `<user>\.oflm`, `<user>\.config\oflm`, `<user>\.flm`,
`<user>\.config\flm`. On POSIX, with `<user>` = `$HOME/.config`: `<user>/oflm`,
`<user>/flm`. The `.config` form exists on Windows because the Python `oflm-add` writes to
`Path.home()/.config/oflm` on every platform; the `flm` forms because an install that
predates the rename must keep working without moving gigabytes of weights.

**Acceptance criteria:**
- The list has exactly those entries in exactly that order, unfiltered by existence.

### USERDIR-REGISTRY-DIRS: where user registries are looked for
**Applies to:** openflowlm-next (`src/common/utils.cpp`)
**Test category:** unit
**Test:** `src/common/user_dirs_test.cpp` (`registry_directories`, `model_info.json`)

`utils::registry_directories()` is `$OFLM_MODEL_PATH` (or `FLM_MODEL_PATH`) when set, then
`user_directories()`. The models directory is included because that is where `oflm add`
(#34) writes its `model_list.json` and `model_info.json` -- beside the models -- and an
explicit `OFLM_MODEL_PATH` is not one of the user directories.

**Acceptance criteria:**
- Without `OFLM_MODEL_PATH`: exactly `user_directories()`.
- With it: that directory first, then `user_directories()`.

### USERDIR-REGISTRY-LAYERS: which model lists are read
**Applies to:** openflowlm-next (`src/common/utils.cpp`, `src/src/main.cpp`)
**Test category:** unit
**Test:** `src/common/user_dirs_test.cpp` (`model_list_layers`, `find_model_lists with OFLM_CONFIG_PATH`)

`utils::find_model_lists()` returns the registries the process reads, in merge order.

- When `OFLM_CONFIG_PATH` (or the pre-rename `FLM_CONFIG_PATH`) names an existing file
  that is not the built-in registry, that file alone: an explicit registry is the whole
  registry, exactly as before #30, so an install that exported it at a full copy keeps
  the behaviour it had.
- When it names **the built-in registry itself** -- `home_install.sh` writes an
  `oflm_env.sh` that exports exactly that -- it is not a custom registry, and the user
  files are merged over it as below. Otherwise no install made that way would ever see a
  user model.
- When it is set but names nothing, one line says it is ignored, and the merge applies.
- Otherwise the built-in registry, then `<dir>/model_list.json` for every registry
  directory that has one, **oldest first**, so the newest is applied last.

The built-in registry is the first of `<exe>/model_list.json`, `./model_list.json`,
`<exe>/../share/oflm/model_list.json`, `<prefix>/share/oflm/model_list.json` that exists
**and is not a user registry**. The CWD candidate makes that exclusion necessary:
`cd ~/.config/oflm && oflm list` would otherwise take a user file as the base, and a user
file that holds only what was added would lose every built-in model.

**Acceptance criteria:**
- With no user files: one layer, the built-in registry.
- With a file in `.oflm` and one in `.config/flm`: three layers -- built-in, `.config/flm`,
  `.oflm`.
- A user file that is the same file as the built-in registry is not read twice.
- An explicit custom registry: exactly that file, through `find_model_lists()` itself.
  An explicit path equivalent to the built-in registry: the same layers as no explicit path.
- A candidate list whose first existing entry is `<user dir>/model_list.json` yields the
  next candidate, not the user file.

### USERDIR-REGISTRY-MERGE: how the model lists combine
**Applies to:** openflowlm-next (`src/include/model_registry.hpp`, `src/include/model_list.hpp`)
**Test category:** unit
**Test:** `src/common/user_dirs_test.cpp` (`model_registry::merge`, `model_list over merged registries`)

Only `models` is merged; every other top-level key, `model_path` included, comes from the
built-in registry. For each user entry:

- **A built-in tag is never replaced.** The built-ins are what the application ships and
  tests; a user entry under the same tag would make `oflm run qwen3:8b` mean something
  different on one machine. An entry identical to the built-in one is skipped in silence
  -- `oflm-add` has always seeded user files with a full copy, and those copies are
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
  rows it prints for a registry read alone, byte-identical, plus 3 user models (47), and
  reports the 37 stale copies in one line.

### USERDIR-MODEL-PER-ENTRY: an installed model is found per model
**Applies to:** openflowlm-next (`src/include/model_list.hpp`, `src/common/utils.cpp`, `src/pull/model_downloader.cpp`)
**Test category:** unit
**Test:** `src/common/user_dirs_test.cpp` (`model_list over merged registries`, `models_search_roots`)

`model_list::get_model_path(tag)` returns `<root>/<model_path>/<name>` for the first root
in `utils::models_directories()` holding a **complete** copy -- every file the entry's
`files` lists exists (an entry without `files`: the directory is not empty) -- and
otherwise `<get_models_directory()>/<model_path>/<name>`. `models_directories()` is
`{OFLM_MODEL_PATH}` when that is set (the one store, as it always was) and
`user_directories()` otherwise. `get_models_directory()` itself is unchanged: it still
picks the default directory for new downloads.

Complete, not merely present, because the same path is where `oflm pull` writes and what
`oflm remove` deletes. A partial or abandoned copy in another store -- possibly one a
different installation still uses -- must never become that target; anything short of
complete belongs to the default directory. `oflm pull` prints the resolved path.

**Acceptance criteria:**
- With `Qwen3-4B-NPU2` complete under `.oflm/models`, and `Qwen3-8B-NPU2` complete under
  `.config/flm/models` but partial under `.oflm/models`: each tag resolves to its complete
  copy.
- A user model under `.config/oflm/models` is found.
- A model that is only a partial copy under `.config/flm/models` resolves to the default
  directory.
- A model installed nowhere resolves to the default directory.
- With `OFLM_MODEL_PATH` set, that directory is the only root searched.

### USERDIR-MODEL-INFO: model_info.json is merged the same way
**Applies to:** openflowlm-next (`src/common/utils.cpp`, `src/include/model_registry.hpp`, `src/pull/model_downloader.cpp`)
**Test category:** unit, plus the application
**Test:** `src/common/user_dirs_test.cpp` (`model_info.json`)

`model_info.json` holds, per tag, a model's files with their sizes and hashes; the
downloader reads it to size downloads and to verify files. `oflm add` (#34) writes one
beside the user registry, and without it a model it added fails verification with
`key '<tag>' not found`. Until now that only worked with `OFLM_CONFIG_PATH` exported,
because `find_model_info()` looks beside that file.

`utils::find_model_infos()` returns, in merge order:

- `{$OFLM_MODELINFO_PATH}` when that names an existing file -- the whole file, as before;
- otherwise the shipped file -- the first of beside `$OFLM_CONFIG_PATH`,
  `<exe>/model_info.json`, `<exe>/../share/oflm/model_info.json`,
  `<prefix>/share/oflm/model_info.json` that exists and is not a user registry's -- then
  `<dir>/model_info.json` for every registry directory that has one, oldest first.

The Windows branch of the old lookup stopped at the executable's directory; both platforms
now try all three installed locations.

`model_registry::load_model_info()` merges them by tag with the same two rules as the model
list: a user file never replaces a shipped tag -- those hashes are what downloads are
checked against -- and a later user file wins. It is silent: it runs once per model on
every `oflm list`, and an unreadable user file only means its models go unverified.

**Acceptance criteria:**
- `model_info_candidates("E", "P")` is `E/model_info.json`, `E/../share/oflm/model_info.json`,
  `P/share/oflm/model_info.json`, on every platform.
- A user `qwen3:8b` entry leaves the shipped hashes in place; a newer user file's
  `mine:1b` wins over an older one's; a user-only tag is added.
- With no shipped file, the user entries still load; a broken user file is skipped.
- With `OFLM_MODEL_PATH` naming a directory that holds a `model_info.json`, that file is the
  last layer `find_model_infos()` returns. With `OFLM_MODELINFO_PATH` set, only that file.
- The application, with no `OFLM_CONFIG_PATH`, and `OFLM_MODEL_PATH` naming a directory laid
  out the way `oflm add` writes one (`model_list.json`, `model_info.json`,
  `models/<name>/`): `oflm list` shows the tag as downloaded and `oflm check <tag>` reports
  every file `Success!`. With that `model_info.json` removed, the same check fails with
  `key '<tag>' not found`.
