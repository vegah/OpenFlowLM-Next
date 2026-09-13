# Traces: USERDIR-ADD-SEED, USERDIR-ADD-EXPORTS (canonical spec: specs/user-directory/spec.md)
#
# The engine merges every user-level model_list.json over its built-in registry
# and searches the user directories for models and xclbins by name (#30). So
# oflm-add must seed a registry the engine MERGES with the added entries only,
# must still seed a full copy for a registry the engine reads WHOLE (the one
# OFLM_CONFIG_PATH names), and must stop telling people to export variables the
# engine no longer needs.
import json
import os
import sys
from pathlib import Path

import pytest

OFLM_ADD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OFLM_ADD))

import oflm_add  # noqa: E402

SYSTEM = {
    "model_path": "models",
    "models": {
        "qwen3": {"8b": {"name": "Qwen3-8B-NPU2", "details": {"family": "qwen3"}}},
        "llama3.2": {"1b": {"name": "Llama-3.2-1B-NPU2", "details": {"family": "llama"}}},
    },
}
ENTRY = {"name": "Mine-8B-NPU2", "details": {"family": "qwen3"}}
VARS = ["OFLM_CONFIG_PATH", "FLM_CONFIG_PATH", "OFLM_XCLBIN_PATH", "FLM_XCLBIN_PATH",
        "OFLM_MODEL_PATH", "FLM_MODEL_PATH"]


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: h))
    for v in VARS:
        monkeypatch.delenv(v, raising=False)
    return h


def default_registry(home):
    return home / ".config" / "oflm" / "model_list.json"


def test_user_directories_match_the_engine(home):
    """Same list, same order as utils::user_directories() (USERDIR-ORDER)."""
    dirs = oflm_add.engine_user_directories()
    if os.name == "nt":
        assert dirs == [home / ".oflm", home / ".config" / "oflm", home / ".flm", home / ".config" / "flm"]
    else:
        assert dirs == [home / ".config" / "oflm", home / ".config" / "flm"]


def test_a_merged_registry_is_seeded_with_the_added_entry_only(home):
    path = default_registry(home)
    assert oflm_add.engine_merges_registry(path)
    oflm_add.register(path, "mine:8b", ENTRY, SYSTEM)
    reg = json.loads(path.read_text(encoding="utf-8"))
    assert reg["models"] == {"mine": {"8b": ENTRY}}
    assert "qwen3" not in reg["models"], "a built-in entry was copied into a merged user registry"


def test_the_registry_oflm_config_path_names_is_seeded_whole(home, monkeypatch):
    """The engine reads that file as the ENTIRE registry; without the built-ins they vanish."""
    path = default_registry(home)
    monkeypatch.setenv("OFLM_CONFIG_PATH", str(path))
    assert oflm_add.engine_reads_whole_registry(path)
    assert not oflm_add.engine_merges_registry(path)
    oflm_add.register(path, "mine:8b", ENTRY, SYSTEM)
    reg = json.loads(path.read_text(encoding="utf-8"))
    assert reg["models"]["qwen3"] == SYSTEM["models"]["qwen3"]
    assert reg["models"]["mine"]["8b"] == ENTRY


def test_the_pre_rename_variable_counts_as_naming_it(home, monkeypatch):
    path = default_registry(home)
    monkeypatch.setenv("FLM_CONFIG_PATH", str(path))
    assert oflm_add.engine_reads_whole_registry(path)


def test_a_registry_the_engine_does_not_read_is_seeded_whole(home, tmp_path):
    """It only works once exported as OFLM_CONFIG_PATH, which makes it the whole registry."""
    path = tmp_path / "elsewhere" / "model_list.json"
    assert not oflm_add.engine_merges_registry(path)
    oflm_add.register(path, "mine:8b", ENTRY, SYSTEM)
    reg = json.loads(path.read_text(encoding="utf-8"))
    assert "llama3.2" in reg["models"]


def test_an_existing_registry_is_extended_not_reseeded(home):
    path = default_registry(home)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"model_path": "models", "models": {"old": {"1b": {"name": "Old"}}}}),
                    encoding="utf-8")
    oflm_add.register(path, "mine:8b", ENTRY, SYSTEM)
    reg = json.loads(path.read_text(encoding="utf-8"))
    assert set(reg["models"]) == {"old", "mine"}


def test_default_paths_need_no_exports(home):
    exports, warnings = oflm_add.shell_setup(
        oflm_add.user_registry_path(None), oflm_add.user_xclbin_dir(None), oflm_add.models_root_dir(None))
    assert exports == []
    assert warnings == []


def test_a_custom_registry_and_xclbin_root_need_exports(home, tmp_path):
    reg = tmp_path / "custom" / "model_list.json"
    xdir = tmp_path / "kernels" / "xclbins"
    exports, _ = oflm_add.shell_setup(reg, xdir, oflm_add.models_root_dir(None))
    assert ("OFLM_CONFIG_PATH", str(reg)) in exports
    assert ("OFLM_XCLBIN_PATH", str(xdir.parent)) in exports


def test_xclbins_beside_an_exported_registry_need_no_second_export(home, tmp_path):
    """utils::xclbin_roots() includes the directory holding OFLM_CONFIG_PATH."""
    reg = tmp_path / "custom" / "model_list.json"
    exports, _ = oflm_add.shell_setup(reg, reg.parent / "xclbins", oflm_add.models_root_dir(None))
    assert exports == [("OFLM_CONFIG_PATH", str(reg))]


def test_a_models_root_the_engine_does_not_search_is_a_warning_not_an_export(home, tmp_path):
    """OFLM_MODEL_PATH makes one directory the ONLY store, so it is not advice to give blindly."""
    exports, warnings = oflm_add.shell_setup(
        oflm_add.user_registry_path(None), oflm_add.user_xclbin_dir(None), tmp_path / "big-disk" / "models")
    assert all(name != "OFLM_MODEL_PATH" for name, _ in exports)
    assert len(warnings) == 1 and "ONLY models directory" in warnings[0]


def test_an_explicit_model_path_is_the_one_store(home, tmp_path, monkeypatch):
    monkeypatch.setenv("OFLM_MODEL_PATH", str(tmp_path / "store"))
    _, warnings = oflm_add.shell_setup(
        oflm_add.user_registry_path(None), oflm_add.user_xclbin_dir(None), oflm_add.models_root_dir(None))
    assert warnings == []


def test_oflm_config_path_naming_the_builtin_list_still_merges(home, tmp_path, monkeypatch):
    """home_install.sh's oflm_env.sh exports OFLM_CONFIG_PATH at the install's own list."""
    builtin = tmp_path / "prefix" / "share" / "oflm" / "model_list.json"
    builtin.parent.mkdir(parents=True)
    builtin.write_text(json.dumps(SYSTEM), encoding="utf-8")
    monkeypatch.setattr(oflm_add, "builtin_registry_candidates", lambda: [builtin])
    monkeypatch.setenv("OFLM_CONFIG_PATH", str(builtin))
    path = oflm_add.user_registry_path(None)
    assert path == default_registry(home), "oflm-add must not write into the install's built-in list"
    assert oflm_add.engine_merges_registry(path)
    oflm_add.register(path, "mine:8b", ENTRY, SYSTEM)
    assert json.loads(path.read_text(encoding="utf-8"))["models"] == {"mine": {"8b": ENTRY}}
    exports, _ = oflm_add.shell_setup(path, oflm_add.user_xclbin_dir(None), oflm_add.models_root_dir(None))
    assert all(name != "OFLM_CONFIG_PATH" for name, _ in exports)


def test_oflm_config_path_naming_a_missing_file_is_ignored_like_the_engine(home, tmp_path, monkeypatch):
    """utils::find_model_lists() ignores it and merges, so the default file is merged too."""
    monkeypatch.setenv("FLM_CONFIG_PATH", str(tmp_path / "gone" / "model_list.json"))
    assert oflm_add.engine_merges_registry(default_registry(home))
