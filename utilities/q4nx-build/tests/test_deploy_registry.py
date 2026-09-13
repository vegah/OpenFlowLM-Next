"""Tests for how deploy seeds the user registry (USERDIR-ADD-SEED, specs/user-directory/spec.md).

OFLM merges every user-level model_list.json over its built-in registry (#30), so a
registry it merges must hold only the deployed entries, while the registry
OFLM_CONFIG_PATH names is read whole and must keep the official ones.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from q4nx import deploy  # noqa: E402

SYSTEM = {"model_path": "models", "models": {"qwen3": {"8b": {"name": "Qwen3-8B-NPU2"}}}}
ENTRY = {"name": "Mine-8B-NPU2", "details": {"family": "qwen3"}}
VARS = ["OFLM_CONFIG_PATH", "FLM_CONFIG_PATH", "OFLM_XCLBIN_PATH", "FLM_XCLBIN_PATH",
        "OFLM_MODEL_PATH", "FLM_MODEL_PATH"]


class DeployRegistryTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.home = self.tmp / "home"
        self.home.mkdir()
        self.system = self.tmp / "share" / "model_list.json"
        self.system.parent.mkdir()
        self.system.write_text(json.dumps(SYSTEM), encoding="utf-8")
        env = {k: v for k, v in os.environ.items() if k not in VARS}
        self._env = mock.patch.dict(os.environ, env, clear=True)
        self._env.start()
        self._home = mock.patch.object(Path, "home", classmethod(lambda cls: self.home))
        self._home.start()

    def tearDown(self):
        self._home.stop()
        self._env.stop()
        self._tmp.cleanup()

    def read(self, path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def test_the_default_registry_is_merged_and_holds_only_the_deployment(self):
        path = deploy.get_user_registry_path()
        self.assertTrue(deploy.engine_merges_registry(path))
        deploy.register_model("mine:8b", ENTRY, path, self.system)
        self.assertEqual(self.read(path)["models"], {"mine": {"8b": ENTRY}})

    def test_the_registry_oflm_config_path_names_keeps_the_official_entries(self):
        path = self.home / ".config" / "oflm" / "model_list.json"
        os.environ["OFLM_CONFIG_PATH"] = str(path)
        self.assertFalse(deploy.engine_merges_registry(path))
        self.assertTrue(deploy.engine_reads_registry(path))
        deploy.register_model("mine:8b", ENTRY, path, self.system)
        models = self.read(path)["models"]
        self.assertIn("qwen3", models)
        self.assertEqual(models["mine"]["8b"], ENTRY)

    def test_a_registry_oflm_does_not_read_is_not_reported_as_read(self):
        path = self.tmp / "elsewhere" / "model_list.json"
        self.assertFalse(deploy.engine_reads_registry(path))

    def test_oflm_config_path_naming_the_builtin_list_still_merges(self):
        """home_install.sh's oflm_env.sh exports OFLM_CONFIG_PATH at the install's own list."""
        os.environ["OFLM_CONFIG_PATH"] = str(self.system)
        with mock.patch.object(deploy, "builtin_registry_candidates", lambda: [self.system]):
            path = deploy.get_user_registry_path()
            self.assertNotEqual(path.resolve(), self.system.resolve())
            self.assertTrue(deploy.engine_merges_registry(path))
            deploy.register_model("mine:8b", ENTRY, path, self.system)
        self.assertEqual(self.read(path)["models"], {"mine": {"8b": ENTRY}})
        self.assertEqual(self.read(self.system), SYSTEM, "the built-in list was edited")

    def test_oflm_config_path_naming_a_missing_file_is_ignored_like_oflm(self):
        """OFLM ignores a variable naming nothing and merges the default file, so must deploy."""
        gone = self.tmp / "gone" / "model_list.json"
        os.environ["FLM_CONFIG_PATH"] = str(gone)
        self.assertTrue(deploy.engine_merges_registry(self.home / ".config" / "oflm" / "model_list.json"))
        # The variable's own path is still where deploy writes, and it is seeded whole:
        # once it exists OFLM reads it as the entire registry.
        self.assertEqual(deploy.get_user_registry_path(), gone)
        self.assertFalse(deploy.engine_merges_registry(gone))

    def test_the_user_xclbin_tree_is_searched_without_an_export(self):
        self.assertTrue(deploy.engine_searches_xclbin_root(deploy.get_models_root().parent))
        self.assertFalse(deploy.engine_searches_xclbin_root(self.tmp / "kernels"))


if __name__ == "__main__":
    unittest.main()
