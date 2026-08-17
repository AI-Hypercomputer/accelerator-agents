"""Unit tests for compiler module (verification and translation)."""

import json
import pathlib
import shutil
import tempfile
import unittest

try:
  from accelerator_agents.tpu_nexus.framework import compiler, utils
except ImportError:
  try:
    from .. import compiler, utils
  except ImportError:
    import compiler
    import utils

Path = pathlib.Path


class TestCompiler(unittest.TestCase):
  """Unit tests for package verification, delegation graph, and translation."""

  def setUp(self):
    super().setUp()
    self.temp_dir = Path(tempfile.mkdtemp())

  def tearDown(self):
    shutil.rmtree(self.temp_dir, ignore_errors=True)
    super().tearDown()

  def _create_valid_package(self) -> Path:
    pkg_dir = self.temp_dir / "valid-pkg"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "name": "valid-pkg",
        "version": "1.0.0",
        "description": "Valid test package",
        "entrypoint": {
            "name": "entry",
            "instructions": "instructions/entry.md",
            "delegates": ["worker"],
        },
        "agents": [{
            "name": "worker",
            "description": "Worker",
            "instructions": "instructions/worker.md",
            "accepts": "schemas/in.json",
            "produces": "schemas/out.json",
            "delegates": [],
            "tools": ["read", "write"],
        }],
        "copy": ["schemas", "environment"],
        "environment": {
            "questions": "environment/questions.json",
            "schema": "schemas/env.json",
        },
        "compatibility_targets": [
            {
                "name": "claude",
                "harness": "claude-code",
                "versions": ">=2.0.0",
                "capabilities": {},
            },
            {
                "name": "codex",
                "harness": "codex",
                "versions": ">=0.145.0",
                "capabilities": {},
            },
        ],
        "schemas": ["schemas/in.json", "schemas/out.json", "schemas/env.json"],
    }
    (pkg_dir / "package.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (pkg_dir / "instructions").mkdir(parents=True, exist_ok=True)
    (pkg_dir / "instructions" / "entry.md").write_text(
        "Call {{agent:worker}}.", encoding="utf-8"
    )
    (pkg_dir / "instructions" / "worker.md").write_text(
        "Work.", encoding="utf-8"
    )
    (pkg_dir / "schemas").mkdir(parents=True, exist_ok=True)
    (pkg_dir / "schemas" / "in.json").write_text(
        json.dumps({"type": "object"}), encoding="utf-8"
    )
    (pkg_dir / "schemas" / "out.json").write_text(
        json.dumps({"type": "object"}), encoding="utf-8"
    )
    (pkg_dir / "schemas" / "env.json").write_text(
        json.dumps({"type": "object"}), encoding="utf-8"
    )
    (pkg_dir / "environment").mkdir(parents=True, exist_ok=True)
    (pkg_dir / "environment" / "questions.json").write_text(
        json.dumps({"questions": []}), encoding="utf-8"
    )
    return pkg_dir

  def test_verify_valid_package(self):
    pkg_dir = self._create_valid_package()
    # Should complete without error
    compiler.verify_package(pkg_dir)

  def test_cycle_detection(self):
    pkg_dir = self.temp_dir / "cycle-pkg"
    pkg_dir.mkdir()
    manifest = {
        "name": "cycle-pkg",
        "version": "1.0.0",
        "description": "Cycle",
        "entrypoint": {
            "name": "entry",
            "instructions": "entry.md",
            "delegates": ["a"],
        },
        "agents": [
            {
                "name": "a",
                "description": "a",
                "instructions": "a.md",
                "accepts": "s.json",
                "produces": "s.json",
                "delegates": ["b"],
                "tools": ["read"],
            },
            {
                "name": "b",
                "description": "b",
                "instructions": "b.md",
                "accepts": "s.json",
                "produces": "s.json",
                "delegates": ["a"],
                "tools": ["read"],
            },
        ],
        "environment": {"questions": "q.json", "schema": "s.json"},
        "compatibility_targets": [
            {"name": "c", "harness": "claude-code", "versions": ">=1.0.0"}
        ],
    }
    (pkg_dir / "package.json").write_text(json.dumps(manifest), encoding="utf-8")
    (pkg_dir / "entry.md").write_text("entry", encoding="utf-8")
    (pkg_dir / "a.md").write_text("a", encoding="utf-8")
    (pkg_dir / "b.md").write_text("b", encoding="utf-8")
    (pkg_dir / "s.json").write_text("{}", encoding="utf-8")
    (pkg_dir / "q.json").write_text(
        json.dumps({"questions": []}), encoding="utf-8"
    )

    with self.assertRaises(utils.CoworkerError) as ctx:
      compiler.load_package(pkg_dir)
    self.assertIn("cycle", str(ctx.exception).lower())

  def test_translate_claude(self):
    pkg_dir = self._create_valid_package()
    dist_dir = self.temp_dir / "dist_claude"
    compiler.translate(pkg_dir, "claude-code", dist_dir)

    self.assertTrue((dist_dir / ".claude-plugin" / "plugin.json").is_file())
    self.assertTrue((dist_dir / "skills" / "entry" / "SKILL.md").is_file())
    self.assertTrue((dist_dir / "agents" / "worker.md").is_file())
    self.assertTrue((dist_dir / "coworker-build.json").is_file())
    self.assertTrue((dist_dir / "coworker-install.json").is_file())
    self.assertTrue((dist_dir / "runtime" / "runtime.py").is_file())
    self.assertTrue((dist_dir / "runtime" / "utils.py").is_file())

  def test_translate_codex(self):
    pkg_dir = self._create_valid_package()
    dist_dir = self.temp_dir / "dist_codex"
    compiler.translate(pkg_dir, "codex", dist_dir)

    self.assertTrue((dist_dir / ".codex-plugin" / "plugin.json").is_file())
    self.assertTrue((dist_dir / "skills" / "entry" / "SKILL.md").is_file())
    self.assertTrue((dist_dir / "agents" / "worker.toml").is_file())


if __name__ == "__main__":
  unittest.main()
