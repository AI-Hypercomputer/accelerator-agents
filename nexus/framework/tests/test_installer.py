"""Unit tests for installer module (install, upgrade, rollback, uninstall)."""

import json
import pathlib
import shutil
import tempfile
import unittest

try:
  from accelerator_agents.tpu_nexus.framework import compiler, installer, utils
except ImportError:
  try:
    from .. import compiler, installer, utils
  except ImportError:
    import compiler
    import installer
    import utils

Path = pathlib.Path


class TestInstaller(unittest.TestCase):
  """Unit tests for installer functions, pre-validations, collisions, and rollback."""

  def setUp(self):
    super().setUp()
    self.temp_dir = Path(tempfile.mkdtemp())
    self.pkg_dir = self._create_test_package()
    self.dist_dir = self.temp_dir / "dist"
    compiler.translate(self.pkg_dir, "claude-code", self.dist_dir)

  def tearDown(self):
    shutil.rmtree(self.temp_dir, ignore_errors=True)
    super().tearDown()

  def _create_test_package(self) -> Path:
    pkg_dir = self.temp_dir / "pkg"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "name": "pkg",
        "version": "1.0.0",
        "description": "Installer test package",
        "entrypoint": {
            "name": "entry",
            "instructions": "instructions/entry.md",
            "delegates": ["helper"],
        },
        "agents": [{
            "name": "helper",
            "description": "Helper",
            "instructions": "instructions/helper.md",
            "accepts": "schemas/in.json",
            "produces": "schemas/out.json",
            "delegates": [],
            "tools": ["read"],
        }],
        "copy": ["schemas", "environment"],
        "environment": {
            "questions": "environment/questions.json",
            "schema": "schemas/env.json",
        },
        "compatibility_targets": [{
            "name": "claude",
            "harness": "claude-code",
            "versions": ">=2.0.0",
            "capabilities": {},
        }],
        "schemas": ["schemas/in.json", "schemas/out.json", "schemas/env.json"],
    }
    (pkg_dir / "package.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (pkg_dir / "instructions").mkdir(parents=True, exist_ok=True)
    (pkg_dir / "instructions" / "entry.md").write_text(
        "Call {{agent:helper}}.", encoding="utf-8"
    )
    (pkg_dir / "instructions" / "helper.md").write_text(
        "Help.", encoding="utf-8"
    )
    (pkg_dir / "schemas").mkdir(parents=True, exist_ok=True)
    (pkg_dir / "schemas" / "in.json").write_text(
        json.dumps({"type": "object"}), encoding="utf-8"
    )
    (pkg_dir / "schemas" / "out.json").write_text(
        json.dumps({"type": "object"}), encoding="utf-8"
    )
    (pkg_dir / "schemas" / "env.json").write_text(
        json.dumps({
            "type": "object",
            "properties": {
                "tpu_target": {"type": "string"},
                "timeout": {"type": "integer"},
            },
            "required": ["tpu_target", "timeout"],
        }),
        encoding="utf-8",
    )
    (pkg_dir / "environment").mkdir(parents=True, exist_ok=True)
    (pkg_dir / "environment" / "questions.json").write_text(
        json.dumps({
            "questions": [
                {
                    "name": "tpu_target",
                    "prompt": "TPU target",
                    "type": "string",
                    "default": "local-tpu",
                },
                {
                    "name": "timeout",
                    "prompt": "Timeout",
                    "type": "integer",
                    "default": 30,
                },
            ]
        }),
        encoding="utf-8",
    )
    return pkg_dir

  def test_install_and_environment_resolution(self):
    workspace = self.temp_dir / "workspace"
    workspace.mkdir()

    answers = self.temp_dir / "answers.json"
    answers.write_text(
        json.dumps({"tpu_target": "node-1", "timeout": 60}), encoding="utf-8"
    )

    installer.install(
        distribution=self.dist_dir,
        destination=workspace,
        answers_path=answers,
        non_interactive=True,
        upgrade=False,
        skip_harness_check=True,
    )

    env_file = workspace / ".coworker" / "pkg" / "environment.json"
    self.assertTrue(env_file.is_file())
    env = json.loads(env_file.read_text(encoding="utf-8"))
    self.assertEqual(env["tpu_target"], "node-1")
    self.assertEqual(env["timeout"], 60)
    self.assertEqual(env["revision"], 1)

    ownership_file = workspace / ".coworker" / "pkg" / "ownership.json"
    self.assertTrue(ownership_file.is_file())

  def test_collision_detection(self):
    workspace = self.temp_dir / "workspace_collision"
    workspace.mkdir()
    (workspace / ".claude" / "skills" / "entry").mkdir(parents=True)
    (workspace / ".claude" / "skills" / "entry" / "SKILL.md").write_text(
        "EXISTING UNMANAGED FILE", encoding="utf-8"
    )

    with self.assertRaises(utils.CoworkerError) as ctx:
      installer.install(
          distribution=self.dist_dir,
          destination=workspace,
          answers_path=None,
          non_interactive=True,
          upgrade=False,
          skip_harness_check=True,
      )
    self.assertIn("refusing to overwrite", str(ctx.exception).lower())

  def test_uninstall_and_rollback(self):
    workspace = self.temp_dir / "workspace_uninstall"
    workspace.mkdir()

    installer.install(
        distribution=self.dist_dir,
        destination=workspace,
        answers_path=None,
        non_interactive=True,
        upgrade=False,
        skip_harness_check=True,
    )

    # Modify an agent file
    agent_file = workspace / ".claude" / "agents" / "helper.md"
    self.assertTrue(agent_file.is_file())
    agent_file.write_text("# USER MODIFIED", encoding="utf-8")

    modified = installer.uninstall(workspace, "pkg")
    self.assertIn(".claude/agents/helper.md", modified)
    self.assertTrue(agent_file.is_file())  # Preserved
    self.assertFalse((workspace / ".coworker" / "pkg" / "ownership.json").exists())


if __name__ == "__main__":
  unittest.main()
