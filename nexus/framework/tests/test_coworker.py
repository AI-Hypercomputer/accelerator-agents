"""Unit tests for the Coworker agent framework."""

import json
import pathlib
import shutil
import tempfile
import unittest

from accelerator_agents.tpu_nexus.framework import coworker as cw

Path = pathlib.Path


class TestCoworker(unittest.TestCase):
  """Unit tests for coworker framework verification, translation, and lifecycle."""

  def setUp(self):
    super().setUp()
    self.temp_dir = Path(tempfile.mkdtemp())

  def tearDown(self):
    shutil.rmtree(self.temp_dir, ignore_errors=True)
    super().tearDown()

  def test_semver_and_constraints(self):
    """Test semantic version parsing, comparison, and constraint evaluation."""
    v1 = cw.SemVer.parse("1.2.3")
    v2 = cw.SemVer.parse("1.2.4")
    v3 = cw.SemVer.parse("2.0.0")
    self.assertTrue(v1 < v2 < v3)
    self.assertTrue(cw.version_satisfies(v1, ">=1.0.0,<2.0.0"))
    self.assertTrue(cw.version_satisfies(v1, "==1.2.3"))
    self.assertFalse(cw.version_satisfies(v1, ">1.2.3"))

  def test_schema_validator(self):
    """Test deterministic JSON schema validator subset."""
    schema = {
        "type": "object",
        "required": ["name", "age"],
        "properties": {
            "name": {
                "type": "string",
                "minLength": 2,
                "pattern": r"^[A-Z][a-z]+$",
            },
            "age": {"type": "integer", "minimum": 0, "maximum": 120},
            "role": {"type": "string", "enum": ["admin", "user"]},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "additionalProperties": False,
    }
    # Valid instance
    self.assertEqual(
        cw.validate_instance(
            {"name": "Alice", "age": 30, "role": "admin", "tags": ["lead"]},
            schema,
        ),
        [],
    )

    # Missing required property
    errs = cw.validate_instance({"name": "Alice"}, schema)
    self.assertIn("$/age: required property is missing", errs)

    # Type discrimination: boolean vs integer
    errs = cw.validate_instance({"name": "Alice", "age": True}, schema)
    self.assertIn("$/age: expected integer, got bool", errs)

    # Pattern failure
    errs = cw.validate_instance({"name": "alice123", "age": 30}, schema)
    self.assertIn("$/name: string does not match '^[A-Z][a-z]+$'", errs)

    # Additional property rejection
    errs = cw.validate_instance(
        {"name": "Alice", "age": 30, "extra": 123}, schema
    )
    self.assertIn("$/extra: additional property is not allowed", errs)

  def test_toml_manager(self):
    """Test TOML section and value manipulation."""
    toml_file = self.temp_dir / "config.toml"
    toml_file.write_text('[section1]\nfoo = "bar"\n', encoding="utf-8")

    # Apply setting in existing section
    record = cw.TomlConfigManager.apply_set(
        toml_file, "section1", "key1", "val1"
    )
    self.assertIsNotNone(record)

    # Apply setting in new section
    record2 = cw.TomlConfigManager.apply_set(
        toml_file, "section2", "key2", True
    )
    self.assertIsNotNone(record2)

    # Verify content
    text = toml_file.read_text(encoding="utf-8")
    self.assertIn('key1 = "val1"', text)
    self.assertIn("[section2]", text)
    self.assertIn("key2 = true", text)

  def test_claude_lifecycle(self):
    """Test Claude Code package translation, install, and artifact flow."""
    pkg_dir = self.temp_dir / "my-pkg"
    pkg_dir.mkdir()

    manifest = {
        "name": "my-pkg",
        "version": "1.0.0",
        "description": "Test Package",
        "entrypoint": {
            "name": "main-entry",
            "instructions": "instructions/entry.md",
            "delegates": ["helper"],
        },
        "agents": [{
            "name": "helper",
            "description": "Helper agent",
            "instructions": "instructions/helper.md",
            "accepts": "schemas/input.json",
            "produces": "schemas/output.json",
            "delegates": [],
            "tools": ["read", "search"],
        }],
        "copy": ["schemas", "environment"],
        "environment": {
            "questions": "environment/questions.json",
            "schema": "schemas/env.json",
        },
        "compatibility_targets": [{
            "name": "claude",
            "harness": "claude-code",
            "versions": ">=1.0.0",
            "capabilities": {},
        }],
        "schemas": [
            "schemas/input.json",
            "schemas/output.json",
            "schemas/env.json",
        ],
    }
    (pkg_dir / "package.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (pkg_dir / "instructions").mkdir()
    (pkg_dir / "instructions" / "entry.md").write_text(
        "Call {{agent:helper}} now.", encoding="utf-8"
    )
    (pkg_dir / "instructions" / "helper.md").write_text(
        "I am helper.", encoding="utf-8"
    )
    (pkg_dir / "schemas").mkdir()
    (pkg_dir / "schemas" / "input.json").write_text(
        json.dumps({"type": "object"}), encoding="utf-8"
    )
    (pkg_dir / "schemas" / "output.json").write_text(
        json.dumps({"type": "object"}), encoding="utf-8"
    )
    (pkg_dir / "schemas" / "env.json").write_text(
        json.dumps({
            "type": "object",
            "properties": {"api_key": {"type": "string"}},
            "required": ["api_key"],
        }),
        encoding="utf-8",
    )
    (pkg_dir / "environment").mkdir()
    (pkg_dir / "environment" / "questions.json").write_text(
        json.dumps({
            "questions": [
                {"name": "api_key", "prompt": "Enter API Key", "type": "string"}
            ]
        }),
        encoding="utf-8",
    )

    # 1. Verify package
    cw.verify_package(pkg_dir)

    # 2. Translate package
    dist_dir = self.temp_dir / "dist"
    cw.translate(pkg_dir, "claude-code", dist_dir)
    self.assertTrue((dist_dir / "coworker-build.json").exists())
    self.assertTrue((dist_dir / "coworker-install.json").exists())
    self.assertTrue(
        (dist_dir / "project" / "skills" / "main-entry" / "SKILL.md").exists()
    )
    self.assertTrue((dist_dir / "project" / "agents" / "helper.md").exists())

    # 3. Install package
    workspace_dir = self.temp_dir / "workspace"
    workspace_dir.mkdir()
    answers_file = self.temp_dir / "answers.json"
    answers_file.write_text(
        json.dumps({"api_key": "secret123"}), encoding="utf-8"
    )

    cw.install(
        distribution=dist_dir,
        destination=workspace_dir,
        answers_path=answers_file,
        non_interactive=True,
        upgrade=False,
        skip_harness_check=True,
    )

    env_file = workspace_dir / ".coworker" / "my-pkg" / "environment.json"
    self.assertTrue(env_file.exists())
    env_data = json.loads(env_file.read_text(encoding="utf-8"))
    self.assertEqual(env_data["api_key"], "secret123")
    self.assertEqual(env_data["revision"], 1)

    # 4. Start-run
    run_data = cw.start_run(workspace_dir, "my-pkg")
    run_id = run_data["run_id"]

    # 5. Describe artifact
    artifact_dir = (
        workspace_dir / ".coworker" / "my-pkg" / "runs" / run_id / "artifacts"
    )
    artifact_file = artifact_dir / "result.json"
    artifact_file.write_text(json.dumps({"status": "ok"}), encoding="utf-8")

    descriptor = cw.describe_artifact(
        workspace=workspace_dir,
        package="my-pkg",
        run_id=run_id,
        file="result.json",
        schema="schemas/output.json",
        media_type="application/json",
    )
    self.assertEqual(descriptor["run_id"], run_id)
    self.assertEqual(descriptor["media_type"], "application/json")
    self.assertTrue(descriptor["uri"].endswith("result.json"))

    # 6. Uninstall
    cw.uninstall(workspace_dir, "my-pkg")
    self.assertFalse(
        (workspace_dir / ".coworker" / "my-pkg" / "ownership.json").exists()
    )

  def test_codex_lifecycle_and_rollback(self):
    """Test Codex package translation, install, and modified file preservation."""
    pkg_dir = self.temp_dir / "codex-pkg"
    pkg_dir.mkdir()

    manifest = {
        "name": "codex-pkg",
        "version": "0.1.0",
        "description": "Codex Package",
        "entrypoint": {
            "name": "orchestrator",
            "instructions": "instructions/entry.md",
            "delegates": ["worker"],
        },
        "agents": [{
            "name": "worker",
            "description": "Worker agent",
            "instructions": "instructions/worker.md",
            "accepts": "schemas/in.json",
            "produces": "schemas/out.json",
            "delegates": [],
            "tools": ["write", "edit"],
        }],
        "copy": ["schemas", "environment"],
        "environment": {
            "questions": "environment/questions.json",
            "schema": "schemas/env.json",
        },
        "compatibility_targets": [{
            "name": "codex",
            "harness": "codex",
            "versions": ">=0.1.0",
            "capabilities": {},
        }],
        "schemas": ["schemas/in.json", "schemas/out.json", "schemas/env.json"],
    }
    (pkg_dir / "package.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (pkg_dir / "instructions").mkdir()
    (pkg_dir / "instructions" / "entry.md").write_text(
        "Delegate to {{agent:worker}}.", encoding="utf-8"
    )
    (pkg_dir / "instructions" / "worker.md").write_text(
        "Working.", encoding="utf-8"
    )
    (pkg_dir / "schemas").mkdir()
    (pkg_dir / "schemas" / "in.json").write_text(
        json.dumps({"type": "object"}), encoding="utf-8"
    )
    (pkg_dir / "schemas" / "out.json").write_text(
        json.dumps({"type": "object"}), encoding="utf-8"
    )
    (pkg_dir / "schemas" / "env.json").write_text(
        json.dumps({"type": "object"}), encoding="utf-8"
    )
    (pkg_dir / "environment").mkdir()
    (pkg_dir / "environment" / "questions.json").write_text(
        json.dumps({"questions": []}), encoding="utf-8"
    )

    # 1. Translate
    dist_dir = self.temp_dir / "dist_codex"
    cw.translate(pkg_dir, "codex", dist_dir)
    self.assertTrue((dist_dir / "agents" / "worker.toml").exists())
    worker_toml = (dist_dir / "agents" / "worker.toml").read_text(
        encoding="utf-8"
    )
    self.assertIn('sandbox_mode = "workspace-write"', worker_toml)

    # 2. Install
    workspace_dir = self.temp_dir / "workspace_codex"
    workspace_dir.mkdir()
    cw.install(
        distribution=dist_dir,
        destination=workspace_dir,
        answers_path=None,
        non_interactive=True,
        upgrade=False,
        skip_harness_check=True,
    )

    codex_config = workspace_dir / ".codex" / "config.toml"
    self.assertTrue(codex_config.exists())
    self.assertIn("enabled = true", codex_config.read_text(encoding="utf-8"))

    # Modify a file to test preservation on uninstall
    worker_target = workspace_dir / ".codex" / "agents" / "worker.toml"
    self.assertTrue(worker_target.exists())
    worker_target.write_text(
        "# USER MODIFIED\n" + worker_target.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    # 3. Uninstall
    modified = cw.uninstall(workspace_dir, "codex-pkg")
    self.assertIn(".codex/agents/worker.toml", modified)
    self.assertTrue(worker_target.exists())

  def test_cycle_and_depth_detection(self):
    """Test delegation graph cycle detection and max-depth enforcement."""
    pkg_dir = self.temp_dir / "cycle-pkg"
    pkg_dir.mkdir()

    # Cycle
    manifest_cycle = {
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
    (pkg_dir / "package.json").write_text(
        json.dumps(manifest_cycle), encoding="utf-8"
    )
    (pkg_dir / "entry.md").write_text("entry", encoding="utf-8")
    (pkg_dir / "a.md").write_text("a", encoding="utf-8")
    (pkg_dir / "b.md").write_text("b", encoding="utf-8")
    (pkg_dir / "s.json").write_text("{}", encoding="utf-8")
    (pkg_dir / "q.json").write_text(
        json.dumps({"questions": []}), encoding="utf-8"
    )

    with self.assertRaises(cw.CoworkerError) as ctx:
      cw.load_package(pkg_dir)
    self.assertIn("cycle", str(ctx.exception).lower())

    # Depth > 2: entry -> a -> b -> c
    manifest_depth = {
        "name": "depth-pkg",
        "version": "1.0.0",
        "description": "Depth",
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
                "delegates": ["c"],
                "tools": ["read"],
            },
            {
                "name": "c",
                "description": "c",
                "instructions": "c.md",
                "accepts": "s.json",
                "produces": "s.json",
                "delegates": [],
                "tools": ["read"],
            },
        ],
        "environment": {"questions": "q.json", "schema": "s.json"},
        "compatibility_targets": [
            {"name": "c", "harness": "claude-code", "versions": ">=1.0.0"}
        ],
    }
    (pkg_dir / "package.json").write_text(
        json.dumps(manifest_depth), encoding="utf-8"
    )
    (pkg_dir / "c.md").write_text("c", encoding="utf-8")

    with self.assertRaises(cw.CoworkerError) as ctx:
      cw.load_package(pkg_dir)
    self.assertIn("two levels", str(ctx.exception).lower())

  def test_cli_main(self):
    """Test CLI validate subcommand with valid and invalid JSON inputs."""
    schema_path = self.temp_dir / "schema.json"
    schema_path.write_text(json.dumps({"type": "integer"}), encoding="utf-8")
    valid_instance = self.temp_dir / "valid.json"
    valid_instance.write_text("42", encoding="utf-8")
    invalid_instance = self.temp_dir / "invalid.json"
    invalid_instance.write_text('"forty-two"', encoding="utf-8")

    ret_ok = cw.main(
        ["validate", "--schema", str(schema_path), str(valid_instance)]
    )
    self.assertEqual(ret_ok, 0)

    ret_invalid = cw.main(
        ["validate", "--schema", str(schema_path), str(invalid_instance)]
    )
    self.assertEqual(ret_invalid, 2)


if __name__ == "__main__":
  unittest.main()
