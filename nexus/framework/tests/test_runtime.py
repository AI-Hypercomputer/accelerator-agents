"""Unit tests for standalone runtime operations."""

import json
import pathlib
import shutil
import tempfile
import unittest

try:
  from accelerator_agents.tpu_nexus.framework import runtime, utils
except ImportError:
  try:
    from .. import runtime, utils
  except ImportError:
    import runtime
    import utils

Path = pathlib.Path


class TestRuntime(unittest.TestCase):
  """Unit tests for runtime namespace allocation, descriptors, and validations."""

  def setUp(self):
    super().setUp()
    self.temp_dir = Path(tempfile.mkdtemp())
    self.workspace_dir = self.temp_dir / "workspace"
    self.workspace_dir.mkdir()

    # Set up installed package structure
    self.pkg_root = self.workspace_dir / ".coworker" / "test-pkg"
    self.pkg_root.mkdir(parents=True)
    (self.pkg_root / "environment.json").write_text(
        json.dumps({"revision": 1, "target": "local"}), encoding="utf-8"
    )
    (self.pkg_root / "schemas").mkdir()
    (self.pkg_root / "schemas" / "out.json").write_text(
        json.dumps({
            "type": "object",
            "properties": {"status": {"type": "string"}},
            "required": ["status"],
        }),
        encoding="utf-8",
    )

  def tearDown(self):
    shutil.rmtree(self.temp_dir, ignore_errors=True)
    super().tearDown()

  def test_start_run_and_namespace(self):
    doc = runtime.start_run(self.workspace_dir, "test-pkg")
    self.assertIn("run_id", doc)
    self.assertEqual(doc["package"], "test-pkg")
    self.assertEqual(doc["environment_revision"], 1)

    run_dir = self.pkg_root / "runs" / doc["run_id"]
    self.assertTrue(run_dir.is_dir())
    self.assertTrue((run_dir / "artifacts").is_dir())
    self.assertTrue((run_dir / "run.json").is_file())

  def test_describe_artifact(self):
    run_doc = runtime.start_run(self.workspace_dir, "test-pkg")
    run_id = run_doc["run_id"]

    artifact_file = (
        self.pkg_root / "runs" / run_id / "artifacts" / "sub" / "output.json"
    )
    artifact_file.parent.mkdir(parents=True, exist_ok=True)
    artifact_file.write_text(json.dumps({"status": "ok"}), encoding="utf-8")

    desc = runtime.describe_artifact(
        workspace=self.workspace_dir,
        package="test-pkg",
        run_id=run_id,
        file="sub/output.json",
        schema="schemas/out.json",
        media_type="application/json",
    )

    self.assertEqual(desc["run_id"], run_id)
    self.assertEqual(desc["media_type"], "application/json")
    self.assertEqual(desc["schema"], "schemas/out.json")
    self.assertEqual(
        desc["uri"],
        f"workspace://test-pkg/runs/{run_id}/artifacts/sub/output.json",
    )
    self.assertEqual(desc["sha256"], utils.compute_digest(artifact_file))

  def test_runtime_cli(self):
    run_doc = runtime.start_run(self.workspace_dir, "test-pkg")
    run_id = run_doc["run_id"]

    valid_instance = self.temp_dir / "valid.json"
    valid_instance.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    schema_path = self.pkg_root / "schemas" / "out.json"

    ret = runtime.main(
        ["validate", "--schema", str(schema_path), str(valid_instance)]
    )
    self.assertEqual(ret, 0)


if __name__ == "__main__":
  unittest.main()
