#!/usr/bin/env python3
"""Unit tests for run_in_pod.sh."""

import os
import shutil
import subprocess
import tempfile
import unittest

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_PATH = os.path.join(TOOLS_DIR, "run_in_pod.sh")


class TestRunInPod(unittest.TestCase):
  """Tests CLI argument validation and dry-run execution of run_in_pod.sh."""

  def setUp(self):
    self.test_dir = tempfile.mkdtemp()
    self.problem_dir = os.path.join(self.test_dir, "test_problem")
    os.makedirs(self.problem_dir, exist_ok=True)
    with open(
      os.path.join(self.problem_dir, "baseline.py"), "w", encoding="utf-8"
    ) as f:
      f.write("# baseline implementation\n")

  def tearDown(self):
    shutil.rmtree(self.test_dir, ignore_errors=True)

  def run_script(self, args: list[str], env: dict[str, str] | None = None):
    run_env = os.environ.copy()
    if env:
      run_env.update(env)
    return subprocess.run(
      [SCRIPT_PATH] + args,
      capture_output=True,
      text=True,
      env=run_env,
      check=False,
    )

  def test_missing_all_arguments_fails(self):
    res = self.run_script([])
    self.assertNotEqual(res.returncode, 0)
    self.assertIn("Missing required problem directory", res.stderr)

  def test_missing_problem_dir_flag_value_fails(self):
    res = self.run_script(["--problem-dir"])
    self.assertNotEqual(res.returncode, 0)
    self.assertIn("--problem-dir requires a path argument", res.stderr)

  def test_nonexistent_problem_directory_fails(self):
    res = self.run_script(
      ["--dry-run", "--problem-dir", "/tmp/nonexistent_problem_path_12345"]
    )
    self.assertNotEqual(res.returncode, 0)
    self.assertIn("Problem directory not found", res.stderr)

  def test_problem_directory_missing_entry_file_fails(self):
    empty_problem_dir = os.path.join(self.test_dir, "empty_problem")
    os.makedirs(empty_problem_dir, exist_ok=True)
    res = self.run_script(["--dry-run", "--problem-dir", empty_problem_dir])
    self.assertNotEqual(res.returncode, 0)
    self.assertIn("No Python entry file found", res.stderr)

  def test_dry_run_formatting_with_flag(self):
    out_dir = os.path.join(self.test_dir, "out")
    res = self.run_script(
      [
        "--dry-run",
        "--problem-dir",
        self.problem_dir,
        "--output-dir",
        out_dir,
        "--model",
        "claude-3-opus",
        "--tpu-port",
        "5555",
        "--timeout",
        "2h",
      ]
    )
    self.assertEqual(res.returncode, 0, msg=res.stderr)
    output = res.stdout
    self.assertIn("[dry-run] Checking node and claude CLI in PATH", output)
    self.assertIn(
      "[dry-run] Checking TPU server health on 127.0.0.1:5555/health", output
    )
    self.assertIn("tpu_server.py --port=5555", output)
    self.assertIn("install.py --with-guard", output)
    self.assertIn("Pointing tpu_config.json at 127.0.0.1:5555", output)
    self.assertIn("claude -p", output)
    self.assertIn("--model claude-3-opus", output)
    self.assertIn("export_run_metrics.py --run_dir", output)

  def test_dry_run_with_positional_problem_path(self):
    res = self.run_script(["--dry-run", self.problem_dir])
    self.assertEqual(res.returncode, 0, msg=res.stderr)
    self.assertIn("[dry-run] Command sequence:", res.stdout)
    self.assertIn("claude -p", res.stdout)

  def test_vertex_ai_detection(self):
    res = self.run_script(
      ["--dry-run", "--problem-dir", self.problem_dir],
      env={"CLAUDE_CODE_USE_VERTEX": "1"},
    )
    self.assertEqual(res.returncode, 0, msg=res.stderr)
    self.assertIn("Detected Vertex AI configuration", res.stdout)

  def test_vertex_ai_disabled(self):
    res = self.run_script(
      ["--dry-run", "--problem-dir", self.problem_dir],
      env={"CLAUDE_CODE_USE_VERTEX": "0"},
    )
    self.assertEqual(res.returncode, 0, msg=res.stderr)
    self.assertNotIn("Detected Vertex AI configuration", res.stdout)


if __name__ == "__main__":
  unittest.main()
