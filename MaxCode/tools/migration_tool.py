"""Migration tool for ADK."""

import datetime
import json
import logging
import os
import pathlib
import shutil
import subprocess
import sys

import models
from agents.migration import primary_agent
from evaluation import make_data
from tools import evaluation_tool
from google.adk.tools.function_tool import FunctionTool


EVAL_DIR_NAME = "evaluation"
DATA_SUBDIR_NAME = "data"
TESTS_SUBDIR_NAME = "tests"
CONFIG_FILE_NAME = "model_configs.json"
MAPPING_FILE_NAME = "mapping.json"


def _write_artifact(output_path: pathlib.Path, code: str) -> None:
  """Safely writes code to output_path, creating directories as needed."""
  if output_path.parent:
    try:
      output_path.parent.mkdir(parents=True)
    except FileExistsError:
      pass
  output_path.write_text(code, encoding="utf-8")


def migrate_module(path: str, destination: str, api_key: str) -> str:
  """Migrates the module at path to destination.

  Args:
    path: The path to the Python file or directory to migrate.
    destination: The directory where the migrated files should be saved.
    api_key: The Google AI API key to use for migration.

  Returns:
    A string indicating the result of the migration, or a string
    representation of the migrated code if destination is not provided.
  """
  logging.info(
      "migrate_module called with path=%s, destination=%s, api_key=%s",
      path,
      destination,
      api_key,
  )
  if path is None:
    return "Error: path is None"
  if destination is None:
    return "Error: destination is None"
  if api_key is None:
    return "Error: api_key is None"

  workspace_dir = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
  abs_path = path
  if not os.path.isabs(path) and workspace_dir:
    abs_path = os.path.join(workspace_dir, path)

  logging.info("Attempting to migrate %s to %s...", abs_path, destination)

  model = models.GeminiTool(
      model_name=models.GeminiModel.GEMINI_2_5_PRO, api_key=api_key
  )
  agent = primary_agent.PrimaryAgent(model, api_key=api_key)
  results = agent.run(abs_path)
  if destination:
    logging.info("Writing converted files to: %s", destination)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dest_path = pathlib.Path(destination) / timestamp
    logging.info("Outputting to timestamped directory: %s", dest_path)
    p = pathlib.Path(abs_path)
    is_dir = p.is_dir()

    # Copy original source to destination for user reference
    try:
      source_copy_dir = dest_path / "original_source"
      if is_dir:
        shutil.copytree(abs_path, source_copy_dir, dirs_exist_ok=True)
      else:
        source_copy_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(abs_path, source_copy_dir / p.name)
    except OSError as e:
      logging.warning("Failed to copy source files to destination: %s", e)

    written_files = []
    mapping_log = []
    for file_path, code in results.items():
      if is_dir:
        relative_path = pathlib.Path(file_path).relative_to(p)
      else:
        relative_path = pathlib.Path(file_path).name
      output_path = dest_path / relative_path
      _write_artifact(output_path, code)
      written_files.append(output_path)
      mapping_log.append({
          "source_file": file_path,
          "generated_file": str(output_path),
          "status": "success",
      })

    # Create __init__.py files for all directories containing migrated files.
    dirs_in_results = set(f.parent for f in written_files)
    init_paths_to_create = set()
    for d in dirs_in_results:
      current_d = d
      while current_d and (
          current_d == dest_path or dest_path in current_d.parents
      ):
        init_py = current_d / "__init__.py"
        init_paths_to_create.add(init_py)
        if current_d == dest_path:
          break
        current_d = current_d.parent

    for init_py in init_paths_to_create:
      if not init_py.exists():
        _write_artifact(init_py, "")

    mapping_path = dest_path / "mapping.json"
    with mapping_path.open("w", encoding="utf-8") as f:
      json.dump(mapping_log, f, indent=2)

    # --- Evaluation Steps ---
    logging.info("Starting evaluation steps...")
    eval_dir = dest_path / EVAL_DIR_NAME
    config_path = eval_dir / CONFIG_FILE_NAME
    data_dir = eval_dir / DATA_SUBDIR_NAME
    tests_dir = eval_dir / TESTS_SUBDIR_NAME

    try:
      # Determine PyTorch source directory for config and data generation
      pytorch_dir = (
          abs_path if os.path.isdir(abs_path) else os.path.dirname(abs_path)
      )

      # 1. Generate model configurations
      logging.info("Generating model configurations...")
      eval_dir.mkdir(exist_ok=True)
      config_result = evaluation_tool.generate_model_configs(
          pytorch_dir, str(config_path), api_key
      )
      logging.info(config_result)

      # 2. Generate oracle data
      logging.info("Generating oracle data...")
      data_dir.mkdir(exist_ok=True)
      make_data.generate_data(pytorch_dir, str(data_dir), str(config_path))
      logging.info("Oracle data generated in %s.", data_dir)

      # 3. Generate equivalence tests and run them
      logging.info("Generating and running equivalence tests...")
      tests_dir.mkdir(exist_ok=True)
      test_run_results = {}
      for item in mapping_log:
        if item["status"] == "success":
          jax_file = item["generated_file"]
          torch_file = item["source_file"]
          module_stem = pathlib.Path(jax_file).stem
          test_file_name = f"test_{module_stem}.py"
          output_test_path = tests_dir / test_file_name

          try:
            test_gen_result = evaluation_tool.generate_equivalence_tests(
                jax_file, torch_file, str(output_test_path), api_key
            )
            logging.info(test_gen_result)

            if not test_gen_result.startswith("Successfully generated"):
              test_run_results[test_file_name] = (
                  "ERROR (test generation failed)"
              )
              continue

            # Run equivalence test
            pickle_file = data_dir / f"{module_stem}.pkl"
            if output_test_path.exists() and pickle_file.exists():
              logging.info("Running equivalence test: %s", output_test_path)
              python_executable = sys.executable or "python"
              cmd = [
                  python_executable,
                  str(output_test_path),
                  f"--pickle_path={pickle_file}",
              ]
              result = subprocess.run(
                  cmd,
                  capture_output=True,
                  text=True,
                  check=False,
                  timeout=60,
              )
              if result.returncode == 0:
                test_run_results[test_file_name] = "PASSED"
                logging.info("Test %s PASSED", test_file_name)
              else:
                test_run_results[test_file_name] = "FAILED"
                logging.info(
                    "Test %s FAILED\nSTDOUT:%s\nSTDERR:%s",
                    test_file_name,
                    result.stdout,
                    result.stderr,
                )
            else:
              test_run_results[test_file_name] = (
                  "SKIPPED (missing test or data file)"
              )
              logging.info(
                  "Skipping test run for %s, test or data file not found.",
                  test_file_name,
              )
          except (subprocess.TimeoutExpired, OSError) as e:
            test_run_results[test_file_name] = (
                f"ERROR ({type(e).__name__}: {e})"
            )
            logging.exception(
                "Error during test generation or execution for %s: %s",
                test_file_name,
                e,
            )
    except (OSError, ValueError, RuntimeError) as e:
      logging.exception("Error during evaluation steps: %s", e)
      return (
          f"Successfully migrated module {path} to {dest_path}, but failed"
          f" during evaluation steps: {e}"
      )

    results_summary = f"""Successfully migrated module {path} to {dest_path}.
Evaluation artifacts generated in {eval_dir}.
Equivalence test results:
""" + "\n".join([f"- {k}: {v}" for k, v in test_run_results.items()])

    try:
      shutil.copy2("/tmp/agent_server.log", dest_path / "migration.log")
      logging.info("Copied /tmp/agent_server.log to %s", dest_path)
    except OSError as e:
      logging.warning("Failed to copy log file to destination: %s", e)

    return results_summary
  else:
    return str(results)


migrate_module_tool = FunctionTool(migrate_module)
