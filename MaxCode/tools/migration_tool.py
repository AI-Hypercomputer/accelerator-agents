"""Migration tool for ADK."""

import datetime
import json
import logging
import os
import pathlib
import shutil

import models
from agents.migration import primary_agent
from google.adk.tools.function_tool import FunctionTool


MAPPING_FILE_NAME = "mapping.json"
ORIGINAL_SOURCE_DIR_NAME = "original_source"


def _write_artifact(output_path: pathlib.Path, code: str) -> None:
  """Safely writes code to output_path, creating directories as needed."""
  if output_path.parent:
    try:
      output_path.parent.mkdir(parents=True)
    except FileExistsError:
      pass
  output_path.write_text(code, encoding="utf-8")


def convert_code(
    source_path: str,
    destination: str,
    api_key: str,
    model_name: str | None = None,
    validate: bool = True,
) -> str:
  """Converts PyTorch code to JAX and saves it to the destination.

  Args:
    source_path: The path to the Python file or directory to migrate.
    destination: The directory where the migrated files should be saved.
    api_key: The Google AI API key to use for migration.
    model_name: The Gemini model to use for migration.
    validate: Whether to run validation and repair after conversion.

  Returns:
    A JSON string containing the destination paths for subsequent steps.
  """
  logging.info(
      "convert_code called with source_path=%s, destination=%s, api_key=%s",
      source_path,
      destination,
      api_key,
  )
  if source_path is None:
    return json.dumps({"error": "source_path is None"})
  if destination is None:
    return json.dumps({"error": "destination is None"})
  if api_key is None:
    return json.dumps({"error": "api_key is None"})

  workspace_dir = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
  abs_path = source_path
  if not os.path.isabs(source_path) and workspace_dir:
    abs_path = os.path.join(workspace_dir, source_path)

  logging.info("Attempting to convert %s to JAX...", abs_path)

  model_kwargs = {"api_key": api_key}
  if model_name:
    model_kwargs["model_name"] = model_name
  model = models.GeminiTool(**model_kwargs)
  agent = primary_agent.PrimaryAgent(model, api_key=api_key, validate=validate)
  results = agent.run(abs_path)

  logging.info("Writing converted files to: %s", destination)
  timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
  dest_path = pathlib.Path(destination) / timestamp
  logging.info("Outputting to timestamped directory: %s", dest_path)
  p = pathlib.Path(abs_path)
  is_dir = p.is_dir()

  # Copy original source to destination for user reference and evaluation
  source_copy_dir = dest_path / ORIGINAL_SOURCE_DIR_NAME
  try:
    if is_dir:
      shutil.copytree(abs_path, source_copy_dir, dirs_exist_ok=True)
    else:
      source_copy_dir.mkdir(parents=True, exist_ok=True)
      shutil.copy2(abs_path, source_copy_dir / p.name)
  except OSError as e:
    logging.warning("Failed to copy source files to destination: %s", e)
    return json.dumps({
        "error": f"Failed to copy source files to destination: {e}",
    })

  # Handle two result formats:
  # - Merge path (directory): keys are "model" and optionally "utils"
  # - Single-file / legacy path: keys are file paths
  is_merge_result = "model" in results
  written_files = []
  mapping_log = []

  if is_merge_result:
    # Write model output
    model_output = dest_path / "model_jax.py"
    _write_artifact(model_output, results["model"])
    written_files.append(model_output)
    mapping_log.append({
        "source_file": abs_path,
        "generated_file": str(model_output),
        "component": "model",
        "status": "success",
    })
    # Write utils output (if present)
    if "utils" in results:
      utils_output = dest_path / "utils_jax.py"
      _write_artifact(utils_output, results["utils"])
      written_files.append(utils_output)
      mapping_log.append({
          "source_file": abs_path,
          "generated_file": str(utils_output),
          "component": "utils",
          "status": "success",
      })
  else:
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

  # Ensure original_source is importable by adding __init__.py files
  for dirpath, _, _ in os.walk(source_copy_dir):
    init_py = pathlib.Path(dirpath) / "__init__.py"
    if not init_py.exists():
      _write_artifact(init_py, "")

  mapping_path = dest_path / MAPPING_FILE_NAME
  with mapping_path.open("w", encoding="utf-8") as f:
    json.dump(mapping_log, f, indent=2)

  response = {
      "dest_path": str(dest_path),
      "mapping_path": str(mapping_path),
      "original_source_dir": str(source_copy_dir),
  }

  # Write validation results if validation was enabled and produced results
  validation_results = agent.get_validation_results()
  if validate and validation_results:
    validation_path = dest_path / "validation_results.json"
    with validation_path.open("w", encoding="utf-8") as f:
      json.dump(validation_results, f, indent=2)
    response["validation_path"] = str(validation_path)

  # Auto-verify converted files
  try:
    from agents.migration.verification_agent import VerificationAgent
    verifier = VerificationAgent()
    scorecard = {}

    if is_merge_result:
      # Use cached merge result from PrimaryAgent to avoid re-running merge
      cached_merge = agent.get_merge_result()
      if cached_merge:
        source_code_map = {"model": cached_merge.model_code}
        if cached_merge.utility_code:
          source_code_map["utils"] = cached_merge.utility_code
      else:
        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
          source_code_map = {"model": f.read()}

      for component, jax_code in results.items():
        if component in source_code_map:
          vr = verifier.verify(source_code_map[component], jax_code)
          scorecard[component] = {
              "completeness": vr.completeness,
              "overall": vr.overall,
          }
    else:
      for file_path, jax_code in results.items():
        try:
          with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            src = f.read()
          vr = verifier.verify(src, jax_code)
          scorecard[file_path] = {
              "completeness": vr.completeness,
              "overall": vr.overall,
          }
        except OSError:
          pass

    if scorecard:
      scorecard_path = dest_path / "verification_scorecard.json"
      with scorecard_path.open("w", encoding="utf-8") as f:
        json.dump(scorecard, f, indent=2)
      response["verification_scorecard_path"] = str(scorecard_path)
      response["verification_summary"] = {
          k: v["overall"] for k, v in scorecard.items()
      }
  except Exception as e:
    logging.warning("Auto-verification failed: %s", e)

  return json.dumps(response)


convert_code_tool = FunctionTool(convert_code)
