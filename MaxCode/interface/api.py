"""High-level Python API (Entry point for ALL front-ends)."""

from dataclasses import dataclass
import datetime
import json
import logging
import os
import pathlib
import shutil
from typing import Dict, List, Optional

import models
from agents.migration import primary_agent
from agents.migration.verification_agent import VerificationAgent


@dataclass
class ConvertConfig:
  """Configuration for code conversion."""

  source_path: str
  destination: str
  api_key: str
  model_name: Optional[str] = None
  validate: bool = True


@dataclass
class ConversionResult:
  """Result of the conversion process."""

  dest_path: str
  mapping_path: str
  original_source_dir: str
  validation_path: Optional[str] = None
  verification_scorecard_path: Optional[str] = None
  verification_summary: Optional[Dict[str, float]] = None


@dataclass
class VerifyConfig:
  """Configuration for code verification."""

  source_code: str
  jax_code: str
  api_key: Optional[str] = None
  model_name: Optional[str] = None


@dataclass
class VerificationReport:
  """Report of the verification process."""

  completeness: Dict
  overall: float
  correctness: Optional[Dict] = None


def _write_artifact(output_path: pathlib.Path, code: str) -> None:
  """Safely writes code to output_path, creating directories as needed."""
  if output_path.parent:
    try:
      output_path.parent.mkdir(parents=True)
    except FileExistsError:
      pass
  output_path.write_text(code, encoding="utf-8")


def convert(config: ConvertConfig) -> ConversionResult:
  """Converts PyTorch code to JAX.

  Args:
      config: The conversion configuration.

  Returns:
      A ConversionResult object.
  """
  logging.info(
      "convert called with source_path=%s, destination=%s",
      config.source_path,
      config.destination,
  )

  workspace_dir = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
  abs_path = config.source_path
  if not os.path.isabs(config.source_path) and workspace_dir:
    abs_path = os.path.join(workspace_dir, config.source_path)

  logging.info("Attempting to convert %s to JAX...", abs_path)

  model_kwargs = {"api_key": config.api_key}
  if config.model_name:
    model_kwargs["model_name"] = config.model_name
  model = models.GeminiTool(**model_kwargs)
  agent = primary_agent.PrimaryAgent(
      model, api_key=config.api_key, validate=config.validate
  )
  results = agent.run(abs_path)

  logging.info("Writing converted files to: %s", config.destination)
  timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
  dest_path = pathlib.Path(config.destination) / timestamp
  logging.info("Outputting to timestamped directory: %s", dest_path)
  p = pathlib.Path(abs_path)
  is_dir = p.is_dir()

  # Copy original source to destination for user reference and evaluation
  source_copy_dir = dest_path / "original_source"
  try:
    if is_dir:
      shutil.copytree(abs_path, source_copy_dir, dirs_exist_ok=True)
    else:
      source_copy_dir.mkdir(parents=True, exist_ok=True)
      shutil.copy2(abs_path, source_copy_dir / p.name)
  except OSError as e:
    logging.warning("Failed to copy source files to destination: %s", e)
    raise RuntimeError(f"Failed to copy source files to destination: {e}")

  is_merge_result = "model" in results
  written_files = []
  mapping_log = []

  if is_merge_result:
    model_output = dest_path / "model_jax.py"
    _write_artifact(model_output, results["model"])
    written_files.append(model_output)
    mapping_log.append({
        "source_file": abs_path,
        "generated_file": str(model_output),
        "component": "model",
        "status": "success",
    })
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

  # Create __init__.py files
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

  for dirpath, _, _ in os.walk(source_copy_dir):
    init_py = pathlib.Path(dirpath) / "__init__.py"
    if not init_py.exists():
      _write_artifact(init_py, "")

  mapping_path = dest_path / "mapping.json"
  with mapping_path.open("w", encoding="utf-8") as f:
    json.dump(mapping_log, f, indent=2)

  validation_path = None
  validation_results = agent.get_validation_results()
  if config.validate and validation_results:
    validation_path = dest_path / "validation_results.json"
    with validation_path.open("w", encoding="utf-8") as f:
      json.dump(validation_results, f, indent=2)

  scorecard_path = None
  scorecard = {}
  try:
    verifier = VerificationAgent()
    if is_merge_result:
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
  except Exception as e:
    logging.warning("Auto-verification failed: %s", e)

  return ConversionResult(
      dest_path=str(dest_path),
      mapping_path=str(mapping_path),
      original_source_dir=str(source_copy_dir),
      validation_path=str(validation_path) if validation_path else None,
      verification_scorecard_path=str(scorecard_path) if scorecard_path else None,
      verification_summary={k: v["overall"] for k, v in scorecard.items()} if scorecard else None,
  )


def verify(config: VerifyConfig) -> VerificationReport:
  """Verifies converted JAX code against original PyTorch code.

  Args:
      config: The verification configuration.

  Returns:
      A VerificationReport object.
  """
  model = None
  if config.api_key:
    model_kwargs = {"api_key": config.api_key}
    if config.model_name:
      model_kwargs["model_name"] = config.model_name
    model = models.GeminiTool(**model_kwargs)

  verifier = VerificationAgent(model=model)
  vr = verifier.verify(
      config.source_code, config.jax_code, api_key=config.api_key
  )

  return VerificationReport(
      completeness=vr.completeness,
      correctness=vr.correctness,
      overall=vr.overall,
  )


class MigrationSession:
  """Handles multi-step migration workflows."""

  def __init__(self, config: ConvertConfig):
    self.config = config
    self.result: Optional[ConversionResult] = None

  def run(self) -> ConversionResult:
    """Runs the full migration workflow."""
    self.result = convert(self.config)
    return self.result
