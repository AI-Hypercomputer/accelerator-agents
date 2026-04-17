"""High-level Python API (Entry point for ALL front-ends)."""

from dataclasses import dataclass
import datetime
import json
import logging
import os
import pathlib
import shutil
from typing import Dict, List, Literal, Optional, Tuple

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
  target: Literal["jax", "maxtext"] = "jax"


@dataclass
class MaxTextArtifacts:
  """Paths to artifacts produced by the MaxText conversion path."""

  config_yaml_path: str
  layers_py_path: Optional[str] = None
  ckpt_converter_path: Optional[str] = None
  decoder_block: str = "default"


@dataclass
class ConversionResult:
  """Result of the conversion process."""

  dest_path: str
  mapping_path: str
  original_source_dir: str
  validation_path: Optional[str] = None
  verification_scorecard_path: Optional[str] = None
  verification_summary: Optional[Dict[str, float]] = None
  maxtext_artifacts: Optional[MaxTextArtifacts] = None


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


def convert(config: ConvertConfig) -> ConversionResult:
  """Converts PyTorch code to a JAX-family target.

  Args:
      config: The conversion configuration.

  Returns:
      A ConversionResult object.
  """
  logging.info(
      "convert called with source_path=%s, destination=%s, target=%s",
      config.source_path,
      config.destination,
      config.target,
  )

  abs_path = _resolve_source_path(config.source_path)
  logging.info("Attempting to convert %s to %s...", abs_path, config.target)

  model = _build_model(config.api_key, config.model_name)
  agent = primary_agent.PrimaryAgent(
      model, api_key=config.api_key, validate=config.validate,
      target=config.target,
  )
  results = agent.run(abs_path)

  return _persist_conversion(
      results=results,
      abs_path=abs_path,
      destination=config.destination,
      validate=config.validate,
      agent=agent,
      target=config.target,
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
    model = _build_model(config.api_key, config.model_name)

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


# ─────────────────────────────────────────────────────────────────────
# Internal helpers (filesystem plumbing, not part of the public API)
# ─────────────────────────────────────────────────────────────────────


def _resolve_source_path(source_path: str) -> str:
  """Resolves source_path to an absolute path, honoring BUILD_WORKSPACE_DIRECTORY."""
  workspace_dir = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
  if not os.path.isabs(source_path) and workspace_dir:
    return os.path.join(workspace_dir, source_path)
  return source_path


def _build_model(
    api_key: str, model_name: Optional[str]
) -> "models.GeminiTool":
  """Builds a GeminiTool with the given credentials and model name."""
  model_kwargs = {"api_key": api_key}
  if model_name:
    model_kwargs["model_name"] = model_name
  return models.GeminiTool(**model_kwargs)


def _write_artifact(output_path: pathlib.Path, code: str) -> None:
  """Safely writes code to output_path, creating directories as needed."""
  if output_path.parent:
    try:
      output_path.parent.mkdir(parents=True)
    except FileExistsError:
      pass
  output_path.write_text(code, encoding="utf-8")


def _persist_conversion(
    *,
    results: Dict[str, str],
    abs_path: str,
    destination: str,
    validate: bool,
    agent: primary_agent.PrimaryAgent,
    target: str = "jax",
) -> ConversionResult:
  """Writes all conversion artifacts to disk and returns a ConversionResult.

  Handles the timestamped output directory, original-source copy, converted
  file writes, __init__.py stubs, mapping.json, validation_results.json, and
  best-effort auto-verification scorecard. For `target="maxtext"` it
  additionally writes the MaxText artifact set (YAML overlay + optional
  layers and checkpoint converter).
  """
  logging.info("Writing converted files to: %s", destination)
  timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
  dest_path = pathlib.Path(destination) / timestamp
  logging.info("Outputting to timestamped directory: %s", dest_path)
  p = pathlib.Path(abs_path)
  is_dir = p.is_dir()

  source_copy_dir = _copy_original_source(abs_path, dest_path, p, is_dir)

  maxtext_artifacts: Optional[MaxTextArtifacts] = None
  written_files: List[pathlib.Path] = []
  mapping_log: List[Dict] = []
  is_merge_result = False

  if target == "maxtext":
    maxtext_artifacts, written_files, mapping_log = _write_maxtext_artifacts(
        agent=agent,
        dest_path=dest_path,
        source_path=abs_path,
    )
  else:
    written_files, mapping_log, is_merge_result = _write_converted_files(
        results, dest_path, p, abs_path, is_dir
    )
    _write_init_py_stubs(written_files, dest_path, source_copy_dir)

  mapping_path = dest_path / "mapping.json"
  with mapping_path.open("w", encoding="utf-8") as f:
    json.dump(mapping_log, f, indent=2)

  validation_path = _write_validation_results(agent, dest_path, validate)

  scorecard: Dict[str, Dict[str, float]] = {}
  scorecard_path: Optional[pathlib.Path] = None
  if target != "maxtext":
    scorecard, scorecard_path = _auto_verify(
        results, agent, abs_path, dest_path, is_merge_result
    )

  return ConversionResult(
      dest_path=str(dest_path),
      mapping_path=str(mapping_path),
      original_source_dir=str(source_copy_dir),
      validation_path=str(validation_path) if validation_path else None,
      verification_scorecard_path=(
          str(scorecard_path) if scorecard_path else None
      ),
      verification_summary=(
          {k: v["overall"] for k, v in scorecard.items()} if scorecard else None
      ),
      maxtext_artifacts=maxtext_artifacts,
  )


def _copy_original_source(
    abs_path: str,
    dest_path: pathlib.Path,
    p: pathlib.Path,
    is_dir: bool,
) -> pathlib.Path:
  """Copies the original source into dest_path/original_source/ for reference."""
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
  return source_copy_dir


def _write_converted_files(
    results: Dict[str, str],
    dest_path: pathlib.Path,
    p: pathlib.Path,
    abs_path: str,
    is_dir: bool,
) -> Tuple[List[pathlib.Path], List[Dict], bool]:
  """Writes converted JAX files to dest_path. Returns (written_files, mapping_log, is_merge_result)."""
  is_merge_result = "model" in results
  written_files: List[pathlib.Path] = []
  mapping_log: List[Dict] = []

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

  return written_files, mapping_log, is_merge_result


def _write_maxtext_artifacts(
    *,
    agent: primary_agent.PrimaryAgent,
    dest_path: pathlib.Path,
    source_path: str,
) -> Tuple[Optional[MaxTextArtifacts], List[pathlib.Path], List[Dict]]:
  """Writes the MaxText artifact set under dest_path.

  Returns:
    (artifacts, written_files, mapping_log). `artifacts` is None when the
    agent did not produce a MaxText result (e.g. the input path was invalid).
  """
  result = agent.get_maxtext_result()
  if result is None:
    logging.warning("target='maxtext' but no MaxTextRunResult was produced")
    return None, [], []

  written_files: List[pathlib.Path] = []
  mapping_log: List[Dict] = []

  config_yaml_path = (
      dest_path / "MaxText" / "configs" / "models" / f"{result.model_name}.yml"
  )
  _write_artifact(config_yaml_path, result.config_yaml)
  written_files.append(config_yaml_path)
  mapping_log.append({
      "source_file": source_path,
      "generated_file": str(config_yaml_path),
      "component": "maxtext_config",
      "decoder_block": result.decoder_block,
      "status": "success",
  })

  layers_py_path: Optional[pathlib.Path] = None
  if result.layers_py:
    layers_py_path = (
        dest_path / "MaxText" / "layers" / f"{result.model_name}.py"
    )
    _write_artifact(layers_py_path, result.layers_py)
    written_files.append(layers_py_path)
    mapping_log.append({
        "source_file": source_path,
        "generated_file": str(layers_py_path),
        "component": "maxtext_layers",
        "decoder_block": result.decoder_block,
        "status": "success",
    })

  ckpt_converter_path: Optional[pathlib.Path] = None
  if result.ckpt_converter_py:
    ckpt_converter_path = (
        dest_path / "utils" / f"convert_{result.model_name}_ckpt.py"
    )
    _write_artifact(ckpt_converter_path, result.ckpt_converter_py)
    written_files.append(ckpt_converter_path)
    mapping_log.append({
        "source_file": source_path,
        "generated_file": str(ckpt_converter_path),
        "component": "maxtext_ckpt_converter",
        "decoder_block": result.decoder_block,
        "status": "success",
    })

  artifacts = MaxTextArtifacts(
      config_yaml_path=str(config_yaml_path),
      layers_py_path=str(layers_py_path) if layers_py_path else None,
      ckpt_converter_path=(
          str(ckpt_converter_path) if ckpt_converter_path else None
      ),
      decoder_block=result.decoder_block,
  )
  return artifacts, written_files, mapping_log


def _write_init_py_stubs(
    written_files: List[pathlib.Path],
    dest_path: pathlib.Path,
    source_copy_dir: pathlib.Path,
) -> None:
  """Creates empty __init__.py stubs in every ancestor directory of written files."""
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


def _write_validation_results(
    agent: primary_agent.PrimaryAgent,
    dest_path: pathlib.Path,
    validate: bool,
) -> Optional[pathlib.Path]:
  """Writes validation_results.json when validation ran and produced results."""
  validation_results = agent.get_validation_results()
  if not (validate and validation_results):
    return None
  validation_path = dest_path / "validation_results.json"
  with validation_path.open("w", encoding="utf-8") as f:
    json.dump(validation_results, f, indent=2)
  return validation_path


def _auto_verify(
    results: Dict[str, str],
    agent: primary_agent.PrimaryAgent,
    abs_path: str,
    dest_path: pathlib.Path,
    is_merge_result: bool,
) -> Tuple[Dict[str, Dict[str, float]], Optional[pathlib.Path]]:
  """Runs best-effort verification over converted files. Returns (scorecard, path or None)."""
  scorecard: Dict[str, Dict[str, float]] = {}
  scorecard_path: Optional[pathlib.Path] = None
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

  return scorecard, scorecard_path
