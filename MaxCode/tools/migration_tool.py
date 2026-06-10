"""Migration tool for ADK — thin adapter over interface/api.py."""

import dataclasses
import json
import logging

from google.adk.tools.function_tool import FunctionTool

from interface import api


def convert_code(
    source_path: str,
    destination: str,
    api_key: str,
    model_name: str | None = None,
    validate: bool = True,
    target: str = "jax",
) -> str:
  """Converts PyTorch code to a JAX-family target and saves it to disk.

  Args:
    source_path: The path to the Python file or directory to migrate.
    destination: The directory where the migrated files should be saved.
    api_key: The Google AI API key to use for migration.
    model_name: The Gemini model to use for migration.
    validate: Whether to run validation and repair after conversion.
    target: Conversion target — "jax" (default) or "maxtext".

  Returns:
    A JSON string containing the destination paths for subsequent steps.
  """
  logging.info(
      "convert_code called with source_path=%s, destination=%s, target=%s",
      source_path,
      destination,
      target,
  )
  if source_path is None:
    return json.dumps({"error": "source_path is None"})
  if destination is None:
    return json.dumps({"error": "destination is None"})
  if api_key is None:
    return json.dumps({"error": "api_key is None"})

  try:
    config = api.ConvertConfig(
        source_path=source_path,
        destination=destination,
        api_key=api_key,
        model_name=model_name,
        validate=validate,
        target=target,
    )
    result = api.convert(config)
  except Exception as e:
    logging.exception("Error in convert_code tool")
    return json.dumps({"error": str(e)})

  response = {
      "dest_path": result.dest_path,
      "mapping_path": result.mapping_path,
      "original_source_dir": result.original_source_dir,
  }
  if result.validation_path:
    response["validation_path"] = result.validation_path
  if result.verification_scorecard_path:
    response["verification_scorecard_path"] = result.verification_scorecard_path
    response["verification_summary"] = result.verification_summary
  if result.maxtext_artifacts is not None:
    response["maxtext_artifacts"] = dataclasses.asdict(result.maxtext_artifacts)

  return json.dumps(response)


convert_code_tool = FunctionTool(convert_code)
