"""MCP server for MaxCode API."""

import dataclasses
import json
import logging
import os
from typing import Optional

from absl import app
from interface import api
from mcp.server import fastmcp

logging.basicConfig(level=logging.INFO)
mcp = fastmcp.FastMCP("MaxCode API")


@mcp.tool()
async def convert_code(
    source_path: str,
    destination: str,
    api_key: str,
    model_name: Optional[str] = None,
    validate: bool = True,
    target: str = "jax",
) -> str:
  """Converts PyTorch code to a JAX-family target.

  Args:
      source_path: Path to PyTorch file or directory.
      destination: Directory to save results.
      api_key: Google AI API key.
      model_name: Optional model name.
      validate: Whether to run validation.
      target: Conversion target — "jax" (default) or "maxtext".
  """
  config = api.ConvertConfig(
      source_path=source_path,
      destination=destination,
      api_key=api_key,
      model_name=model_name,
      validate=validate,
      target=target,
  )
  try:
    result = api.convert(config)
    payload = {
        "dest_path": result.dest_path,
        "mapping_path": result.mapping_path,
        "original_source_dir": result.original_source_dir,
        "validation_path": result.validation_path,
        "verification_scorecard_path": result.verification_scorecard_path,
        "verification_summary": result.verification_summary,
    }
    if result.maxtext_artifacts is not None:
      payload["maxtext_artifacts"] = dataclasses.asdict(
          result.maxtext_artifacts
      )
    return json.dumps(payload)
  except Exception as e:
    logging.exception("Error in convert_code tool")
    return json.dumps({"error": str(e)})


@mcp.tool()
async def verify_code(
    source_code: str,
    jax_code: str,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
) -> str:
  """Verifies converted JAX code.

  Args:
      source_code: Original PyTorch code.
      jax_code: Converted JAX code.
      api_key: Optional API key for correctness check.
      model_name: Optional model name.
  """
  config = api.VerifyConfig(
      source_code=source_code,
      jax_code=jax_code,
      api_key=api_key,
      model_name=model_name,
  )
  try:
    report = api.verify(config)
    return json.dumps({
        "completeness": report.completeness,
        "correctness": report.correctness,
        "overall": report.overall,
    })
  except Exception as e:
    logging.exception("Error in verify_code tool")
    return json.dumps({"error": str(e)})


def main(argv):
  del argv  # Unused.
  mcp.run()


if __name__ == "__main__":
  app.run(main)
