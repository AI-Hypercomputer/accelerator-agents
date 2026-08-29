"""Verification tool for ADK — scores PyTorch-to-JAX conversion quality."""

import json
import logging

from agents.migration.verification_agent import VerificationAgent
from google.adk.tools.function_tool import FunctionTool


def verify_conversion(
    source_path: str,
    output_path: str,
    api_key: str = "",
) -> str:
  """Verify quality of a PyTorch-to-JAX conversion.

  Computes a completeness score (AST-based) and optionally a correctness
  score (LLM-based, requires api_key). Returns JSON with both scores and
  an overall score.

  Args:
    source_path: Path to the original PyTorch source file.
    output_path: Path to the converted JAX output file.
    api_key: Optional Google AI API key for LLM-based correctness check.

  Returns:
    A JSON string with completeness, correctness, and overall scores.
  """
  logging.info(
      "verify_conversion called with source_path=%s, output_path=%s",
      source_path, output_path,
  )

  try:
    with open(source_path, "r", encoding="utf-8") as f:
      source_code = f.read()
  except OSError as e:
    return json.dumps({"error": f"Cannot read source file: {e}"})

  try:
    with open(output_path, "r", encoding="utf-8") as f:
      output_code = f.read()
  except OSError as e:
    return json.dumps({"error": f"Cannot read output file: {e}"})

  verifier = VerificationAgent()
  result = verifier.verify(
      source_code, output_code,
      api_key=api_key if api_key else None,
  )

  response = {
      "source_path": source_path,
      "output_path": output_path,
      "completeness": result.completeness,
      "overall": result.overall,
  }
  if result.correctness is not None:
    response["correctness"] = {
        "score": result.correctness["score"],
        "deviation_count": result.correctness["deviation_count"],
        "by_category": result.correctness["by_category"],
        "by_severity": result.correctness["by_severity"],
    }

  return json.dumps(response)


verify_conversion_tool = FunctionTool(verify_conversion)
