"""Primary orchestration agent for repository migration."""
import logging
import os
import re
import subprocess
import tempfile
from typing import Any, Tuple

import models
from agents import base
from agents import utils
from agents.migration import model_conversion_agent
from agents.migration import single_file_agent
from agents.migration.prompts import prompts
from rag import rag_agent

MAX_DEBUG_ITERATIONS = 10
logger = logging.getLogger(__name__)

def _strip_markdown_formatting(text: str) -> str:
  """Strips markdown and returns only the first python code block."""
  code_block_match = re.search(r"```(?:python)?\n?(.*?)\n?```", text, re.DOTALL)
  if code_block_match:
    return code_block_match.group(1).strip()
  return text


class PrimaryAgent(base.Agent):
  """Primary orchestration agent for repository migration."""

  def __init__(self, model: Any, api_key: str | None = None,
               validate: bool = True):
    """Initializes the agent."""
    super().__init__(
        model=model,
        agent_domain=utils.AgentDomain.MIGRATION,
        agent_type=utils.AgentType.PRIMARY,
    )
    self._model_ref = model
    self._validate = validate
    self._validation_results: dict[str, dict] = {}
    self._rag_agent = rag_agent.RAGAgent(
        model,
        embedding_model_name=models.EmbeddingModel.GEMINI_EMBEDDING_001,
        api_key=api_key,
    )
    self._single_file_agent = single_file_agent.PytorchToJaxSingleFileAgent(
        model, self._rag_agent
    )
    self._model_conversion_agent = model_conversion_agent.ModelConversionAgent(
        model, self._rag_agent
    )

  def _convert_file(self, pytorch_code: str, file_path: str) -> str:
    """Routes a file to the appropriate conversion agent."""
    if utils.is_model_file(pytorch_code, file_path):
      return self._model_conversion_agent.run(pytorch_code)
    return self._single_file_agent.run(pytorch_code)

  def _execute_test(
      self, pytorch_code: str, jax_code: str, test_code: str
  ) -> Tuple[bool, str]:
    """Executes the test script and returns success status and output."""
    with tempfile.TemporaryDirectory() as tempdir:
      torch_module_path = os.path.join(tempdir, "torch_module.py")
      jax_module_path = os.path.join(tempdir, "jax_module.py")
      test_script_path = os.path.join(tempdir, "test_script.py")

      with open(torch_module_path, "w") as f:
        f.write(pytorch_code)
      with open(jax_module_path, "w") as f:
        f.write(jax_code)
      with open(test_script_path, "w") as f:
        f.write(test_code)

      try:
        result = subprocess.run(
            ["python3", test_script_path],
            capture_output=True,
            text=True,
            check=True,
            cwd=tempdir,
            timeout=600,
        )
        return True, result.stdout
      except subprocess.CalledProcessError as e:
        return False, e.stderr

  def _validate_and_repair(self, pytorch_code: str, converted_code: str,
                           file_path: str) -> str:
    """Validates converted code and repairs deviations if found.

    Args:
      pytorch_code: The original PyTorch source code.
      converted_code: The converted JAX/Flax code.
      file_path: The file path (used as key for storing results).

    Returns:
      The final code (repaired if deviations were found, original otherwise).
    """
    validator = validation_agent.ValidationAgent(self._model_ref)
    deviations = validator.validate(pytorch_code, converted_code)
    logger.info("Validation of %s: found %d deviations",
                file_path, len(deviations))

    result = {
        "deviations_found": len(deviations),
        "deviations": deviations,
        "remaining_deviations_count": 0,
        "remaining_deviations": [],
    }

    if deviations:
      repaired_code = validator.repair(
          converted_code, deviations, pytorch_code=pytorch_code
      )
      remaining = validator.validate(pytorch_code, repaired_code)
      logger.info("Re-validation of %s: %d remaining deviations",
                  file_path, len(remaining))
      result["remaining_deviations_count"] = len(remaining)
      result["remaining_deviations"] = remaining
      self._validation_results[file_path] = result
      return repaired_code

    self._validation_results[file_path] = result
    return converted_code

  def get_validation_results(self) -> dict[str, dict]:
    """Returns validation results for all processed files.

    Returns:
      A dictionary mapping file paths to their validation results, each
      containing deviations_found, deviations, remaining_deviations_count,
      and remaining_deviations.
    """
    return self._validation_results

  def run(self, repo_path: str) -> dict[str, str]:
    """Orchestrates the migration of a repository from PyTorch to JAX.

    Args:
      repo_path: The path to the repository file or directory.
      context: Optional raw context to use instead of RAG retrieval.

    Returns:
      A dictionary mapping original file paths to converted JAX code.

    Raises:
      RuntimeError: If the code conversion and validation fails after
        `MAX_DEBUG_ITERATIONS` attempts.
    """
    if os.path.isfile(repo_path):
      with open(repo_path, "r", encoding="utf-8", errors="replace") as f:
        pytorch_code = f.read()
      logger.info("Converting %s ...", repo_path)
      converted_code = self._convert_file(pytorch_code, repo_path)
      if self._validate:
        converted_code = self._validate_and_repair(
            pytorch_code, converted_code, repo_path
        )
      return {repo_path: converted_code}
    except OSError:
      # If opening as a file fails, check if it's a directory.
      if not os.path.isdir(repo_path):
        return {
            repo_path: f"# Error: path {repo_path} is not a file or directory."
        }

    if not os.path.isdir(repo_path):
      return {
          repo_path: f"# Error: path {repo_path} is not a file or directory."
      }

      if context is None:
        rag_context_list = self._rag_agent.retrieve_context(
            pytorch_code, top_k=7
        )
        rag_context = "\n\n".join([
            f"File: {c['file']}\n```python\n{c['text']}\n```"
            for c in rag_context_list
        ])
      else:
        rag_context = context

      jax_code = _strip_markdown_formatting(
          self.generate(
              prompts.MIGRATE_MODULE_TO_JAX_PROMPT,
              {"pytorch_code": pytorch_code, "rag_context": rag_context},
          )
      )

      for i in range(MAX_DEBUG_ITERATIONS):
        logging.info("Starting testing iteration %d.", i)
        test_code = _strip_markdown_formatting(
            self.generate(
                prompts.EVALUATE_CODE_PROMPT,
                {"pytorch_code": pytorch_code, "jax_code": jax_code},
            )
        )

        if "NOTESTCASE" in test_code:
          print(
              "Test generation returned NOTESTCASE, assuming conversion is ok."
          )
          return {repo_path: jax_code}

        success, output = self._execute_test(pytorch_code, jax_code, test_code)

        if success:
          print(f"Validation successful after {i} debugging iterations.")
          logging.info(
              "Validation successful after %d debugging iterations.", i
          )
          return {repo_path: jax_code}
        else:
          traceback = output
          logging.error(
              "Validation failed on iteration %d. Traceback:\n%s", i, traceback
          )
          logging.info("Starting debug iteration %d.", i + 1)
          bug_analysis = self.generate(
              prompts.BUG_ANALYSIS_PROMPT,
              {
                  "pytorch_code": pytorch_code,
                  "jax_code": jax_code,
                  "test_code": test_code,
                  "traceback": traceback,
              },
          )
          print(f"Bug analysis:\n{bug_analysis}")
          logging.info("Bug analysis:\n%s", bug_analysis)
          jax_code = _strip_markdown_formatting(
              self.generate(
                  prompts.SELF_DEBUGGING_PROMPT,
                  {
                      "pytorch_code": pytorch_code,
                      "jax_code": jax_code,
                      "test_code": test_code,
                      "traceback": traceback,
                      "bug_analysis": bug_analysis,
                      "rag_context": rag_context,
                  },
              )
          )
          print(f"Attempting fix with new JAX code for iteration {i+1}.")
    for i, file_rel_path in enumerate(ordered_files, 1):
      file_path = os.path.join(repo_path, file_rel_path)
      logger.info("Converting file %d/%d: %s ...", i, len(ordered_files),
                  file_rel_path)
      with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        pytorch_code = f.read()
      converted_code = self._convert_file(pytorch_code, file_path)
      if self._validate:
        converted_code = self._validate_and_repair(
            pytorch_code, converted_code, file_path
        )
      converted_files[file_path] = converted_code

      raise RuntimeError(
          "Failed to convert and validate code after"
          f" {MAX_DEBUG_ITERATIONS} iterations."
      )
    elif os.path.isdir(repo_path):
      graph = utils.build_dependency_graph(repo_path)
      ordered_files = utils.topological_sort(graph)
      converted_files: dict[str, str] = {}

      for file_rel_path in ordered_files:
        file_path = os.path.join(repo_path, file_rel_path)
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
          pytorch_code = f.read()
        converted_code = self._convert_file(pytorch_code, file_path)
        converted_files[file_path] = converted_code
      return converted_files
    else:
      return {
          repo_path: f"# Error: path {repo_path} is not a file or directory."
      }
