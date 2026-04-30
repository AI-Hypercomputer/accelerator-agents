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
from rag_pipeline.retrieval import retrieval_service
from dotenv import load_dotenv

# Load .env file
load_dotenv()

MAX_DEBUG_ITERATIONS = 10


def _strip_markdown_formatting(text: str) -> str:
  """Strips markdown and returns only the first python code block."""
  code_block_match = re.search(r"```(?:python)?\n?(.*?)\n?```", text, re.DOTALL)
  if code_block_match:
    return code_block_match.group(1).strip()
  return text


class PrimaryAgent(base.Agent):
  """Primary orchestration agent for repository migration."""

  def __init__(self, model: Any, api_key: str | None = None):
    """Initializes the agent."""
    super().__init__(
        model=model,
        agent_domain=utils.AgentDomain.MIGRATION,
        agent_type=utils.AgentType.PRIMARY,
    )
    db_config = {
        'host': os.environ.get('ALLOYDB_HOST'),
        'database': os.environ.get('ALLOYDB_DB', 'postgres'),
        'user': os.environ.get('ALLOYDB_USER', 'postgres'),
        'password': os.environ.get('ALLOYDB_PASS'),
        'port': int(os.environ.get('ALLOYDB_PORT', '5432'))
    }
    
    self._rag_agent = retrieval_service.RetrievalService(
        model=model,
        db_config=db_config,
        embedding_model_name="text-embedding-005",
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

  async def run(self, repo_path: str, context: str | None = None) -> dict[str, str]:
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
    # Initialize the RAG pool if it hasn't been done yet
    await self._rag_agent.init_pool()
    
    if os.path.isfile(repo_path):
      with open(repo_path, "r", encoding="utf-8", errors="replace") as f:
        pytorch_code = f.read()

      if context is None:
        logging.info("[RAG] Context is None. Triggering RAG retrieval...")
        rag_context_list = await self._rag_agent.search_and_retrieve(
            pytorch_code, top_k=10
        )
        logging.info(f"[RAG] Retrieved {len(rag_context_list)} snippets from database.")
        for idx, c in enumerate(rag_context_list):
            logging.info(f"[RAG] Snippet {idx+1} from {c['file']}:\n{c['text'][:200]}...")
        rag_context = "\n\n".join([
            f"File: {c['file']}\n```python\n{c['text']}\n```"
            for c in rag_context_list
        ])
      else:
        logging.info("[RAG] Using explicitly provided context. Skipping RAG retrieval.")
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
