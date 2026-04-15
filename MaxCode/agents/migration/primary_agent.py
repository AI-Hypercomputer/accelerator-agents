"""Primary orchestration agent for repository migration."""
import ast
import logging
import os
import re
import subprocess
import tempfile
import textwrap
from typing import Any, Tuple

import models
from agents import base
from agents import utils
from agents.migration import model_conversion_agent
from agents.migration import single_file_agent
from agents.migration import validation_agent
from agents.migration.prompts import prompts
from rag import rag_agent

MAX_DEBUG_ITERATIONS = 10
logger = logging.getLogger(__name__)


def _strip_markdown_formatting(text: str) -> str:
  """Strips markdown and returns only the first python code block."""
  code_block_match = re.search(r"```(?:python)?\n?(.*?)\n?```", text, re.DOTALL)
  if code_block_match:
    return code_block_match.group(1).strip()
  # Handle truncated responses: opening ``` present but closing ``` missing
  stripped = text.strip()
  if stripped.startswith("```"):
    first_nl = stripped.find("\n")
    if first_nl != -1:
      stripped = stripped[first_nl + 1:]
    if stripped.endswith("```"):
      stripped = stripped[:-3]
    return stripped.strip()
  # Strip triple-quote wrappers the LLM may use instead of backticks.
  if stripped.startswith('"""') and stripped.endswith('"""'):
    return stripped[3:-3].strip()
  return text


def _find_missing_components(pytorch_code: str, jax_code: str) -> list[str]:
  """Returns names of top-level classes/functions in pytorch_code missing from jax_code."""
  try:
    src_tree = ast.parse(pytorch_code)
  except SyntaxError:
    return []
  try:
    out_tree = ast.parse(jax_code)
  except SyntaxError:
    return []

  src_names = {
      node.name for node in ast.iter_child_nodes(src_tree)
      if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
  }
  out_names = {
      node.name for node in ast.iter_child_nodes(out_tree)
      if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
  }
  return sorted(src_names - out_names)


def _extract_component_source(source_code: str, component_name: str) -> str:
  """Extracts the full source text of a top-level class or function."""
  try:
    tree = ast.parse(source_code)
  except SyntaxError:
    return ""
  lines = source_code.splitlines(keepends=True)
  for node in ast.iter_child_nodes(tree):
    if (isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == component_name):
      start = node.lineno - 1  # ast is 1-indexed
      end = node.end_lineno if node.end_lineno else len(lines)
      return "".join(lines[start:end])
  return ""


def _is_stub_body(body: list[ast.stmt]) -> bool:
  """Checks if a function body is a stub (pass, return None, ..., or docstring+pass)."""
  stmts = body
  # Strip leading docstring
  if stmts and isinstance(stmts[0], ast.Expr) and isinstance(stmts[0].value, (ast.Constant, ast.Str)):
    stmts = stmts[1:]
  if not stmts:
    return True
  if len(stmts) == 1:
    s = stmts[0]
    # pass
    if isinstance(s, ast.Pass):
      return True
    # ... (Ellipsis)
    if isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and s.value.value is ...:
      return True
    # return None
    if isinstance(s, ast.Return) and (s.value is None or (isinstance(s.value, ast.Constant) and s.value.value is None)):
      return True
    # raise NotImplementedError(...)
    if isinstance(s, ast.Raise) and isinstance(s.exc, ast.Call):
      func = s.exc.func
      if isinstance(func, ast.Name) and func.id == "NotImplementedError":
        return True
  return False


def _find_stub_implementations(code: str) -> list[dict]:
  """Walks AST and returns stub functions/methods.

  Returns:
    List of dicts with keys: name, kind ('function' or 'method'), parent_class (or None).
  """
  try:
    tree = ast.parse(code)
  except SyntaxError:
    return []
  stubs = []
  for node in ast.iter_child_nodes(tree):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
      if _is_stub_body(node.body):
        stubs.append({"name": node.name, "kind": "function", "parent_class": None})
    elif isinstance(node, ast.ClassDef):
      for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
          if _is_stub_body(child.body):
            stubs.append({"name": child.name, "kind": "method", "parent_class": node.name})
  return stubs


def _find_missing_methods(pytorch_code: str, jax_code: str) -> list[dict]:
  """Compares methods within matching classes and returns missing ones.

  Returns:
    List of dicts with keys: class_name, method_name.
  """
  try:
    src_tree = ast.parse(pytorch_code)
    out_tree = ast.parse(jax_code)
  except SyntaxError:
    return []

  def _class_methods(tree: ast.Module) -> dict[str, set[str]]:
    result = {}
    for node in ast.iter_child_nodes(tree):
      if isinstance(node, ast.ClassDef):
        methods = set()
        for child in ast.iter_child_nodes(node):
          if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            methods.add(child.name)
        result[node.name] = methods
    return result

  src_classes = _class_methods(src_tree)
  out_classes = _class_methods(out_tree)

  missing = []
  for cls_name, src_methods in src_classes.items():
    if cls_name in out_classes:
      for method in sorted(src_methods - out_classes[cls_name]):
        # Skip dunder methods other than __init__ and __call__
        if method.startswith("__") and method.endswith("__") and method not in ("__init__", "__call__"):
          continue
        missing.append({"class_name": cls_name, "method_name": method})
  return missing


def _extract_method_source(code: str, class_name: str, method_name: str) -> str:
  """Extracts a method's source from within a class."""
  try:
    tree = ast.parse(code)
  except SyntaxError:
    return ""
  lines = code.splitlines(keepends=True)
  for node in ast.iter_child_nodes(tree):
    if isinstance(node, ast.ClassDef) and node.name == class_name:
      for child in ast.iter_child_nodes(node):
        if (isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            and child.name == method_name):
          start = child.lineno - 1
          end = child.end_lineno if child.end_lineno else len(lines)
          return "".join(lines[start:end])
  return ""


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
    self._merge_result = None  # Set when running on a directory
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

  _FILL_PROMPT = textwrap.dedent("""\
      Convert the following PyTorch classes/functions to JAX/Flax.
      Return ONLY valid Python code. No markdown, no explanation.

      {rag_section}
      ## PyTorch components to convert:
      ```python
      {components_source}
      ```
  """)

  _FILL_STUBS_PROMPT = textwrap.dedent("""\
      The following JAX/Flax code contains stub implementations (functions or
      methods with placeholder bodies like `pass`, `return None`, `...`, or
      `raise NotImplementedError`). Replace every stub with a complete, correct
      implementation based on the original PyTorch source provided below.

      Return the COMPLETE JAX file with all stubs filled in. Do not remove any
      existing non-stub code. Return ONLY valid Python code. No markdown, no
      explanation.

      ## Original PyTorch source for reference:
      ```python
      {pytorch_source}
      ```

      ## Current JAX/Flax code (with stubs to fill):
      ```python
      {jax_code}
      ```
  """)

  def _fill_missing_components(self, pytorch_code: str,
                               jax_code: str) -> str:
    """Detects components missing from the JAX output and converts them.

    Also detects stub implementations and missing methods within classes,
    and makes a targeted LLM call to replace them with real implementations.
    """
    # --- Phase 1: Fill missing top-level components (existing logic) ---
    missing = _find_missing_components(pytorch_code, jax_code)
    if missing:
      logger.info("Missing components detected: %s", missing)

      sources = []
      for name in missing:
        src = _extract_component_source(pytorch_code, name)
        if src:
          sources.append(src)

      if sources:
        components_source = "\n\n".join(sources)
        rag_section = ""
        if self._rag_agent:
          query = "JAX Flax conversion " + " ".join(missing)
          try:
            docs = self._rag_agent.retrieve_context(query, top_k=10)
            if docs:
              rag_section = "\n## Reference Patterns (from RAG):\n"
              for doc in docs:
                rag_section += f"\n### {doc.get('name', 'unknown')}\n{doc.get('text', '')}\n"
          except Exception:
            pass

        prompt = self._FILL_PROMPT.format(
            components_source=components_source,
            rag_section=rag_section,
        )
        response = self.generate(prompt)
        converted = _strip_markdown_formatting(response)
        if converted and len(converted.strip()) > 20:
          jax_code = jax_code.rstrip() + "\n\n" + converted.strip() + "\n"

    # --- Phase 2: Fix stubs and missing methods ---
    stubs = _find_stub_implementations(jax_code)
    missing_methods = _find_missing_methods(pytorch_code, jax_code)

    if not stubs and not missing_methods:
      return jax_code

    # Collect PyTorch source snippets for the problematic components
    pytorch_snippets = []
    seen = set()
    for stub in stubs:
      if stub["parent_class"]:
        key = (stub["parent_class"], stub["name"])
        if key not in seen:
          seen.add(key)
          src = _extract_method_source(pytorch_code, stub["parent_class"], stub["name"])
          if src:
            pytorch_snippets.append(f"# {stub['parent_class']}.{stub['name']}\n{src}")
      else:
        key = (None, stub["name"])
        if key not in seen:
          seen.add(key)
          src = _extract_component_source(pytorch_code, stub["name"])
          if src:
            pytorch_snippets.append(f"# {stub['name']}\n{src}")

    for mm in missing_methods:
      key = (mm["class_name"], mm["method_name"])
      if key not in seen:
        seen.add(key)
        src = _extract_method_source(pytorch_code, mm["class_name"], mm["method_name"])
        if src:
          pytorch_snippets.append(f"# {mm['class_name']}.{mm['method_name']}\n{src}")

    if not pytorch_snippets:
      return jax_code

    stub_names = [
        f"{s['parent_class']}.{s['name']}" if s["parent_class"] else s["name"]
        for s in stubs
    ]
    mm_names = [f"{m['class_name']}.{m['method_name']}" for m in missing_methods]
    logger.info("Stub implementations found: %s", stub_names)
    logger.info("Missing methods found: %s", mm_names)

    pytorch_source = "\n\n".join(pytorch_snippets)
    prompt = self._FILL_STUBS_PROMPT.format(
        pytorch_source=pytorch_source,
        jax_code=jax_code,
    )
    response = self.generate(prompt)
    repaired = _strip_markdown_formatting(response)

    # Only accept if result is a reasonable-length complete file that parses
    if repaired and len(repaired.strip()) > len(jax_code) * 0.5:
      try:
        ast.parse(repaired)
        return repaired
      except SyntaxError:
        logger.warning("Stub-filled code has syntax errors, keeping original")
    return jax_code

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

  _MAX_REPAIR_ITERATIONS = 3

  def _validate_and_repair(self, pytorch_code: str, converted_code: str,
                           file_path: str) -> str:
    """Validates converted code and repairs deviations in a loop.

    Runs up to _MAX_REPAIR_ITERATIONS rounds of validate-then-repair.
    Exits early if no deviations remain or if the deviation count does
    not decrease (no progress).

    Args:
      pytorch_code: The original PyTorch source code.
      converted_code: The converted JAX/Flax code.
      file_path: The file path (used as key for storing results).

    Returns:
      The final code (repaired if deviations were found, original otherwise).
    """
    validator = validation_agent.ValidationAgent(
        self._model_ref, rag_agent_instance=self._rag_agent
    )

    current_code = converted_code
    prev_count = float("inf")
    initial_deviations = None
    initial_count = 0
    iteration_history = []
    final_deviations = []

    for iteration in range(1, self._MAX_REPAIR_ITERATIONS + 1):
      deviations = validator.validate(pytorch_code, current_code)
      count = len(deviations)
      logger.info("Validation of %s (iteration %d): found %d deviations",
                  file_path, iteration, count)

      # Capture initial state for backward compat
      if iteration == 1:
        initial_deviations = deviations
        initial_count = count

      iteration_history.append({
          "iteration": iteration,
          "deviation_count": count,
      })

      # Clean — no deviations remain
      if not deviations:
        final_deviations = []
        break

      # No progress — deviation count did not decrease
      if count >= prev_count:
        logger.info("No progress on %s at iteration %d (prev=%d, cur=%d), "
                    "stopping repair loop", file_path, iteration,
                    prev_count, count)
        final_deviations = deviations
        break

      current_code = validator.repair(
          current_code, deviations, pytorch_code=pytorch_code
      )
      prev_count = count
      final_deviations = deviations
    else:
      # Loop exhausted without break — run one final validation
      final_check = validator.validate(pytorch_code, current_code)
      final_deviations = final_check
      iteration_history.append({
          "iteration": self._MAX_REPAIR_ITERATIONS + 1,
          "deviation_count": len(final_check),
      })
      logger.info("Final validation of %s: %d remaining deviations",
                  file_path, len(final_check))

    result = {
        "deviations_found": initial_count,
        "deviations": initial_deviations or [],
        "remaining_deviations_count": len(final_deviations),
        "remaining_deviations": final_deviations,
        "iterations": len([h for h in iteration_history
                           if h["iteration"] <= self._MAX_REPAIR_ITERATIONS]),
        "iteration_history": iteration_history,
    }
    self._validation_results[file_path] = result
    return current_code

  def get_validation_results(self) -> dict[str, dict]:
    """Returns validation results for all processed files.

    Returns:
      A dictionary mapping file paths to their validation results, each
      containing deviations_found, deviations, remaining_deviations_count,
      and remaining_deviations.
    """
    return self._validation_results

  def get_merge_result(self):
    """Returns the MergeResult from the last directory run, or None."""
    return self._merge_result

  def run(self, repo_path: str) -> dict[str, str]:
    """Orchestrates the migration of a repository from PyTorch to JAX.

    Args:
      repo_path: The path to the repository file or directory.

    Returns:
      A dictionary mapping original file paths to converted JAX code.
    """
    if os.path.isfile(repo_path):
      with open(repo_path, "r", encoding="utf-8", errors="replace") as f:
        pytorch_code = f.read()
      logger.info("Converting %s ...", repo_path)
      converted_code = self._convert_file(pytorch_code, repo_path)
      converted_code = self._fill_missing_components(
          pytorch_code, converted_code
      )
      if self._validate:
        converted_code = self._validate_and_repair(
            pytorch_code, converted_code, repo_path
        )
      return {repo_path: converted_code}
    elif os.path.isdir(repo_path):
      from agents.migration.merge_agent import MergeAgent

      merger = MergeAgent()
      merge_result = merger.run(repo_path)
      self._merge_result = merge_result
      results = {}

      # Convert model code
      logger.info("Converting merged model code (%d files, %d chars)...",
                  len(merge_result.model_files), len(merge_result.model_code))
      model_jax = self._convert_file(
          merge_result.model_code, "merged_model.py"
      )
      model_jax = self._fill_missing_components(
          merge_result.model_code, model_jax
      )
      if self._validate:
        model_jax = self._validate_and_repair(
            merge_result.model_code, model_jax, "merged_model.py"
        )
      results["model"] = model_jax

      # Convert utility code (if any)
      if merge_result.utility_code:
        logger.info("Converting merged utility code (%d files, %d chars)...",
                    len(merge_result.utility_files),
                    len(merge_result.utility_code))
        utils_jax = self._single_file_agent.run(merge_result.utility_code)
        utils_jax = self._fill_missing_components(
            merge_result.utility_code, utils_jax
        )
        results["utils"] = utils_jax

      return results
    else:
      return {
          repo_path: f"# Error: path {repo_path} is not a file or directory."
      }
