import logging
import time

from google import genai

from evaluation.code_adapter.prompts import (
  adapt_optimized_prompt,
  adapt_reference_prompt,
)
from evaluation.custom_types.kernel_task import KernelTask

GEMINI_MODEL = "gemini-2.5-flash"

logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s [%(levelname)s] - %(message)s",
)


class CodeAdapter:
  """
  Uses an LLM to refactor raw Python/JAX code into a structured format
  with # Imports, # Initialization, and # Computation sections.
  """

  def __init__(self, client: genai.Client, max_retries: int = 3):
    self.client = client
    self.max_retries = max_retries

  def adapt(
    self,
    original_code: str,
    adapt_optimized: bool = False,
    get_inputs_code: str = None,
  ) -> str:
    """
    Takes raw Python code and uses an LLM to refactor it into a
    structured format with Imports, Initialization, and Computation sections.
    Returns a new, structured code string.

    Args:
        original_code: The original Python code to refactor.
        adapt_optimized: Whether to adapt the optimized code or the reference code.
            If True, get_inputs_code must be provided.
        get_inputs_code: Optional code for the get_inputs function, used for
            optimized code adaptation.
    """
    logging.info("Refactoring code ...")
    if not adapt_optimized:
      prompt = self._get_adapt_reference_prompt(original_code)
    else:
      if not get_inputs_code:
        raise ValueError(
          "get_inputs_code must be provided when adapt_optimized is True."
        )
      prompt = self._get_adapt_optimized_prompt(original_code, get_inputs_code)

    config = genai.types.GenerateContentConfig(temperature=0.1)
    attempt = 0
    while attempt < self.max_retries:
      try:
        response = self.client.models.generate_content(
          model=GEMINI_MODEL, contents=prompt, config=config
        )
        code = response.text.strip()
        if code.startswith("```python"):
          code = code[len("```python") :].strip()
        elif code.startswith("```"):
          code = code[len("```") :].strip()
        if code.endswith("```"):
          code = code[: -len("```")].strip()

        # Basic validation to ensure the structure is present
        if (
          "# Imports" not in code
          or "# Initialization" not in code
          or "# Computation" not in code
        ):
          raise ValueError("LLM output did not contain the required sections.")

        return code
      except Exception as e:
        attempt += 1
        wait_time = 2**attempt
        logging.warning(
          f"LLM refactoring call failed. "
          f"Retrying in {wait_time}s... \n"
          f"(Attempt {attempt}/{self.max_retries}) - Error: {e}"
        )
        time.sleep(wait_time)
    raise RuntimeError(
      f"Failed to refactor code after {self.max_retries} retries."
    )

  def generate_kernel_task(
    self, task_id: str, description: str, jax_code: str
  ) -> KernelTask:
    """
    Generates a KernelTask YAML by extracting the relevant sections
    from the refactored JAX code.
    """
    # Extract the input_gen_code from the jax_code
    input_gen_code = self._extract_input_gen_code(jax_code)

    kernel_task = KernelTask(
      task_id=task_id,
      description=description,
      input_gen_code=input_gen_code,
    )

    return kernel_task

  def _get_adapt_reference_prompt(
    self,
    original_code: str,
  ) -> str:
    """
    Creates a prompt for refactoring an entire script into the three-section format.
    """

    return adapt_reference_prompt.PROMT.substitute(original_code=original_code)

  def _get_adapt_optimized_prompt(
    self,
    original_code: str,
    get_inputs_code: str,
  ) -> str:
    """
    Creates a prompt for refactoring an optimized kernel script into the three-section format.
    """
    return adapt_optimized_prompt.PROMPT.substitute(
      original_code=original_code, get_inputs_code=get_inputs_code
    )

  def _extract_input_gen_code(self, jax_base_code: str) -> str:
    """
    Extracts code for the `input_gen_code` field.

    This function specifically:
    1. Extracts global variables from the `# Initialization` section.
    2. Extracts the `get_inputs` function.
    3. Extracts import statements from the `# Imports` section.
    4. Injects the import statements into the body of the `get_inputs` function.
    5. Combines the global variables and the modified `get_inputs` function.
    """
    lines = jax_base_code.splitlines()
    import_lines = []
    global_var_lines = []
    get_inputs_func_lines = []

    # 1. Extract import lines from the # Imports section
    in_imports_section = False
    for line in lines:
      if line.strip() == "# Imports":
        in_imports_section = True
        continue
      if in_imports_section and line.strip().startswith("#"):
        break
      if in_imports_section and line.strip():
        import_lines.append(line.strip())

    # 2. Extract globals and the get_inputs function from # Initialization
    in_initialization_section = False
    in_get_inputs_func = False
    for line in lines:
      if line.strip() == "# Initialization":
        in_initialization_section = True
        continue
      # Stop parsing when the next major section is found, not on any comment.
      if in_initialization_section and line.strip() == "# Computation":
        break
      if not in_initialization_section:
        continue

      if line.strip().startswith("def get_inputs"):
        in_get_inputs_func = True

      if in_get_inputs_func:
        get_inputs_func_lines.append(line.rstrip())
      elif line.strip():  # Only add non-empty global lines
        global_var_lines.append(line.rstrip())

    if not get_inputs_func_lines:
      return ""

    # 3. Determine indentation and inject imports into the function
    def_line = get_inputs_func_lines[0]
    body_lines = get_inputs_func_lines[1:]
    indentation = "    "  # Default
    for line in body_lines:
      if line.strip():
        indentation = " " * (len(line) - len(line.lstrip()))
        break
    injected_imports = [f"{indentation}{imp}" for imp in import_lines]
    if injected_imports and any(body_lines):
      injected_imports.append("")

    injected_globals = [f"{indentation}{g}" for g in global_var_lines]
    if injected_globals and (injected_imports or body_lines):
      injected_globals.append("")

    # 4. Reconstruct the code snippet to be a single, self-contained function
    final_lines = [def_line] + injected_imports + injected_globals + body_lines

    result = "\n".join(final_lines).strip()
    return result if result else ""
