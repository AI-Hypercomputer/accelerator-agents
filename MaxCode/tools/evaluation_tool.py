"""Evaluation tool for ADK."""

import json
import os

import models
from agents.evaluation import config_agent
from agents.evaluation import test_generation_agent
from google.adk.tools.function_tool import FunctionTool
from google.api_core import exceptions


def generate_model_configs(
    input_dir: str, output_config_path: str, api_key: str
) -> str:
  """Generates model configurations by analyzing PyTorch source files.

  Args:
    input_dir: The directory containing PyTorch source files.
    output_config_path: The path to save the generated JSON configuration file.
    api_key: The Google AI API key to use for configuration generation.

  Returns:
    A string indicating the result of the configuration generation.
  """
  try:
    model = models.GeminiTool(
        model_name=models.GeminiModel.GEMINI_2_5_PRO, api_key=api_key
    )
    agent = config_agent.ConfigGenerationAgent(model)

    master_config = {}
    for filename in os.listdir(input_dir):
      if not filename.endswith(".py") or filename == "__init__.py":
        continue

      filepath = os.path.join(input_dir, filename)
      with open(filepath, "r") as f:
        code = f.read()

      try:
        response = agent.run(code)
        # The response should be a JSON string. We need to remove potential
        # markdown formatting like ```json ... ```
        if response.startswith("```json"):
          response = response[len("```json") : -3].strip()
        config = json.loads(response)
        master_config.update(config)
      except json.JSONDecodeError as e:
        return f"Error decoding JSON from agent for file {filename}: {e}"
      except exceptions.GoogleAPIError as e:
        return f"Error during API call for file {filename}: {e}"

    with open(output_config_path, "w") as f:
      json.dump(master_config, f, indent=2)

    return (
        f"Successfully generated model configurations at {output_config_path}"
    )
  except (
      FileNotFoundError,
      PermissionError,
      OSError,
      TypeError,
      ValueError,
  ) as e:
    return f"An unexpected error occurred in generate_model_configs: {e}"


generate_model_configs_tool = FunctionTool(generate_model_configs)


def generate_equivalence_tests(
    jax_file_path: str,
    pytorch_file_path: str,
    output_test_path: str,
    api_key: str,
) -> str:
  """Generates an equivalence test script for JAX and PyTorch models.

  Args:
    jax_file_path: Path to the JAX model source file.
    pytorch_file_path: Path to the PyTorch model source file.
    output_test_path: Path to save the generated test script.
    api_key: The Google AI API key to use for test generation.

  Returns:
    A string indicating the result of the test generation.
  """
  try:
    with open(jax_file_path, "r") as f:
      jax_code = f.read()
    with open(pytorch_file_path, "r") as f:
      pytorch_code = f.read()

    model = models.GeminiTool(
        model_name=models.GeminiModel.GEMINI_2_5_PRO, api_key=api_key
    )
    agent = test_generation_agent.TestGenerationAgent(model)
    response = agent.run(jax_code=jax_code, pytorch_code=pytorch_code)

    if response.startswith("```python"):
      response = response[len("```python") : -3].strip()

    os.makedirs(os.path.dirname(output_test_path), exist_ok=True)
    with open(output_test_path, "w") as f:
      f.write(response)

    return f"Successfully generated equivalence test at {output_test_path}"
  except (
      FileNotFoundError,
      PermissionError,
      OSError,
      TypeError,
      ValueError,
      exceptions.GoogleAPIError,
  ) as e:
    return f"An unexpected error occurred in generate_equivalence_tests: {e}"


generate_equivalence_tests_tool = FunctionTool(generate_equivalence_tests)
