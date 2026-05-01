"""Evaluation tool for ADK."""

import json
import logging
import os
import pathlib
import subprocess
import sys

import models
from agents.evaluation import config_agent
from agents.evaluation import test_generation_agent
from evaluation import make_data
from google.adk.tools.function_tool import FunctionTool
from google.api_core import exceptions


def generate_model_configs(
    input_dir: str,
    output_config_path: str,
    api_key: str,
    model_name: str | None = None,
) -> str:
  """Generates model configurations by analyzing PyTorch source files.

  Args:
    input_dir: The directory containing PyTorch source files.
    output_config_path: The path to save the generated JSON configuration file.
    api_key: The Google AI API key to use for configuration generation.
    model_name: The Gemini model to use for configuration generation.

  Returns:
    A string indicating the result of the configuration generation.
  """
  try:
    if pathlib.Path(output_config_path).parent:
      pathlib.Path(output_config_path).parent.mkdir(parents=True, exist_ok=True)
    model_kwargs = {"api_key": api_key}
    if model_name:
      model_kwargs["model_name"] = model_name
    model = models.GeminiTool(**model_kwargs)
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
    model_name: str | None = None,
) -> str:
  """Generates an equivalence test script for JAX and PyTorch models.

  Args:
    jax_file_path: Path to the JAX model source file.
    pytorch_file_path: Path to the PyTorch model source file.
    output_test_path: Path to save the generated test script.
    api_key: The Google AI API key to use for test generation.
    model_name: The Gemini model to use for test generation.

  Returns:
    A string indicating the result of the test generation.
  """
  try:
    with open(jax_file_path, "r") as f:
      jax_code = f.read()
    with open(pytorch_file_path, "r") as f:
      pytorch_code = f.read()

    model_kwargs = {"api_key": api_key}
    if model_name:
      model_kwargs["model_name"] = model_name
    model = models.GeminiTool(**model_kwargs)
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


def generate_oracle_data(
    input_dir: str, output_dir: str, config_path: str
) -> str:
  """Generates oracle data (.pkl files) for PyTorch models.

  Args:
    input_dir: The directory containing PyTorch source files.
    output_dir: The directory to save the generated .pkl files.
    config_path: Path to the model configurations JSON file.

  Returns:
    A string indicating the result of the data generation.
  """
  path_inserted = False
  try:
    logging.info(
        "Generating oracle data from %s using config %s.",
        input_dir,
        config_path,
    )
    sys.path.insert(0, input_dir)
    path_inserted = True
    output_dir_path = pathlib.Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    make_data.generate_data(input_dir, output_dir, config_path)
    return f"Successfully generated oracle data in {output_dir}"
  except (OSError, ValueError, RuntimeError) as e:
    logging.exception("Error during oracle data generation: %s", e)
    return f"An unexpected error occurred in generate_oracle_data: {e}"
  finally:
    if path_inserted:
      sys.path.pop(0)


generate_oracle_data_tool = FunctionTool(generate_oracle_data)


def run_equivalence_tests(
    mapping_path: str,
    data_dir: str,
    tests_dir: str,
    api_key: str,
    model_name: str | None = None,
) -> str:
  """Generates and runs equivalence tests for migrated models.

  Args:
    mapping_path: Path to mapping.json file.
    data_dir: Directory containing oracle data (.pkl files).
    tests_dir: Directory to save generated test files.
    api_key: The Google AI API key to use for test generation.
    model_name: The Gemini model to use for test generation.

  Returns:
    A string summarizing the test results.
  """
  try:
    with open(mapping_path, "r", encoding="utf-8") as f:
      mapping_log = json.load(f)
  except (OSError, json.JSONDecodeError) as e:
    return f"Error reading mapping file {mapping_path}: {e}"

  logging.info("Generating and running equivalence tests...")
  tests_dir_path = pathlib.Path(tests_dir)
  tests_dir_path.mkdir(parents=True, exist_ok=True)
  test_run_results = {}

  for item in mapping_log:
    if item["status"] == "success":
      jax_file = item["generated_file"]
      torch_file = item["source_file"]
      module_stem = pathlib.Path(jax_file).stem
      test_file_name = f"test_{module_stem}.py"
      output_test_path = tests_dir_path / test_file_name

      try:
        test_gen_result = generate_equivalence_tests(
            jax_file,
            torch_file,
            str(output_test_path),
            api_key,
            model_name=model_name,
        )
        logging.info("%s", test_gen_result)

        if not test_gen_result.startswith("Successfully generated"):
          test_run_results[test_file_name] = "ERROR (test generation failed)"
          continue

        # Run equivalence test
        pickle_file = pathlib.Path(data_dir) / f"{module_stem}.pkl"
        if output_test_path.exists() and pickle_file.exists():
          # Add migration root dir to PYTHONPATH for imports
          root_dir = str(tests_dir_path.parent.parent)
          env = os.environ.copy()
          env["PYTHONPATH"] = (
              f"{root_dir}{os.pathsep}{env.get('PYTHONPATH', '')}"
          )
          logging.info("Running equivalence test: %s", output_test_path)
          python_executable = sys.executable or "python"
          cmd = [
              python_executable,
              str(output_test_path),
              f"--pickle_path={pickle_file}",
          ]
          result = subprocess.run(
              cmd,
              capture_output=True,
              text=True,
              check=False,
              timeout=60,
              env=env,
          )
          if result.returncode == 0:
            test_run_results[test_file_name] = "PASSED"
            logging.info("Test %s PASSED", test_file_name)
          else:
            failure_detail = "FAILED"
            if result.stderr or result.stdout:
              failure_detail += ":"
            if result.stderr:
              failure_detail += f"\n--- STDERR ---\n{result.stderr.strip()}"
            if result.stdout:
              failure_detail += f"\n--- STDOUT ---\n{result.stdout.strip()}"
            test_run_results[test_file_name] = failure_detail
            logging.info(
                "Test %s FAILED\nSTDOUT:%s\nSTDERR:%s",
                test_file_name,
                result.stdout,
                result.stderr,
            )
        else:
          test_run_results[test_file_name] = (
              "SKIPPED (missing test or data file)"
          )
          logging.info(
              "Skipping test run for %s, test or data file not found.",
              test_file_name,
          )
      except (subprocess.TimeoutExpired, OSError) as e:
        test_run_results[test_file_name] = f"ERROR ({type(e).__name__}: {e})"
        logging.exception(
            "Error during test generation or execution for %s: %s",
            test_file_name,
            e,
        )

  results_summary = "Equivalence test results:\n" + "\n".join(
      f"- {k}: {v}" for k, v in test_run_results.items()
  )
  return results_summary


run_equivalence_tests_tool = FunctionTool(run_equivalence_tests)
