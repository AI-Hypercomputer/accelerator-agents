"""Generates Ground Truth Oracle data from PyTorch models.

The data generation is based on a JSON config.
"""

import argparse
import importlib.util
import inspect
import json
import logging
import os
import pickle
import sys
from typing import Any

import torch
from torch import nn


def _get_forward_hook(hook_name: str, dest_dict: dict[str, Any]):
  """Returns a forward hook to capture intermediate activations."""

  def hook(unused_module, unused_input, output):
    dest_dict[hook_name] = output

  return hook


def generate_data(
    input_dir: str, output_dir: str, config_path: str, overwrite: bool = False
):
  """Generates oracle data from PyTorch models based on a config file.

  Args:
    input_dir: Directory containing PyTorch source files.
    output_dir: Directory where .pkl files will be saved.
    config_path: Path to JSON config with model instantiation arguments and
      input shapes.
    overwrite: Whether to overwrite existing pickle files.
  """
  # Safety: Create output_dir if it does not exist
  os.makedirs(output_dir, exist_ok=True)

  # Load the LLM-generated configurations
  with open(config_path, "r") as f:
    model_configs = json.load(f)

  # Ensure dynamic imports work
  sys.path.insert(0, os.path.abspath(input_dir))

  # Dynamic Discovery
  for filename in os.listdir(input_dir):
    if not filename.endswith(".py") or filename == "__init__.py":
      continue

    module_name = filename[:-3]
    output_path = os.path.join(output_dir, f"{module_name}.pkl")

    # Resume Logic (The "Check" Step)
    if os.path.exists(output_path) and not overwrite:
      logging.info("[RESUME] Skipping %s (already exists)...", module_name)
      continue

    # Load module dynamically
    filepath = os.path.join(input_dir, filename)
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    if spec is None or spec.loader is None:
      logging.error("Error: could not load module from %s", filepath)
      continue
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Iterate through classes in the module
    for name, obj in inspect.getmembers(module, inspect.isclass):
      if issubclass(obj, nn.Module) and obj is not nn.Module:
        if name not in model_configs:
          logging.warning(
              "Warning: No config found for %s in %s. Skipping.", name, filename
          )
          continue

        config = model_configs[name]
        logging.info("Processing %s from %s...", name, filename)

        try:
          # Instantiate the model
          init_kwargs = config.get("init_kwargs", {})
          model = obj(**init_kwargs)
          model.eval()

          # Input Analysis & Generation
          torch.manual_seed(42)
          input_shape = config["input_shape"]

          # Capture intermediate activations
          intermediates = {}

          for mod_name, mod in model.named_modules():
            if mod_name:
              mod.register_forward_hook(
                  _get_forward_hook(mod_name, intermediates)
              )

          # Generate dummy input(s)
          if isinstance(input_shape[0], list):
            dummy_input = tuple(torch.randn(*shape) for shape in input_shape)
            output = model(*dummy_input)
          else:
            dummy_input = torch.randn(*input_shape)
            output = model(dummy_input)

          # Extraction & Serialization
          weights = model.state_dict()
          data = {
              "input": dummy_input,
              "output": output,
              "state_dict": weights,
              "intermediates": intermediates,
          }

          with open(output_path, "wb") as f:
            pickle.dump(data, f)

          logging.info("Successfully saved oracle data to %s", output_path)
        except (KeyError, TypeError, ValueError, RuntimeError, OSError) as e:
          logging.exception("Error processing %s: %s", name, e)


def main():
  parser = argparse.ArgumentParser(
      description="Generate Ground Truth Oracle data from PyTorch models."
  )
  parser.add_argument(
      "--input_dir",
      default="./gold_refs/torch",
      help="Directory containing PyTorch source files",
  )
  parser.add_argument(
      "--output_dir",
      default="./gold_refs/data",
      help="Directory where .pkl files will be saved",
  )
  parser.add_argument(
      "--config",
      required=True,
      help="Path to JSON config with input shapes",
  )
  parser.add_argument(
      "--overwrite", action="store_true", help="Overwrite existing pickle files"
  )
  args = parser.parse_args()

  if not os.path.exists(args.config):
    logging.error(
        "Error: Configuration file %s not found. Run the LLM analyzer first.",
        args.config,
    )
    sys.exit(1)

  generate_data(args.input_dir, args.output_dir, args.config, args.overwrite)


if __name__ == "__main__":
  main()
