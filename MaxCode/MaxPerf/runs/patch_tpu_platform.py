#!/usr/bin/env python3
# pylint: skip-file
import os

PATH = "/home/gvanica_google_com/tpu-inference/tpu_inference/platforms/tpu_platform.py"

if not os.path.exists(PATH):
  print(f"Error: File {PATH} does not exist!")
  exit(1)

with open(PATH, "r") as f:
  content = f.read()

print("=== Patch 1: Fixing Circular Import (SamplingParams) ===")

# Standardize line endings to \n to ensure match compatibility
content_std = content.replace("\r\n", "\n")

# 1. Remove module-level import
old_import = "from vllm.sampling_params import SamplingParams, SamplingType"
if old_import in content_std:
  content_std = content_std.replace(old_import, "")
  print("  [x] Removed module-level import successfully.")
else:
  print("  [ ] Module-level import already removed or not found.")

# 2. Replace method signature and inject local import (matching exact whitespace)
old_block = """    @classmethod
    def validate_request(
        cls,
        prompt: PromptType,
        params: Union[SamplingParams, PoolingParams],
        processed_inputs: ProcessorInputs,
    ) -> None:"""

new_block = """    @classmethod
    def validate_request(
        cls,
        prompt: PromptType,
        params: Union["SamplingParams", "PoolingParams"],
        processed_inputs: ProcessorInputs,
    ) -> None:
        from vllm.sampling_params import SamplingParams, SamplingType"""

# Standardize block line endings
old_block_std = old_block.replace("\r\n", "\n")
new_block_std = new_block.replace("\r\n", "\n")

if old_block_std in content_std:
  content_std = content_std.replace(old_block_std, new_block_std)
  print(
      "  [x] Injected local import and updated validate_request annotations"
      " successfully."
  )
else:
  print(
      "  [ ] Strict block match failed. Running bulletproof local import"
      " insertion..."
  )
  target_line = "        if isinstance(params, SamplingParams):"
  replacement_line = (
      "        from vllm.sampling_params import SamplingParams, SamplingType\n "
      "       if isinstance(params, SamplingParams):"
  )
  if (
      target_line in content_std
      and "from vllm.sampling_params import SamplingParams" not in content_std
  ):
    content_std = content_std.replace(target_line, replacement_line)
    print("  [x] Successfully injected local import using bulletproof target.")
  else:
    print("  [ ] Local import already injected or target not found.")

print("\n=== Patch 2: Fixing PallasAttentionBackend Import Paths ===")
# 1. Replace standard vllm import path with tpu_inference path in check_and_update_config
old_pallas_import = (
    "from vllm.v1.attention.backends.pallas import PallasAttentionBackend"
)
new_pallas_import = (
    "from tpu_inference.layers.vllm.attention import PallasAttentionBackend"
)
if old_pallas_import in content_std:
  content_std = content_std.replace(old_pallas_import, new_pallas_import)
  print(
      "  [x] Replaced Pallas import from vllm with local tpu_inference package."
  )
else:
  print("  [ ] Pallas import already replaced or not found.")

# 2. Replace string representation in get_attention_backend_cls
old_backend_str = "vllm.attention.backends.pallas.PallasAttentionBackend"
new_backend_str = "tpu_inference.layers.vllm.attention.PallasAttentionBackend"
if old_backend_str in content_std:
  content_std = content_std.replace(old_backend_str, new_backend_str)
  print("  [x] Replaced class path string in get_attention_backend_cls.")
else:
  print("  [ ] Class path string already replaced or not found.")

with open(PATH, "w") as f:
  f.write(content_std)

print("\nAll tpu_platform.py patches completed successfully!")
