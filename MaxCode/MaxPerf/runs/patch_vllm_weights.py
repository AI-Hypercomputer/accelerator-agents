#!/usr/bin/env python3
# pylint: skip-file
import os

PATH = "/home/gvanica_google_com/vllm/vllm/model_executor/model_loader/weight_utils.py"

if not os.path.exists(PATH):
  print(f"Error: File {PATH} does not exist!")
  exit(1)

with open(PATH, "r") as f:
  content = f.read()

print("=== Patching vLLM weight_utils.py for FP8 CPU support ===")

# Standardize line endings to \n to ensure match compatibility
content_std = content.replace("\r\n", "\n")

old_block = """                param.copy_(
                    (high - low)
                    * torch.rand(
                        param.shape,
                        generator=generator,
                        dtype=param.dtype,
                        layout=param.layout,
                        requires_grad=param.requires_grad,
                        device="cpu",
                    )
                    + low
                )"""

new_block = """                # Bypassing FP8 NotImplementedError and math limitations on PyTorch CPU
                is_fp8 = param.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) if hasattr(torch, "float8_e4m3fn") else False
                rand_dtype = torch.bfloat16 if is_fp8 else param.dtype

                rand_tensor = torch.rand(
                    param.shape,
                    generator=generator,
                    dtype=rand_dtype,
                    layout=param.layout,
                    requires_grad=param.requires_grad,
                    device="cpu",
                )

                # Perform scaling on bfloat16/float32 first to bypass CPU math limitations
                scaled_tensor = (high - low) * rand_tensor + low

                # Cast to target FP8 precision at the very end
                if is_fp8:
                    scaled_tensor = scaled_tensor.to(param.dtype)

                param.copy_(scaled_tensor)"""

old_block_std = old_block.replace("\r\n", "\n")
new_block_std = new_block.replace("\r\n", "\n")

# Also search for the intermediate version if it was already patched in some form
intermediate_block = """                # Bypassing FP8 NotImplementedError on PyTorch CPU
                is_fp8 = param.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) if hasattr(torch, "float8_e4m3fn") else False
                rand_dtype = torch.bfloat16 if is_fp8 else param.dtype

                rand_tensor = torch.rand(
                    param.shape,
                    generator=generator,
                    dtype=rand_dtype,
                    layout=param.layout,
                    requires_grad=param.requires_grad,
                    device="cpu",
                )
                if is_fp8:
                    rand_tensor = rand_tensor.to(param.dtype)

                param.copy_(
                    (high - low) * rand_tensor + low
                )"""

intermediate_block_std = intermediate_block.replace("\r\n", "\n")

if old_block_std in content_std:
  content_std = content_std.replace(old_block_std, new_block_std)
  print(
      "  [x] Patched TPU dummy weights FP8 random uniform initialization"
      " successfully."
  )
elif intermediate_block_std in content_std:
  content_std = content_std.replace(intermediate_block_std, new_block_std)
  print(
      "  [x] Updated intermediate TPU dummy weights FP8 patch with CPU scaling"
      " fix successfully."
  )
else:
  print("  [ ] Target block not found or already patched.")

with open(PATH, "w") as f:
  f.write(content_std)

print("\nweight_utils.py patch completed!")
