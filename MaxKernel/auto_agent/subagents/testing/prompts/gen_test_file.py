PROMPT = """You are tasked with generating the input generation function `get_inputs()` for testing a Pallas kernel.

The testing harness is already provided by the framework and handles testing, correctness checking, benchmarking, and layout alignment. It expects you to provide a function called `get_inputs()` that returns a list of `(dynamic_args, static_args)` tuples.

## Finding the Base Kernel
You need to read the base kernel to understand what inputs are required.

**Step 1: Check Session State**
- Base kernel: `{base_kernel_path?}`

If the path is available → proceed to read it using the `read_file` tool.

**Step 2: If Path is Missing**
**STOP immediately and ask the user. DO NOT use list_directory or search for files.**

## Your Task

1. **Read the base kernel file** using the `read_file` tool to understand:
   - The function name and signature
   - Input shapes and types (especially `jax.numpy` arrays)

2. **CRITICAL: Check for existing input generation function or test cases**
   - Examine the base kernel (`{base_kernel_path?}`) to see if it already contains an input generation function (e.g., `get_inputs()` or similar) or specific test shapes.
   - If test cases or input generation logic ALREADY EXIST in the base kernel, you MUST use those exact test cases. Adapt them into the required `get_inputs()` format (a list of `(dynamic_args, static_args)` tuples). Do NOT write completely new shapes or inputs from scratch if they are already provided.

3. **Generate a Python snippet containing ONLY `def get_inputs():` and necessary imports**
   - If no input generation function was found, write one from scratch:
     - Import necessary libraries (e.g., `import jax`, `import jax.numpy as jnp`).
     - Define `def get_inputs():`
     - Create test inputs with multiple sizes and test edge cases (e.g., zeros, ones, random inputs).
   - Separate the arguments for each case into `dynamic_args` (arrays/tensors) and `static_args` (scalars, block sizes).
   - Return a list of `(dynamic_args, static_args)` tuples to match the test harness.
   - Example:
     ```python
     import jax
     import jax.numpy as jnp

     def get_inputs():
         key = jax.random.PRNGKey(0)
         cases = []
         
         # Case 1: Standard random inputs
         x1 = jax.random.normal(key, (1024, 1024), dtype=jnp.float32)
         y1 = jax.random.normal(key, (1024, 1024), dtype=jnp.float32)
         cases.append(([x1, y1], []))
         
         # Case 2: Edge case with zeros
         x2 = jnp.zeros((256, 256), dtype=jnp.float32)
         y2 = jnp.zeros((256, 256), dtype=jnp.float32)
         cases.append(([x2, y2], []))

         # Case 3: Edge case with ones
         x3 = jnp.ones((512, 512), dtype=jnp.float32)
         y3 = jnp.ones((512, 512), dtype=jnp.float32)
         cases.append(([x3, y3], []))
         
         return cases
     ```
## Output Format
1. **Write the complete snippet** containing `get_inputs()` using the `restricted_write_file` tool. The tool will automatically wrap your snippet in the rigorous testing harness and save it.
   - Required Arguments:
     - `content` (string): The Python code snippet.
     - `kernel_name` (string): The exact function name of the base kernel entry point (e.g., `"computation"`, `"matmul"`, etc.).
     - `atol` (float, optional): Absolute tolerance for correctness checks (default 1e-2). Set appropriately based on precision (BF16 should be 1e-2 or higher).
     - `rtol` (float, optional): Relative tolerance for correctness checks (default 1e-2). Set appropriately based on precision.

Generate the `get_inputs()` Python snippet now.
"""
