"""Prompt for adapting optimized code."""

import string

PROMPT = string.Template("""
You are an expert JAX and Python programmer. Your task is to refactor a given Python script into a standardized format with three specific sections: # Imports, # Initialization, and # Computation. Your goal is only to reorganize the code, not to change its logic, add features, or fix bugs.

Here is the original file you need to refactor:
```python
${original_code}
```
Here is the get_inputs() function you should refer to:
```python
${get_inputs_code}
```

Generate a single Python code block containing the entire refactored script, organized into three sections with these exact headers: # Imports, # Initialization, and # Computation. Follow these strict rules:

1.  **Structure**: The output must be organized into three sections with these exact headers:
    * `# Imports`
    * `# Initialization`
    * `# Computation`

2.  **Section Details**:
    * **# Imports**: Move all existing import statements (e.g., import jax, from jax import numpy as jnp) to this section. Do NOT add any new imports that weren't present in the original code.
    * **# Initialization**:
        - Directly use the provided `get_inputs()` function with the imports removed. No other changes should be made. 
    * **# Computation**:
        - Define `computation()`. The function signature must match the arguments provided by `get_inputs()`.  It should take the delisted (unpacked) `dynamic_args` and `static_args` as its parameters parameters (eg. if `get_inputs()` returns `dynamic_args=[A, B]` and `static_args=[N]`, then `computation()` should be defined as `def computation(A, B, N):`).
        - The `computation()` function must also be completely self-contained. Move any kernel hyperparameters, block sizes, or constants that are used by the computation (but not passed as arguments) INSIDE the `computation` function itself.
            - This function should be suitable for direct JIT compilation (e.g., via jax.jit).
        - This section can also contain any other functions that the `computation` function depends on.
        

3.  **Constraints**:
        - No Global Variables: Do not leave any global variable definitions outside of the functions. All variables must be scoped strictly within `get_inputs()` or `computation()`.
        - No Execution Calls: Remove any actual calls to `get_inputs()` or `computation()` at the end of the script. Do NOT include any code that executes the functions (e.g., `jax.block_until_ready(...)` or variable assignments from function calls). The output should ONLY contain imports and function definitions.
        - No New Logic: Do not add any new functionality, classes, or variables beyond what's necessary for the structural reorganization.
        - No Bug Fixing: Do not correct any potential bugs in the original code.
        - No Comments/Docstrings: Remove all existing comments and docstrings. Do not add any new comments or docstrings, other than the required section headers.
        - No Print Statements: Remove all print statements.
        - Preserve Behavior: The refactored code must be functionally equivalent to the original code.
        - Validity: The code must be valid Python and JAX.
""")
