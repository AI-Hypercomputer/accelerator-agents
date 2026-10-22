"""Prompt for adapting reference code."""

import string

PROMT = string.Template("""
You are an expert JAX and Python programmer. Your task is to refactor a given Python script into a standardized format with three specific sections: # Imports, # Initialization, and # Computation. Your goal is only to reorganize the code, not to change its logic, add features, or fix bugs.

Here is the original file you need to refactor:
```python
${original_code}
```

Generate a single Python code block containing the entire refactored script, organized into three sections with these exact headers: # Imports, # Initialization, and # Computation. Follow these strict rules:

1.  **Structure**: The output must be organized into three sections with these exact headers:
    * `# Imports`
    * `# Initialization`
    * `# Computation`

2.  **Section Details**:
    * **# Imports**: Move all existing import statements (e.g., import jax, from jax import numpy as jnp) to this section. Do NOT add any new imports that weren't present in the original code.
    * **# Initialization**:
        - Ensure there is a function named `get_inputs()`. If the original code has a function for generating inputs, rename and adapt it. If not, create it to prepare the inputs for the computation function. 
        - The `get_inputs()` function must be completely self-contained. Move any existing global variable definitions related to data generation (e.g., dimensions like N, M, K, batch sizes, constants) INSIDE this `get_inputs()` function. Do NOT define them as global variables.
        - `get_inputs()` must return two lists: `dynamic_args` and `static_args`.
            - `dynamic_args`: A list of JAX arrays (tensors) that are inputs to the `computation` function.
            - `static_args`: A list of static values (ints, bools, shapes, etc.) that are inputs to the `computation` function.
            - The arguments returned by `get_inputs` must exactly match the signature of the computation function defined in the `# Computation` section.
            - If random numbers are generated for inputs, create the `jax.random.PRNGKey` inside the `get_inputs` function. Do not define it as a global variable.
    * **# Computation**:
        - Define `computation()`. The function signature must match the arguments provided by `get_inputs()`. It should take the delisted `dynamic_args` and `static_args` as its parameters (eg. if `get_inputs()` returns `dynamic_args=[A, B]` and `static_args=[N]`, then `computation()` should be defined as `def computation(A, B, N):`).
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
