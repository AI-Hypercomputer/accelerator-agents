PROMPT = """
You are an expert in JAX and Pallas. Your task is to analyze arbitrary Python code, which is structured into `imports`, `initialization`, and `computation` sections. Based on your analysis, you will rewrite function `computation` to invoke a Pallas kernel named `kernel`. This invocation will replace the existing `computation` section. You do not need to write the body of the `kernel` function itself.

Your primary goal is to construct the `pallas_call` to invoke this `kernel`, focusing on defining the correct execution grid and data chunking specifications.

### Key Principles

1.  **Grid Execution**: The kernel is executed in parallel over different parts of the input data. This is controlled by the `grid` argument in `pallas_call`. A grid of `(n, m)` is analogous to a nested for-loop structure, creating `n * m` kern
el instances.
    ```python
    # A (n, m) grid is like:
    for i in range(n):
      for j in range(m):
        kernel(...) # instance (i, j)
    ```

2.  **Data Chunking with `BlockSpec`**: Inputs and outputs for the kernel must be chunked using `pallas.BlockSpec`. This is done via the `in_specs` and `out_specs` arguments of `pallas_call`. A `BlockSpec` defines a mapping from a grid iteration index (e.g., `i, j`) to a specific slice (or "block") of the full input/output arrays.

    **`BlockSpec` Example: Matrix Multiplication**
    To make this concrete, let’s say we want to multiply two `(1024, 1024)` matrices, `x` and `y`, to produce `z`, and we want to parallelize the computation 4 ways. We can split `z` into four `(512, 512)` blocks, where each block is computed by a `(512, 1024) x (1024, 512)` matrix multiplication.
    - **Grid**: We use a `(2, 2)` grid, creating four program instances.
    - **`in_specs`**:
        - For `x`, we use `pallas.BlockSpec((512, 1024), lambda i, j: (i, 0))`. This carves `x` into horizontal blocks. The lambda `(i, j) -> (i, 0)` means the block is selected only by the first grid index, `i`.
        - For `y`, we use `pallas.BlockSpec((1024, 512), lambda i, j: (0, j))`. This carves `y` into vertical blocks, selected by the second grid index, `j`.
    - **`out_specs`**: For `z`, we use `pallas.BlockSpec((512, 512), lambda i, j: (i, j))`, which maps each grid instance `(i, j)` to a unique `(512, 512)` block in the output.

3.  **Handling Uneven Division**: If the `block_shape` in a `BlockSpec` does not divide the full array's shape evenly, Pallas handles the boundaries with padding. In the last iteration along an axis, the kernel will still receive a block of size `block_shape`. However, for elements that are out-of-bounds, the input values are unspecified (i.e., garbage), and any values written to out-of-bounds locations on output are discarded.

4.  **TPU Compatibility Constraints**: When generating a `BlockSpec`, you must adhere to the following constraints for TPU compatibility:
    - Blocks must have a rank of at least 1.
    - For blocks with rank > 1, the last two dimensions of the `block_shape` must either be equal to the corresponding dimensions of the full array or be divisible by 8 and 128, respectively.
    - For blocks with a rank of 1, the block dimension must either be equal to the array dimension or be divisible by $128 \\cdot (32 / \text{bitwidth(dtype)})$.
5. * **Kernel Signature:** The kernel function must accept reference objects (e.g., `x_ref`, `y_ref`) as inputs, which are mutable buffers in memory.

### Your Task

Given Python code with sections for `imports`, `initialization`, and `computation`, you must:

1.  **Analyze** the `computation` section to understand the operations and data access patterns.
2.  **Define** the `in_specs` and `out_specs` using `pallas.BlockSpec` that correctly chunk the input and output arrays according to the analysis and TPU constraints.
3.  **Construct** the `pallas_call` to the `kernel`, providing the appropriate `grid`, `in_specs`, and `out_specs` arguments. Do not include any other function arguments.

### Tool Usage
You have access to tool `search_api_tool` that can search for and retrieve information about JAX APIs. You can use this tool to ensure you are using JAX and Pallas APIs correctly. For example, you can use it to verify function signatures, understand the purpose of different arguments, and check for official usage notes or constraints mentioned in the documentation. 

To use the tool effectively, make sure to use tool name `search_api_tool` and provide the fully qualified name of the API as the tool's input (e.g., "jax.experimental.pallas.pallas_call" or "jax.experimental.pallas.BlockSpec").

### Full Pallas Documentation:
To provide you with the most comprehensive understanding of Pallas, here is the full documentation:
{pallas_docs}

### Example

Input Code:
```python
# Imports
import jax
import jax.numpy as jnp
import jax.random as random

# Initialization
N = 2048
key = random.PRNGKey(0)
key_A, key_B = random.split(key)

A = random.normal(key_A, (N, N))
B = random.normal(key_B, (N, N))

# Computation
def computation(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    return jnp.matmul(A, B)

C = jax.block_until_ready(computation(A, B))
```

Kernel Invocation:
```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def computation(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    bN = 128

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(A.shape, A.dtype),
        grid=(N // bN, N // bN),
        in_specs=[
            pl.BlockSpec(block_shape=(bN, N), index_map=lambda i, j: (i, 0)),
            pl.BlockSpec(block_shape=(N, bN), index_map=lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec(block_shape=(bN, bN), index_map=lambda i, j: (i, j)),
    )(A, B)

C = jax.block_until_ready(computation(A, B))
```

Now write the `pallas_call` to invoke the `kernel` function based on the provided code. Your final output must *only* include the code for the `pallas_call` invocation and all necessary Python imports.

Input Code:
{jax_base_code}

Kernel Invocation:
<CODE TO GENERATE>
"""
