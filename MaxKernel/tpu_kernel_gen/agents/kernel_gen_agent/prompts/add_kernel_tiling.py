PROMPT = """
You are an expert in JAX and Pallas. Your task is to analyze arbitrary JAX code, which is structured into `imports`, `initialization`, and `computation` sections. You will update the Pallas kernel in the `computation` section to including kernel tiling.

### Steps:
1. **Understand the Code**: Analyze the provided JAX code to understand its logic and data flow.
2. **Define block sizes**: Determine appropriate block sizes for tiling based on the input data shapes. Ensure that the block size is not the same or larger than the input data shape.
3. **Update the kernel invokation**: Add the `grid_spec` argument to the `pl.pallas_call` function in the `computation` function. Do not change or add any other arguments.
4. **Update the kernel definition**: Update the logic in the `kernel` function to include the tiling logic. Make sure to only make changes to support tiling, no other optimizations.


### Tool Usage
You have access to tool `search_api_tool` that can search for and retrieve information about JAX APIs. You can use this tool to ensure you are using JAX and Pallas APIs correctly. For example, you can use it to verify function signatures, understand the purpose of different arguments, and check for official usage notes or constraints mentioned in the documentation. 

To use the tool effectively, make sure to use tool name `search_api_tool` and provide the fully qualified name of the API as the tool's input (e.g., "jax.experimental.pallas.pallas_call" or "jax.experimental.pallas.BlockSpec").

### Full Pallas Documentation:
To provide you with the most comprehensive understanding of Pallas, here is the full documentation:
{pallas_docs}

### TPU Specs:
TPU generation to optimize for: tpu-v4
TPU high bandwidth memory (HBM) size: 32 GiB
TPU vector memory (VMEM) size: 32 MiB

### Example 1
Basic Pallas Kernel:
```python
# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
M = 1024
K = 1024
N = 1024
key = random.PRNGKey(0)
key_A, key_B = random.split(key)

A = random.normal(key_A, (M, K))
B = random.normal(key_B, (K, N))

# Computation
def computation(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:

    # Kernel definition
    def kernel(x_ref, y_ref, z_ref):
        z_ref[...] = x_ref[...] @ y_ref[...]

    # Pallas kernel invocation
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(A.shape, A.dtype),
    )(A, B)

C = jax.block_until_ready(computation(A, B))
```

Pallas Kernel with Tiling:
```
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
M = 1024
K = 1024
N = 1024
key = random.PRNGKey(0)
key_A, key_B = random.split(key)

A = random.normal(key_A, (M, K))
B = random.normal(key_B, (K, N))

# Computation
def computation(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    bM = 128
    bK = 128
    bN = 128

    # Kernel definition
    def kernel(x_ref, y_ref, z_ref):
        @pl.when(pl.program_id(2) == 0)
        def _():
            z_ref[...] = jnp.zeros_like(z_ref)

        z_ref[...] += x_ref[...] @ y_ref[...]

    # Pallas kernel invocation
    return pl.pallas_call(
        kernel,
        grid_spec=pl.GridSpec(
          grid=(M // bM, N // bN, K // bK),
          in_specs=[
            pl.BlockSpec((bM, bK), lambda i, j, k: (i, k)),
            pl.BlockSpec((bK, bN), lambda i, j, k: (k, j)),
          ],
          out_specs=pl.BlockSpec((bM, bN), lambda i, j, k: (i, j)),
        ),
        out_shape=jax.ShapeDtypeStruct((M, N), A.dtype),
    )(A, B)

C = jax.block_until_ready(computation(A, B))
```

### Example 2
Basic Pallas Kernel:
```python
# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl
from functools import partial

# Initialization
M = 1024
N = 1024
multiplier = 2.0
key = random.PRNGKey(0)

A = random.normal(key, (M, N))

# Computation
def computation(A: jnp.ndarray, multiplier: float) -> jnp.ndarray:
    # Kernel definition
    def kernel(x_ref, y_ref, multiplier: float):
        y_ref[...] = x_ref[...] * multiplier

    # Pallas kernel invocation
    return pl.pallas_call(
        partial(kernel, multiplier=multiplier),
        out_shape=jax.ShapeDtypeStruct(A.shape, A.dtype),
    )(A)

C = jax.block_until_ready(computation(A, multiplier))
```

Pallas Kernel with Tiling:
```python
# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl
from functools import partial

# Initialization
M = 1024
N = 1024
multiplier = 2.0
key = random.PRNGKey(0)

A = random.normal(key, (M, N))

# Computation
def computation(A: jnp.ndarray, multiplier: float) -> jnp.ndarray:
    bM = 128
    bN = 128

    # Kernel definition
    def kernel(x_ref, y_ref, multiplier: float):
        y_ref[...] = x_ref[...] * multiplier

    # Pallas kernel invocation
    return pl.pallas_call(
        partial(kernel, multiplier=multiplier),
        grid_spec=pl.GridSpec(
            grid=(M // bM, N // bN),
            in_specs=[
                pl.BlockSpec(block_shape=(bM, bN), index_map=lambda i, j: (i, j)),
            ],
            out_specs=pl.BlockSpec(block_shape=(bM, bN), index_map=lambda i, j: (i, j)),
        ),
        out_shape=jax.ShapeDtypeStruct(A.shape, A.dtype),
    )(A)

C = jax.block_until_ready(computation(A, multiplier))
```

### Task
Basic Pallas Kernel:
{base_kernel_code}

Pallas Kernel with Tiling:
<CODE TO GENERATE>
    
Your final output should only include the updated script.
"""
