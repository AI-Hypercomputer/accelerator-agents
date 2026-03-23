PROMPT = """
Your task is to combine the following code sections into a single, executable script. Organize the code under the following three sections comments while maintaining all necessary imports and logic.

1) # Imports
2) # Initialization
3) # Computation

---
### EXAMPLE
Here is an example of the expected transformation:

Base code:
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

C = computation(A, B)
```

Kernel code:
```python
def kernel(x_ref, y_ref, z_ref):
    @pl.when(pl.program_id(1) == 0)
    def _():
        z_ref[...] = jnp.zeros_like(z_ref)

    z_ref[...] += x_ref[...] @ y_ref[...]
```

Code to call kernel:
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

C = computation(A, B)
```

Combined code:
```python
# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
N = 2048
key = random.PRNGKey(0)
key_A, key_B = random.split(key)

A = random.normal(key_A, (N, N))
B = random.normal(key_B, (N, N))

# Computation
def computation(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    # Kernel definition
    def kernel(x_ref, y_ref, z_ref):
        @pl.when(pl.program_id(1) == 0)
        def _():
            z_ref[...] = jnp.zeros_like(z_ref)

    z_ref[...] += x_ref[...] @ y_ref[...]

    bN = 128

    # Pallas kernel invocation
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

Note: We want to wrap the final result in `jax.block_until_ready` to ensure that the computation is completed before proceeding.

---
### TASK
Now, combine the following code sections into a single script.
Base code:
{jax_base_code}

Kernel code:
{kernel_code}

Code to call kernel:
{kernel_invoking_code}

Combined code:
<CODE TO GENERATE>

Your final output should only include the combined code, with no additional comments or explanations.
"""
