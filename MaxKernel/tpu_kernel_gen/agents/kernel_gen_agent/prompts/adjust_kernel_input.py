PROMPT = """
You are an expert in JAX and Pallas. Your task is to analyze two separate scripts, a base script and an optimized script, which are structured into `imports`, `initialization`, and `computation` sections. You will update the initialization section of the optimized script to match the tensor shapes used in the base script.

Example:
Base script:
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

Optimized script:
```python
# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
N = 1024
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

Updated optimized script:
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

### TAsk
Base script:
{jax_base_code}

Optimized script:
{base_kernel_code}

Updated optimized script:
<CODE TO GENERATE>

Your final output should only include the combined code, with no additional comments or explanations.
"""
