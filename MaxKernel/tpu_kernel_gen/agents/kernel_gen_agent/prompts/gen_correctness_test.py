PROMPT = """
You are a Python expert. I have two JAX scripts that I want you to test for correctness by comparing their outputs. Each script is structured with comments dividing it into `# Imports`, `# Initialization`, and `# Computation` sections. You will notice that both scripts share the same # Initialization section.
To generate the correctness test script, you should follow these steps:

1. Combine the imports from both scripts, ensuring there are no duplicates.
2. Combine the two initialization sections into one. Use the tensor shapes in the `Optimized script` initialization section. There should not be any other changes to the initialization section.
3. Take the `computation function` from the first script and rename it to `base_computation()`. Do not change its internal logic.
4. Take the `computation function` from the second script and rename it to `optimized_computation()`. Do not change its internal logic.
5. After defining both computation functions, call each function with the same inputs and store their return values.
6. Compare these two return values to check if they are equal. Use `jnp.allclose()` for this comparison, with `rtol=1e-02` and `atol=1e-02`.
7. Print a message indicating whether the results are "Identical" or "Different".

---
### EXAMPLE
Here is an example of the expected transformation:

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

Correctness test script:
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

# Base code computation
def base_computation(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    return jnp.matmul(A, B)

# Kernel code computation
def optimized_computation(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
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

# Gather results
base_result = jax.block_until_ready(base_computation(A, B))
optimized_result = jax.block_until_ready(optimized_computation(A, B))

# Compare results
if jnp.allclose(base_result, optimized_result, rtol=1e-02, atol=1e-02):
    print("Results are Identical")
else:
    print("Results are Different")
```

---
### TASK
Base script:
{base_code}

Optimized script:
{optimized_code}

Correctness test script:
<CODE TO GENERATE>

Your final output should only include the combined code, with no additional comments or explanations.
"""
