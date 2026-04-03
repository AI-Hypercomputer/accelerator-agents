PROMPT = """
You are a Python expert. I have two JAX scripts that I want you to test for performance by timing how long their computations take. Each script is structured with comments dividing it into `# Imports`, `# Initialization`, and `# Computation` sections. You will notice that both scripts share the same # Initialization section.
To generate the performance test script, you should follow these steps:

1. Combine the imports from both scripts, ensuring there are no duplicates.
2. Include the shared initialization section once.
3. Take the `computation function` from the first script and rename it to `base_computation()`.
4. Take the `computation function` from the second script and rename it to `optimized_computation()`.
5. Add import `from functools import partial` and add @partial(jax.jit, static_argnames=()) decorator to both computation functions to enable JIT compilation.
6. After defining both computation functions, define a loop that iterates 6 times, calling each function with the same inputs and timing how long each call takes using `time.time()`. Store the elapsed times for each iteration.
7. After the loop, compute the average time taken for each function (excluding the first iteration as a warm-up).
8. Print the average times for both functions.

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

Performance test script:
```python
# Imports
import time
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl
from functools import partial

# Initialization
N = 2048
key = random.PRNGKey(0)
key_A, key_B = random.split(key)

A = random.normal(key_A, (N, N))
B = random.normal(key_B, (N, N))

# Base code computation
@partial(jax.jit, static_argnames=())
def base_computation(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    return jnp.matmul(A, B)

# Kernel code computation
@partial(jax.jit, static_argnames=())
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
base_times = []
optimized_times = []
for i in range(6):
    start_time = time.time()
    base_result = jax.block_until_ready(base_computation(A, B))
    base_elapsed = time.time() - start_time

    start_time = time.time()
    optimized_result = jax.block_until_ready(optimized_computation(A, B))
    optimized_elapsed = time.time() - start_time

    base_times.append(base_elapsed)
    optimized_times.append(optimized_elapsed)

avg_base_time = sum(base_times[1:]) / (len(base_times) - 1)
avg_optimized_time = sum(optimized_times[1:]) / (len(optimized_times) - 1)

print(f"Average time for base computation: {avg_base_time:.6f} seconds")
print(f"Average time for optimized computation: {avg_optimized_time:.6f} seconds")
```

---
### TASK
Base script:
{jax_base_code}

Optimized script:
{kernel_code}

Performance test script:
<CODE TO GENERATE>

Your final output should only include the combined code, with no additional comments or explanations.
"""
