PROMPT = """
You are a Python expert. I have two JAX scripts that I want you to test for performance by timing how long their computations take. Additionally, I want you to sweep through different tiling configurations to determine which is the fastest. Each script is structured with comments dividing it into `# Imports`, `# Initialization`, and `# Computation` sections. You will notice that both scripts share the same # Initialization section.

To generate the tiling tuning script, you should follow these steps:
1. Combine the imports from both scripts, ensuring there are no duplicates.
2. Include the shared initialization section once.
3. Take the `computation function` from the first script and rename it to `base_computation()`.
4. Take the `computation function` from the second script and rename it to `optimized_computation()`.
5. Add import `from functools import partial` and add @partial(jax.jit, static_argnames=()) decorator to both computation functions to enable JIT compilation. If there are any constants in the function signatures, include them in the `static_argnames` list.
6. In the `optimized_computation` function, make the blocks input arguments. Also change the `jax.jit` decorator to include these arguments as static arguments. Do not remove existing static arguments from the decorator.
7. After defining both computation functions, define function `benchmark` that iterates 6 times, running the provided function with the same inputs and timing how long each call takes using `time.time()`. Store the elapsed times for each iteration.
8. In function `benchmark` after the loop, compute the average time taken for each function (excluding the first iteration as a warm-up) and return the average time. Here is what `benchmark` should look like:
    ```python
    def benchmark(func):
        times = []
        jax.block_until_ready(func)
        for _ in range(5):
            start_time = time.time()
            jax.block_until_ready(func)
            times.append(time.time() - start_time)
        avg_time = sum(times) / len(times)
        return avg_time
    ```
9. Get the average time taken by the `base_computation` function by calling `benchmark(partial(base_computation, <all input args>))`.
10. Define reasonable block sizes to test based on the input data shapes. Ensure that the block size is not larger than the input data shape.
11. Iterates through all combinations of the defined block sizes and keep track of the best configuration and the best time. For each combination:
    - Ensure the dimension is divisible by the block size. If not, skip that combination.
    - Get the average time taken by the `optimized_computation` function by calling `benchmark(partial(optimized_computation, <all input args>, <block sizes>))`.
    - Wrap the call to `benchmark` in a try-except block to catch any exceptions that may arise from invalid configurations. If an exception occurs, print it and continue to the next combination.
12. At the end of the script, print out a summary that includes:
    - The average time taken by the base computation (float)
    - The best time taken by the optimized computation (float)
    - The best block configuration found (tuple of block sizes as integers)
    - The speedup achieved by the optimized computation compared to the base computation (float)

    This summary should be generated via the following code:
    ```python
    # --- Final Results ---
    print("="*40)
    print(f"Base JAX implementation time: {avg_base_time:.6f} seconds")
    print(f"Best Pallas kernel time:      {best_time:.6f} seconds")
    print(f"Best block configuration: ", best_config)
    speedup = avg_base_time / best_time
    print(f"Speedup vs. base JAX:       {speedup:.2f}x")
    print("="*40)
    ```

### EXAMPLE
Here is an example of the expected transformation:

Base script:
```python
# Imports
import jax
import jax.numpy as jnp
import jax.random as random

# Initialization
M = 2048
K = 2048
N = 2048
key = random.PRNGKey(0)
key_A, key_B = random.split(key)

A = random.normal(key_A, (M, K))
B = random.normal(key_B, (K, N))

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
M = 2048
K = 2048
N = 2048
key = random.PRNGKey(0)
key_A, key_B = random.split(key)

A = random.normal(key_A, (M, K))
B = random.normal(key_B, (K, N))

# Computation
def computation(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    # Kernel definition
    def kernel(x_ref, y_ref, z_ref):
        @pl.when(pl.program_id(1) == 0)
        def _():
            z_ref[...] = jnp.zeros_like(z_ref)

    z_ref[...] += x_ref[...] @ y_ref[...]

    bN = 128
    bK = 128
    bM = 128

    # Pallas kernel invocation
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), A.dtype),
        grid=(M // bM, N // bN, K // bK),
        in_specs=[
            pl.BlockSpec((bM, bK), lambda i, j, k: (i, k)),
            pl.BlockSpec((bK, bN), lambda i, j, k: (k, j)),
        ],
        out_specs=pl.BlockSpec((bM, bN), lambda i, j, k: (i, j)),
    )(A, B)

C = jax.block_until_ready(computation(A, B))
```

Tile tuning script:
```python
# Imports
import time
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl
from functools import partial

# Initialization
M = 2048
K = 2048
N = 2048
key = random.PRNGKey(0)
key_A, key_B = random.split(key)

A = random.normal(key_A, (M, K))
B = random.normal(key_B, (K, N))

# Base code computation
@partial(jax.jit, static_argnames=())
def base_computation(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    return jnp.matmul(A, B)

# Kernel code computation
@partial(jax.jit, static_argnames=("bN", "bK", "bM"))
def optimized_computation(A: jnp.ndarray, B: jnp.ndarray, bN: int, bK: int, bM: int) -> jnp.ndarray:
    # Kernel definition
    def kernel(x_ref, y_ref, z_ref):
        @pl.when(pl.program_id(1) == 0)
        def _():
            z_ref[...] = jnp.zeros_like(z_ref)

        z_ref[...] += x_ref[...] @ y_ref[...]

    # Pallas kernel invocation
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), A.dtype),
        grid=(M // bM, N // bN, K // bK),
        in_specs=[
            pl.BlockSpec((bM, bK), lambda i, j, k: (i, k)),
            pl.BlockSpec((bK, bN), lambda i, j, k: (k, j)),
        ],
        out_specs=pl.BlockSpec((bM, bN), lambda i, j, k: (i, j)),
    )(A, B)

# Benchmarking 
def benchmark(func):
    times = []
    jax.block_until_ready(func)
    for _ in range(5):
        start_time = time.time()
        jax.block_until_ready(func)
        times.append(time.time() - start_time)
    avg_time = sum(times) / len(times)
    return avg_time

## Benchmark the base JAX implementation
avg_base_time = benchmark(partial(base_computation, A, B))

## Benchmark the Pallas kernel across different block sizes
block_sizes_to_test = [128, 256, 512, 1024, 2048]
results = {}
best_time = float('inf')
best_config = (0, 0, 0)

for bN in block_sizes_to_test:
    for bK in block_sizes_to_test:
        for bM in block_sizes_to_test:
            # Ensure the dimension is divisible by the block size
            if M % bM != 0 or K % bK != 0 or N % bN != 0:
                continue

            try:
                benchmark_func = partial(optimized_computation, A, B, bN, bK, bM)
                avg_optimized_time = benchmark(benchmark_func)
                results[(bN, bK, bM)] = avg_optimized_time

                if avg_optimized_time < best_time:
                    best_time = avg_optimized_time
                    best_config = (bN, bK, bM)
            except Exception as e:
                print("Exception: ", e)

# Benchmarking Results
print("="*40)
print(f"Base JAX implementation time: {avg_base_time:.6f} seconds")
print(f"Best Pallas kernel time:      {best_time:.6f} seconds")
print(f"Best block configuration: ", best_config)
speedup = avg_base_time / best_time
print(f"Speedup vs. base JAX:       {speedup:.2f}x")
print("="*40)
```

---
### TASK
Base script:
{jax_base_code}

Optimized script:
{kernel_code}

Tile tuning script:
<CODE TO GENERATE>

Your final output should only include the combined code, with no additional comments or explanations.
"""
