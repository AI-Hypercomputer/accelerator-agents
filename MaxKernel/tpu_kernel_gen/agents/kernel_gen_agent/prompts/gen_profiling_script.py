PROMPT = """You are a Python expert. I have a JAX script that uses a Pallas kernel, and I want you to generate a script that uses XProf to profile the execution of the Pallas kernel.

To generate the profiling script, you should follow these steps:
1. Start with the original JAX script as input.
2. Add import `from functools import partial` and add @partial(jax.jit, static_argnames=()) decorator to both computation functions to enable JIT compilation. If there are any constants in the function signatures, include them in the `static_argnames` list.
3. Change the block size from the original JAX script to use the best block sizes from the performance study.
4. Define profiling options using `jax.profiler.ProfileOptions()`. Set `python_tracer_level` to 0, `host_tracer_level` to 2, and `advanced_configuration` to `{"tpu_trace_mode": "TRACE_COMPUTE_AND_SYNC"}`.
5. Start the profiler trace using `jax.profiler.start_trace('jax_trace_dir', profiler_options=options)`.
6. Execute the computation 3 times inside a loop, ensuring that the computation is JAX-blocked until ready each time.
7. Stop the profiler trace using `jax.profiler.stop_trace()`.

# Example
Here is an example of how to add profiling to the existing JAX script:

Jax script:
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
@partial(jax.jit, static_argnames=())
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

Profile results:
TODO

Jax Script with Profiling:
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
@jax.jit
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

# Profile options
options = jax.profiler.ProfileOptions()
options.python_tracer_level = 0
options.host_tracer_level = 2
options.advanced_configuration = {"tpu_trace_mode": "TRACE_COMPUTE_AND_SYNC"}

# Profile execution
jax.profiler.start_trace('jax_trace', profiler_options=options)
for i in range(3):
    C = jax.block_until_ready(computation(A, B))
jax.profiler.stop_trace()
```

Follow the format of the example above to add profiling to the following JAX script.

Jax script:
{kernel_code}

Jax Script with Profiling:
<CODE TO GENERATE>

Your final output should only include the code with no additional comments or explanations.
"""
