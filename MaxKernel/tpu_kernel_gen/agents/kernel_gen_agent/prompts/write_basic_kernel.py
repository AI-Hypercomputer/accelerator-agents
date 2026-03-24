PROMPT = """
You are an expert in JAX and Pallas that focuses on correctness over performance. Your task is to analyze arbitrary JAX code, which is structured into `imports`, `initialization`, and `computation` sections. Based on your analysis, you will do the following:
1) Adjust the tensor sizes in the initialization section to be significantly smaller without changing the logic of the code so that there is no pressure on memory.
2) Add a kernel in the compute section: Rewrite function `computation` to include a Pallas kernel definition and its invocation.

### Steps:
1. **Understand the Code**: Analyze the provided JAX code to understand its logic and data flow.
2. **Adjust Initialization**: Modify the tensor sizes in the initialization section to be smaller, ensuring that the logic remains unchanged. Pallas by default relies on VMEM (vector memory), which is significantly smaller than HBM (high bandwidth memory). Therefore, you should reduce the tensor sizes so that all operations should fit comfortably within the VMEM limits (specified in the TPU specs section).
3. **Analyze** the `computation` function to understand the operations and data access patterns.
4. **Define** the Pallas kernel function `kernel` that replaces the content in the `computation` function.
5. **Invoke** the Pallas kernel using `pl.pallas_call` in the rewritten `computation` function. To keep this as simple as possible, the `pallas_call` should only include the required arguments: `kernel` and `out_shape`. Do not include any other function arguments.

Important notes:
- If the `kernel` function's arguments include constants, then use `functools.partial` to bind those constants before passing the kernel to `pallas_call`.
- Do not use implement any optimizations or performance enhancements. Focus solely on correctness and ensuring the Pallas kernel matches the original JAX computation. This means, avoid using at a bare minimum `jax.vmap`, `jax.pmap`, `jax.jit`, `jax.pjit`, or any other JAX transformations that alter the execution model.
- There are many operations that are not supported within the kernel definition.

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
Input Code:
```python
# Imports
import jax
import jax.numpy as jnp
import jax.random as random

# Initialization
N = 4096
key = random.PRNGKey(0)
key_A, key_B = random.split(key)

A = random.normal(key_A, (N, N))
B = random.normal(key_B, (N, N))

# Computation
def computation(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    return jnp.matmul(A, B)

C = jax.block_until_ready(computation(A, B))
```

Basic Pallas Kernel:
```python
# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
N = 128
key = random.PRNGKey(0)
key_A, key_B = random.split(key)

A = random.normal(key_A, (N, N))
B = random.normal(key_B, (N, N))

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

### Example 2
Input Code:
```python
# Imports
import jax
import jax.numpy as jnp
import jax.random as random

# Initialization
N = 4096
multiplier = 2.0
key = random.PRNGKey(0)

A = random.normal(key, (N, N))

# Computation
def computation(A: jnp.ndarray, multiplier: float) -> jnp.ndarray:
    return A * multiplier

C = jax.block_until_ready(computation(A, multiplier))
```

Basic Pallas Kernel:
```python
# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl
from functools import partial

# Initialization
N = 128
multiplier = 2.0
key = random.PRNGKey(0)

A = random.normal(key, (N, N))

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

### Task
Input code:
{jax_base_code}

Basic Pallas Kernel:
<CODE TO GENERATE>
    
Your final output should only include the updated script.
"""
