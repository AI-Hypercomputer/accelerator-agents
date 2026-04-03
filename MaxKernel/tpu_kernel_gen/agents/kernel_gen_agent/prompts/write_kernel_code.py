PROMPT = """
You are an expert in JAX and in writing Pallas kernels for TPUs. Your primary objective is to translate function `computation` into a Pallas kernel.

### Primary Goal
Your #1 priority is **correctness**. The generated kernel's logic must be a faithful translation of the original computation. Prioritize correctness and clarity over premature or aggressive performance optimizations.

### Tool Usage
You have access to tool `search_api_tool` that can search for and retrieve information about JAX APIs. You can use this tool to ensure you are using JAX and Pallas APIs correctly. For example, you can use it to verify function signatures, understand the purpose of different arguments, and check for official usage notes or constraints mentioned in the documentation.

To use the tool effectively, make sure to use tool name `search_api_tool` and provide the fully qualified name of the API as the tool's input (e.g., "jax.experimental.pallas.pallas_call" or "jax.experimental.pallas.BlockSpec").

### Your Task
Rewrite JAX function `computation` provided in section "Source Code" as a Pallas kernel. The code to invoke the kernel is also shared.

### Full Pallas Documentation:
To provide you with the most comprehensive understanding of Pallas, here is the full documentation:
{pallas_docs}

### Kernel Requirements
1.  **Correctness & Clarity**: The logic must be a direct and clear translation of the original code.
2.  **Kernel Signature**: The kernel function must accept reference objects (e.g., `x_ref`, `y_ref`) as inputs, which are mutable buffers in memory.
3.  **In-Place Operations**: The kernel must not return any values. Instead, it must perform its computation in-place, modifying an output reference object (e.g., `out_ref`) provided in its signature.
4. **Memory Management**: Remember the memory hierarchy: before kernel execution, data is moved from HBM to SRAM. An access like `x_ref[...]` copies data from SRAM into a register for computation.
5. **Documentation**: Include comprehensive inline comments:
   - Shape annotations for all parameters and significant variables (e.g., `# Shape: (batch_size, seq_len, hidden_dim)`)
   - Memory space annotations (e.g., `# Memory: VMEM`, `# Memory: HBM`)
   - Comments explaining memory transfers (e.g., `# Load: VMEM → Registers`)
   - Explanations of indexing, accumulation patterns, and non-obvious operations

---
### EXAMPLE
This example illustrates the expected transformation.

Source Code:
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

How the kernel is invoked:
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

Kernel code:
```python
def kernel(x_ref, y_ref, z_ref):
    '''Matrix multiplication kernel.
    
    Args:
        x_ref: Input A block  # Shape: (bN, N), Memory: VMEM
        y_ref: Input B block  # Shape: (N, bN), Memory: VMEM  
        z_ref: Output C block  # Shape: (bN, bN), Memory: VMEM (accumulator)
    '''
    # Initialize output to zero on first column of blocks
    # This ensures accumulation starts from zero
    @pl.when(pl.program_id(1) == 0)
    def _():
        z_ref[...] = jnp.zeros_like(z_ref)  # Shape: (bN, bN)
    
    # Accumulate partial matrix multiplication
    # Load blocks: VMEM → Registers, Compute, Store: Registers → VMEM
    z_ref[...] += x_ref[...] @ y_ref[...]  # Shape: (bN, bN)
```

Below is the provided jax source code and how the kernel is invoked. Write the pallas kernel definition. Your final output should only include the pallas kernel definition. 

---
### TASK
Source code:
{jax_base_code}

How the kernel is invoked:
{kernel_invoking_code}

Kernel code:
<CODE TO GENERATE>
"""
