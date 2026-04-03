PROMPT = """
You are an expert programmer specializing in JAX and Flax, with a deep understanding of idiomatic, high-performance code.

Your objective is to refactor a given Python script into its JAX and Flax equivalent. Focus on simplicity and correctness over performance. The original script is structured with comments dividing it into `# Imports`, `# Initialization`, and `# Computation` sections. You must preserve this exact structure and commenting style in your output.

Here is an example of how this should look:

Original code:
```python
# Imports
import torch

# Initialization
N = 2048
A = torch.randn(N, N)
B = torch.randn(N, N)

# Computation
def computation(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.matmul(A, B)

C = computation(A, B)
```

Jax equivalent:
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

Now, refactor the following code based on these instructions. Make sure to only return the refactored code, without any additional comments or explanations.
Original code:
{organized_code}

Jax equivalent:
<CODE TO GENERATE>
"""
