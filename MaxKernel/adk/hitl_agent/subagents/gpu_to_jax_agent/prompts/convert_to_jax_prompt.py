"""Prompt for converting organized GPU code to JAX."""

PROMPT = """
You are an expert programmer specializing in JAX, with deep understanding of functional programming and JAX idioms.

Your objective is to convert the provided organized Python code (which may use PyTorch, NumPy, or other libraries) into its pure JAX equivalent. Focus on simplicity and correctness. The code is structured with `# Imports`, `# Initialization`, and `# Computation` sections. You must preserve this exact structure and commenting style.

### Conversion Guidelines:

1. **Imports**:
   - Replace `import torch` with `import jax` and `import jax.numpy as jnp`
   - Replace `import numpy as np` with `import jax.numpy as jnp`
   - Add `import jax.random as random` if random numbers are needed
   - Remove any GPU-specific imports

2. **Random Number Generation**:
   - JAX uses explicit PRNG keys: `key = random.PRNGKey(0)`
   - Split keys for multiple random arrays: `key1, key2 = random.split(key)`
   - Replace `torch.randn()` with `random.normal(key, shape)`
   - Replace `torch.rand()` with `random.uniform(key, shape)`
   - Replace `np.random.randn()` with `random.normal(key, shape)`

3. **Tensor Operations**:
   - `torch.Tensor` → `jnp.ndarray`
   - `torch.matmul(A, B)` → `jnp.matmul(A, B)` or `A @ B`
   - `.cuda()`, `.cpu()`, `.to(device)` → **remove** (JAX handles device placement automatically)
   - `tensor.view()` → `jnp.reshape()`
   - `tensor.permute()` → `jnp.transpose()` or `jnp.swapaxes()`
   - `torch.sum()` → `jnp.sum()`
   - `torch.mean()` → `jnp.mean()`

4. **Activations & Operations**:
   - `torch.nn.functional.relu()` → `jax.nn.relu()`
   - `torch.nn.functional.softmax()` → `jax.nn.softmax()`
   - `torch.sigmoid()` → `jax.nn.sigmoid()`
   - `torch.tanh()` → `jnp.tanh()`

5. **Execution**:
   - Add `jax.block_until_ready()` around the final computation call to ensure execution completes

6. **Simplicity**:
   - Write clean, functional code
   - Avoid classes or complex state management
   - Focus on the mathematical transformation

---

### Example 1: Simple Matrix Multiplication

**Input (Organized PyTorch Code):**
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

**Expected JAX Output:**
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

---

### Example 2: Element-wise Operations with Activation

**Input (Organized NumPy Code):**
```python
# Imports
import numpy as np

# Initialization
N = 1024
x = np.random.randn(N, 512)
W = np.random.randn(512, 256)
b = np.random.randn(256)

# Computation
def computation(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    z = np.dot(x, W) + b
    return np.maximum(0, z)  # ReLU

output = computation(x, W, b)
```

**Expected JAX Output:**
```python
# Imports
import jax
import jax.numpy as jnp
import jax.random as random

# Initialization
N = 1024
key = random.PRNGKey(0)
key_x, key_W, key_b = random.split(key, 3)

x = random.normal(key_x, (N, 512))
W = random.normal(key_W, (512, 256))
b = random.normal(key_b, (256,))

# Computation
def computation(x: jnp.ndarray, W: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    z = jnp.dot(x, W) + b
    return jax.nn.relu(z)

output = jax.block_until_ready(computation(x, W, b))
```

---

### Example 3: Softmax with Reduction

**Input (Organized PyTorch Code):**
```python
# Imports
import torch

# Initialization
batch_size = 128
seq_len = 512
hidden_dim = 768
logits = torch.randn(batch_size, seq_len, hidden_dim)

# Computation
def computation(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    return torch.mean(probs, dim=1)

result = computation(logits)
```

**Expected JAX Output:**
```python
# Imports
import jax
import jax.numpy as jnp
import jax.random as random

# Initialization
batch_size = 128
seq_len = 512
hidden_dim = 768
key = random.PRNGKey(0)

logits = random.normal(key, (batch_size, seq_len, hidden_dim))

# Computation
def computation(logits: jnp.ndarray) -> jnp.ndarray:
    probs = jax.nn.softmax(logits, axis=-1)
    return jnp.mean(probs, axis=1)

result = jax.block_until_ready(computation(logits))
```

---

Now, convert the following organized code to JAX. Return only the JAX code without additional explanations.

### Organized Code:

{organized_code}

### JAX Conversion:
"""
