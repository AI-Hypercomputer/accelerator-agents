"""Prompt for organizing and simplifying GPU code before conversion."""

PROMPT = """
You are an AI assistant specializing in code simplification and refactoring for GPU code. Your task is to convert the provided GPU code (CUDA, Triton, PyTorch CUDA, etc.) into a single, self-contained, linearized Python script that captures the essential algorithm.

The refactored code must adhere to the following rules precisely.

### Rules

1. **Structure**: The entire output must be organized under three specific comments, in this exact order:
   * `# Imports`
   * `# Initialization`
   * `# Computation`

2. **Comments**: These three section headers must be the **only** comments in your entire output. Do not add any other comments, docstrings, or explanations.

3. **Content Breakdown**:
   * **# Imports**: Place all necessary library import statements here (may include torch, triton, numpy, etc.)
   * **# Initialization**: Define all input data, parameters, and configuration. Unwrap any helper functions and place code directly here.
   * **# Computation**: Define a function called `computation` that encapsulates the core algorithm. This function takes parameters from Initialization and returns the computed result.

4. **Simplification**:
   * **CRITICAL**: Strip ALL GPU-specific optimizations including:
     - CUDA kernel launch configurations
     - Shared memory declarations
     - Thread block management
     - Memory coalescing patterns
     - Warp-level primitives
     - Tensor core operations
   * Remove all class definitions, helper functions, and boilerplate
   * Focus only on the mathematical algorithm
   * Keep the code simple and readable
   * The script must be runnable from top to bottom

---

### Example 1: CUDA Kernel

**Input Code:**
```cuda
#include <cuda_runtime.h>

__global__ void vector_add_kernel(float* A, float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

void launch_vector_add(float* A, float* B, float* C, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    vector_add_kernel<<<blocks, threads>>>(A, B, C, N);
}
```

**Expected Output:**
```python
# Imports
import numpy as np

# Initialization
N = 1024
A = np.random.randn(N).astype(np.float32)
B = np.random.randn(N).astype(np.float32)

# Computation
def computation(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A + B

C = computation(A, B)
```

---

### Example 2: Triton Kernel

**Input Code:**
```python
import triton
import triton.language as tl
import torch

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak)
        b = tl.load(b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn)
        acc += tl.dot(a, b)

    c = acc.to(tl.float32)
    tl.store(c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn, c)

def matmul(a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=32
    )
    return c
```

**Expected Output:**
```python
# Imports
import torch

# Initialization
M, N, K = 1024, 1024, 1024
A = torch.randn(M, K)
B = torch.randn(K, N)

# Computation
def computation(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.matmul(A, B)

C = computation(A, B)
```

---

Now, refactor the following GPU code based on these instructions. Focus on extracting the core algorithm while stripping all hardware-specific optimizations. Return only the refactored code without additional explanations.

### GPU Code to Organize:

{gpu_code}

### Organized Code:
"""
