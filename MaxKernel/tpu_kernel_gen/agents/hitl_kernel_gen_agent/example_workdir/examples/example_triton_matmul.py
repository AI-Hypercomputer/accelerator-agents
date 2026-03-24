"""
Example Triton kernel for matrix multiplication
Demonstrates a tiled matmul kernel that should be simplified to JAX
"""

import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
  a_ptr,
  b_ptr,
  c_ptr,
  M,
  N,
  K,
  stride_am,
  stride_ak,
  stride_bk,
  stride_bn,
  stride_cm,
  stride_cn,
  BLOCK_M: tl.constexpr,
  BLOCK_N: tl.constexpr,
  BLOCK_K: tl.constexpr,
):
  """Tiled matrix multiplication kernel."""
  # Get program ID
  pid_m = tl.program_id(0)
  pid_n = tl.program_id(1)

  # Compute offsets
  rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
  rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

  # Initialize accumulator
  acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

  # Loop over K dimension with tiling
  for k in range(0, K, BLOCK_K):
    rk = k + tl.arange(0, BLOCK_K)

    # Load tiles from A and B
    a = tl.load(a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak)
    b = tl.load(b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn)

    # Accumulate
    acc += tl.dot(a, b)

  # Store result
  c = acc.to(tl.float32)
  tl.store(c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn, c)


def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  """
  Matrix multiplication using Triton.

  Args:
      a: Input matrix of shape (M, K)
      b: Input matrix of shape (K, N)

  Returns:
      Output matrix of shape (M, N)
  """
  assert a.shape[1] == b.shape[0], "Incompatible dimensions"
  assert a.is_cuda and b.is_cuda, "Inputs must be on CUDA"

  M, K = a.shape
  K, N = b.shape

  # Allocate output
  c = torch.empty((M, N), device=a.device, dtype=a.dtype)

  # Define grid
  def grid(meta):
    return (
      triton.cdiv(M, meta["BLOCK_M"]),
      triton.cdiv(N, meta["BLOCK_N"]),
    )

  # Launch kernel
  matmul_kernel[grid](
    a,
    b,
    c,
    M,
    N,
    K,
    a.stride(0),
    a.stride(1),
    b.stride(0),
    b.stride(1),
    c.stride(0),
    c.stride(1),
    BLOCK_M=128,
    BLOCK_N=128,
    BLOCK_K=32,
  )

  return c


def main():
  """Test the Triton matmul kernel."""
  M, N, K = 1024, 1024, 1024

  # Create random inputs on GPU
  a = torch.randn(M, K, device="cuda", dtype=torch.float32)
  b = torch.randn(K, N, device="cuda", dtype=torch.float32)

  # Run Triton matmul
  c_triton = triton_matmul(a, b)

  # Verify against PyTorch
  c_torch = torch.matmul(a, b)

  # Check correctness
  max_diff = (c_triton - c_torch).abs().max().item()
  print(f"Max difference: {max_diff}")

  if max_diff < 1e-3:
    print("✓ Results match!")
  else:
    print("✗ Results differ!")


if __name__ == "__main__":
  main()
