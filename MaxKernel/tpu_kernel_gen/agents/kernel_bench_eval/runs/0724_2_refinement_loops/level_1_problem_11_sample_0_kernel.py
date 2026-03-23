# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
b = 16
i = 256
j = 512
l = 256
k = 768
key = random.PRNGKey(0)
key_A, key_B = random.split(key)
A = random.normal(key_A, (b, i, j, l))
B = random.normal(key_B, (l, k))


def kernel(a_ref, b_ref, c_ref):
  """Pallas kernel for batched matrix multiplication.

  This kernel computes C[b, i, :, :] = A[b, i, :, :] @ B for each (b, i) in the grid.

  Args:
    a_ref: A reference to a slice of A with shape (1, 1, j, l).
    b_ref: A reference to the entire B matrix with shape (l, k).
    c_ref: A reference to a slice of the output C with shape (1, 1, j, k),
      which will be written to in-place.
  """
  # The einsum "bijl,lk->bijk" is a batched matrix multiplication. Each kernel
  # instance handles one matrix multiplication from the batch.
  # a_ref has shape (1, 1, j, l), so we extract the (j, l) matrix.
  # b_ref has shape (l, k).
  # The result of the matmul is a (j, k) matrix.
  result = a_ref[0, 0] @ b_ref[...]

  # We write the (j, k) result into the corresponding slice of the output.
  # c_ref[0, 0] refers to the (j, k) slice within the (1, 1, j, k) block.
  c_ref[0, 0] = result


# Computation
# The einsum "bijl,lk->bijk" represents a batched matrix multiplication.
# The computation is independent for each element in the batch dimensions 'b' and 'i'.
# We can define a 2D grid of (b, i) to parallelize the computation.
# Each kernel instance (b_idx, i_idx) will be responsible for:
# C[b_idx, i_idx, :, :] = A[b_idx, i_idx, :, :] @ B[:, :]
C = pl.pallas_call(
  kernel,
  # The output shape is (b, i, j, k)
  out_shape=jax.ShapeDtypeStruct(shape=(b, i, j, k), dtype=jnp.float32),
  # Grid iterates over the batch dimensions 'b' and 'i'.
  grid=(b, i),
  in_specs=[
    # For A, each kernel instance (b_idx, i_idx) gets the corresponding (j, l) slice.
    pl.BlockSpec(block_shape=(1, 1, j, l), index_map=lambda b_idx, i_idx: (b_idx, i_idx, 0, 0)),
    # For B, the entire (l, k) matrix is passed to every kernel instance.
    pl.BlockSpec(block_shape=(l, k), index_map=lambda b_idx, i_idx: (0, 0)),
  ],
  out_specs=pl.BlockSpec(
    # Each kernel instance writes its (j, k) result to the correct slice in the output.
    block_shape=(1, 1, j, k),
    index_map=lambda b_idx, i_idx: (b_idx, i_idx, 0, 0),
  ),
)(A, B).block_until_ready()
