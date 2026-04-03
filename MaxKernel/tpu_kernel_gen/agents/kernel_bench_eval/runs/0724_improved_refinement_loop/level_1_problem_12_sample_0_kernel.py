# Imports
import jax
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
M = 4096
N = 4096
key = random.PRNGKey(0)
key_A, key_B = random.split(key)
A = random.normal(key_A, (N,))
B = random.normal(key_B, (N, M))
block_M = 128
block_N = 128

# Reshape A to be 2D to simplify broadcasting within the kernel.
A_2d = A.reshape((N, 1))


# Computation
def kernel(a_ref, b_ref, out_ref):
  """Pallas kernel for element-wise multiplication with broadcasting."""
  # The core computation is A[:, None] * B.
  # `a_ref` is a block from A_2d with shape (block_N, 1).
  # `b_ref` is a block from B with shape (block_N, block_M).
  # The shapes are already set up for broadcasting.
  out_ref[...] = a_ref[...] * b_ref[...]


result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(B.shape, B.dtype),
  grid=(N // block_N, M // block_M),
  in_specs=[
    pl.BlockSpec(block_shape=(block_N, 1), index_map=lambda i, j: (i, 0)),
    pl.BlockSpec(block_shape=(block_N, block_M), index_map=lambda i, j: (i, j)),
  ],
  out_specs=pl.BlockSpec(block_shape=(block_N, block_M), index_map=lambda i, j: (i, j)),
)(A_2d, B).block_until_ready()
