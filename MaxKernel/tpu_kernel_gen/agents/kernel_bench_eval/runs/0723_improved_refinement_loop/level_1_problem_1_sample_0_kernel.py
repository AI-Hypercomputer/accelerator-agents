# Imports
import jax
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
N = 2048
key = random.PRNGKey(0)
key_A, key_B = random.split(key)

A = random.normal(key_A, (N, N))
B = random.normal(key_B, (N, N))

bN = 128


# Computation
def kernel(x_ref, y_ref, z_ref):
  """Pallas kernel for matrix multiplication."""
  # The invocation for this kernel is not a standard tiled matmul,
  # but rather a block-wise multiplication. It loads a (bN, N) row-panel of x
  # and a (N, bN) column-panel of y. The matmul of these two panels
  # directly produces the (bN, bN) output block.
  # Therefore, no accumulation loop is needed within the kernel.
  z_ref[...] = x_ref[...] @ y_ref[...]


C = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(A.shape, A.dtype),
  grid=(N // bN, N // bN),
  in_specs=[
    pl.BlockSpec(block_shape=(bN, N), index_map=lambda i, j: (i, 0)),
    pl.BlockSpec(block_shape=(N, bN), index_map=lambda i, j: (0, j)),
  ],
  out_specs=pl.BlockSpec(block_shape=(bN, bN), index_map=lambda i, j: (i, j)),
)(A, B).block_until_ready()
