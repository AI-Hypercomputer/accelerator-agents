# Imports
import jax
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
dim = 16384
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, dim))
block_dim = 128


def kernel(x_ref, out_ref):
  """Pallas kernel for the SiLU (Swish) activation function."""
  # Load the input block from SRAM.
  x = x_ref[...]
  # Compute the element-wise SiLU activation.
  # The result is written directly to the output buffer.
  out_ref[...] = x * jax.nn.sigmoid(x)


# Computation
# We define a block shape that is valid for TPU (divisible by 8 and 128).
# By omitting the index_map, we rely on Pallas's default tiling behavior,
# which is more robust and avoids the bounds-checking errors seen previously.
result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(batch_size // 8, dim // block_dim),
  in_specs=[pl.BlockSpec(block_shape=(8, block_dim))],
  out_specs=pl.BlockSpec(block_shape=(8, block_dim)),
)(x).block_until_ready()
