# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
negative_slope = 0.01
batch_size = 16
dim = 16384
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, dim))


# Computation
def kernel(x_ref, out_ref, negative_slope: float):
  """Pallas kernel for leaky ReLU.

  Args:
    x_ref: Input block reference.
    out_ref: Output block reference.
    negative_slope: The negative slope for the leaky ReLU activation.
  """
  x = x_ref[...]
  # Apply the leaky ReLU activation function element-wise.
  # This is equivalent to jnp.maximum(x, x * negative_slope).
  result = jnp.where(x > 0, x, x * negative_slope)
  out_ref[...] = result


# The leaky_relu kernel is invoked over blocks of the input array.
# We define a 2D grid to parallelize the computation across both the
# batch and dimension axes of the input tensor `x`.
#
# The block shape for the computation is (8, 128). This shape is chosen
# to comply with TPU hardware constraints:
# - The second-to-last dimension of the block (8) is divisible by 8.
# - The last dimension of the block (128) is divisible by 128.
#
# The execution grid is calculated by dividing the full array shape
# by the block shape, resulting in a (16/8, 16384/128) = (2, 128) grid.
#
# The `index_map` `lambda i, j: (i, j)` maps each grid instance `(i, j)`
# to a unique (8, 128) block in both the input `x` and the output `result`,
# ensuring that each element is processed exactly once.
result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(batch_size // 8, dim // 128),
  in_specs=[pl.BlockSpec(block_shape=(8, 128), index_map=lambda i, j: (i, j))],
  out_specs=pl.BlockSpec(block_shape=(8, 128), index_map=lambda i, j: (i, j)),
)(x, negative_slope=negative_slope).block_until_ready()
