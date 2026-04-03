# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
dim = 1
batch_size = 16
dim1 = 256
dim2 = 256
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, dim1, dim2))


# Computation
def kernel(x_ref, out_ref):
  """Pallas kernel to compute the minimum along axis=1.

  Args:
    x_ref: A reference to a slice of the input tensor of shape
      (b_block, dim1, d2_block).
    out_ref: A reference to the output tensor slice of shape
      (b_block, d2_block) where the result will be stored.
  """
  # Manual reduction to avoid loading the entire slice into SRAM at once.
  # Initialize the output with the first slice along the reduction axis.
  out_ref[...] = x_ref[:, 0, :]
  # Iterate over the rest of the reduction axis.
  for k in range(1, dim1):
    # Update the output with the element-wise minimum.
    out_ref[...] = jnp.minimum(out_ref[...], x_ref[:, k, :])


# We parallelize the reduction over the batch and dim2 axes.
# For TPU compatibility, block shapes need to be chosen carefully.
b_block = 8  # Block size for batch dimension, divisible by 8.
d2_block = 128  # Block size for dim2, divisible by 128.

result = pl.pallas_call(
  kernel,
  # The output shape is the input shape with the reduction axis removed.
  out_shape=jax.ShapeDtypeStruct((batch_size, dim2), x.dtype),
  # Create a 2D grid of program instances.
  grid=(batch_size // b_block, dim2 // d2_block),
  # Each program instance receives a (b_block, dim1, d2_block) slice.
  in_specs=[
    pl.BlockSpec(
      block_shape=(b_block, dim1, d2_block),
      index_map=lambda i, j: (i * b_block, 0, j * d2_block),
    )
  ],
  # Each program instance computes a (b_block, d2_block) slice.
  out_specs=pl.BlockSpec(
    block_shape=(b_block, d2_block),
    index_map=lambda i, j: (i * b_block, j * d2_block),
  ),
)(x).block_until_ready()
