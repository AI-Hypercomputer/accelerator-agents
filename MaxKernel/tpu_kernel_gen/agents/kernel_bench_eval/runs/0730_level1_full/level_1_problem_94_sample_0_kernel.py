# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
input_shape = (4096,)
key = random.PRNGKey(0)
key_predictions, key_targets = random.split(key)

predictions = random.normal(key_predictions, (batch_size, *input_shape))
targets = random.normal(key_targets, (batch_size, *input_shape))

block_batch = 32
block_input = 256


# Computation
def kernel(predictions_ref, targets_ref, out_ref):
  """Pallas kernel to compute element-wise squared error."""
  # This kernel is executed in a grid. Each instance computes the
  # element-wise squared difference for a block of the input data.

  # 1. Compute the element-wise squared difference for the current block.
  squared_diff = (predictions_ref[...] - targets_ref[...]) ** 2

  # 2. Write the result to the output buffer.
  out_ref[...] = squared_diff


# The pallas_call computes the element-wise squared difference.
squared_differences = pl.pallas_call(
  kernel,
  # The output has the same shape as the input.
  out_shape=jax.ShapeDtypeStruct(predictions.shape, predictions.dtype),
  grid=(batch_size // block_batch, input_shape[0] // block_input),
  in_specs=[
    pl.BlockSpec((block_batch, block_input), lambda i, j: (i, j)),
    pl.BlockSpec((block_batch, block_input), lambda i, j: (i, j)),
  ],
  # The output BlockSpec mirrors the input BlockSpecs.
  out_specs=pl.BlockSpec((block_batch, block_input), lambda i, j: (i, j)),
)(predictions, targets)

# We sum the results and then compute the mean outside the kernel.
result = (jnp.sum(squared_differences) / predictions.size).block_until_ready()
