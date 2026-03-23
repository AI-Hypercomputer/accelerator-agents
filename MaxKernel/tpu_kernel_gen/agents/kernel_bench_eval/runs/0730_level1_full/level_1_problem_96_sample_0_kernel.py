# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
input_shape = (4096,)
key = random.PRNGKey(0)
key_pred, key_targets = random.split(key)
predictions = random.normal(key_pred, (batch_size, *input_shape))
targets = random.normal(key_targets, (batch_size, *input_shape))
b_b = 8
b_i = 128


# Computation
def kernel(predictions_ref, targets_ref, loss_ref):
  """Pallas kernel to compute the mean Huber loss."""
  # Hardcoded values from the problem description's context.
  # In a real-world scenario, these might be passed in via GridSpec.
  batch_size = 128
  input_shape = (4096,)
  total_elements = float(batch_size * input_shape[0])

  # The first program instance initializes the output scalar to zero.
  # Since the grid is 2D, we check both program IDs.
  @pl.when((pl.program_id(0) == 0) & (pl.program_id(1) == 0))
  def _init_loss():
    loss_ref[...] = jnp.zeros_like(loss_ref)

  # Load the input blocks from SRAM into registers.
  predictions = predictions_ref[...]
  targets = targets_ref[...]

  # Calculate element-wise Huber loss for the block (with delta=1.0).
  error = predictions - targets
  abs_error = jnp.abs(error)
  # The two parts of the Huber loss formula.
  quadratic_loss = 0.5 * jnp.square(error)
  linear_loss = abs_error - 0.5
  # Choose the appropriate loss value for each element.
  block_loss_values = jnp.where(abs_error <= 1.0, quadratic_loss, linear_loss)

  # Sum the losses within the current block.
  block_loss_sum = jnp.sum(block_loss_values)

  # To compute the mean, each block contributes its partial sum divided by the
  # total number of elements to the final result. This is an atomic add
  # because all program instances write to the same scalar output `loss_ref`.
  loss_ref[...] += block_loss_sum / total_elements


loss = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((1,), predictions.dtype),
  grid=(batch_size // b_b, input_shape[0] // b_i),
  in_specs=[
    pl.BlockSpec(block_shape=(b_b, b_i), index_map=lambda i, j: (i * b_b, j * b_i)),
    pl.BlockSpec(block_shape=(b_b, b_i), index_map=lambda i, j: (i * b_b, j * b_i)),
  ],
  out_specs=pl.BlockSpec(block_shape=(1,), index_map=lambda i, j: (0,)),
)(predictions, targets)
