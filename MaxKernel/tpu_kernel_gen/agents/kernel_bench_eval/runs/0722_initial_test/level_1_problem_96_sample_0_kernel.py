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


# Computation
def kernel(predictions_ref, targets_ref, out_ref):
  """Pallas kernel for computing the element-wise smooth L1 loss.

  This kernel processes a block of predictions and targets and computes the
  element-wise smooth L1 loss, writing the resulting block to the output.

  Args:
    predictions_ref: A reference to a block of the predictions array.
    targets_ref: A reference to a block of the targets array.
    out_ref: A reference to a block of the output array.
  """
  # Load the input blocks from SRAM into registers.
  predictions = predictions_ref[...]
  targets = targets_ref[...]

  # Calculate the difference between predictions and targets.
  diff = predictions - targets

  # Calculate the absolute difference.
  abs_diff = jnp.abs(diff)

  # Apply the smooth L1 loss formula element-wise.
  # The formula is:
  # - 0.5 * x^2 if |x| < 1
  # - |x| - 0.5 otherwise
  loss = jnp.where(abs_diff < 1.0, 0.5 * diff * diff, abs_diff - 0.5)

  # Write the block of losses to the output. The reduction will be done
  # outside the kernel.
  out_ref[...] = loss


# Define block sizes for chunking the data.
# These are chosen to be compatible with TPU hardware constraints.
# The first dimension of the block is a multiple of 8.
# The second dimension of the block is a multiple of 128.
b_size = 8
i_size = 128

# The `pallas_call` replaces the original computation.
# It computes the element-wise smooth L1 loss for each block of data.
# The final reduction (summing the losses and dividing by the total number of elements)
# would happen after this call.
partial_losses = pl.pallas_call(
  kernel,
  # The output shape is the same as the input shape.
  out_shape=jax.ShapeDtypeStruct(predictions.shape, jnp.float32),
  # The grid defines the parallelism. We create a 2D grid that tiles the input arrays.
  grid=(predictions.shape[0] // b_size, predictions.shape[1] // i_size),
  # in_specs defines how to slice the input arrays for each kernel.
  in_specs=[
    pl.BlockSpec(block_shape=(b_size, i_size), index_map=lambda i, j: (i * b_size, j * i_size)),
    pl.BlockSpec(block_shape=(b_size, i_size), index_map=lambda i, j: (i * b_size, j * i_size)),
  ],
  # out_specs defines where each kernel instance writes its output block.
  out_specs=pl.BlockSpec(block_shape=(b_size, i_size), index_map=lambda i, j: (i * b_size, j * i_size)),
)(predictions, targets).block_until_ready()
