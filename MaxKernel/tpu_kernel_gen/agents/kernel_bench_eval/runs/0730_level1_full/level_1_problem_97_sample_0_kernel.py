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


# Computation
def kernel(predictions_ref, targets_ref, loss_ref):
  """Pallas kernel to compute cosine similarity loss."""
  # This single program instance computes the loss for the entire batch.

  # Load the data for the entire batch from SRAM into registers.
  predictions = predictions_ref[...]
  targets = targets_ref[...]

  # --- Start of original computation translation ---

  # 1. Compute dot product for each item in the batch
  dot_product = jnp.sum(predictions * targets, axis=1)

  # 2. Compute norms for each item in the batch
  norm_predictions = jnp.linalg.norm(predictions, axis=1)
  norm_targets = jnp.linalg.norm(targets, axis=1)

  # 3. Compute cosine similarity for each item in the batch
  # The denominator is not protected against being zero to faithfully
  # replicate the original computation.
  cosine_sim = dot_product / (norm_predictions * norm_targets)

  # 4. Compute loss for each item
  item_loss = 1.0 - cosine_sim

  # 5. Compute the mean loss over the batch
  mean_loss = jnp.mean(item_loss)

  # --- End of original computation translation ---

  # Write the final mean loss to the output buffer.
  loss_ref[0] = mean_loss


loss = pl.pallas_call(
  kernel,
  # The output is a single scalar value, but we use a 1-element array
  # to satisfy TPU constraints (output rank must be >= 1).
  out_shape=jax.ShapeDtypeStruct((1,), predictions.dtype),
  # A 1x1 grid means we launch a single program instance.
  grid=(1,),
  in_specs=[
    # The entire input arrays are treated as a single block.
    pl.BlockSpec(block_shape=(batch_size, input_shape[0]), index_map=lambda i: (0, 0)),
    pl.BlockSpec(block_shape=(batch_size, input_shape[0]), index_map=lambda i: (0, 0)),
  ],
  out_specs=pl.BlockSpec(block_shape=(1,), index_map=lambda i: (0,)),
)(predictions, targets)[0].block_until_ready()
