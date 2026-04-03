# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
input_shape = (4096,)
key = random.PRNGKey(0)
key_preds, key_targets = random.split(key)
predictions = random.normal(key_preds, (batch_size, *input_shape))
targets = random.normal(key_targets, (batch_size, *input_shape))


# Computation
def kernel(preds_ref, targets_ref, out_ref):
  """Pallas kernel to compute cosine similarity loss."""
  # Load data from HBM to SRAM.
  # preds_ref and targets_ref have shape (batch_size, input_shape[0])
  predictions = preds_ref[...]
  targets = targets_ref[...]

  # Compute dot product for each vector in the block.
  # The result will have shape (batch_size,).
  dot_product = jnp.sum(predictions * targets, axis=1)

  # Compute L2 norm for each prediction vector in the block.
  norm_preds = jnp.linalg.norm(predictions, axis=1)
  # Compute L2 norm for each target vector in the block.
  norm_targets = jnp.linalg.norm(targets, axis=1)

  # Compute cosine similarity. Add a small epsilon for numerical stability.
  cosine_sim = dot_product / (norm_preds * norm_targets + 1e-8)

  # The final loss for each item is 1 - cosine_similarity.
  # Write the result to the output block in HBM.
  out_ref[:] = 1.0 - cosine_sim


one_minus_cosine_sim = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size,), predictions.dtype),
  grid=(1,),
  in_specs=[
    pl.BlockSpec(block_shape=(batch_size, input_shape[0]), index_map=lambda i: (0, 0)),
    pl.BlockSpec(block_shape=(batch_size, input_shape[0]), index_map=lambda i: (0, 0)),
  ],
  out_specs=pl.BlockSpec(block_shape=(batch_size,), index_map=lambda i: (0,)),
)(predictions, targets)
loss = jnp.mean(one_minus_cosine_sim).block_until_ready()
