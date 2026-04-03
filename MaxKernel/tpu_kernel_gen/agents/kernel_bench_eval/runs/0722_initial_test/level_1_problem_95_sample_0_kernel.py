# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 4096
num_classes = 10
block_size = 128
key = random.PRNGKey(0)
key_preds, key_targets = random.split(key)

predictions = random.normal(key_preds, (batch_size, num_classes))
targets = random.randint(key_targets, (batch_size,), 0, num_classes)


# Computation
def kernel(predictions_ref, targets_ref, losses_ref):
  """Pallas kernel for computing softmax cross-entropy loss."""
  # Load the block of data from HBM into SRAM/registers.
  predictions = predictions_ref[...]  # Shape: (block_size, num_classes)
  targets = targets_ref[...]  # Shape: (block_size,)

  # The cross-entropy loss is defined as:
  # loss = -log(softmax(logits))[target_class]
  # A numerically stable way to compute this is:
  # loss = log(sum(exp(logits))) - logits[target_class]

  # 1. Calculate log(sum(exp(logits))) in a stable way.
  #    This is the log-sum-exp trick.
  max_logits = jnp.max(predictions, axis=1)
  # Subtract the max for stability before exponentiating.
  # Use `[:, None]` to ensure correct broadcasting.
  shifted_logits = predictions - max_logits[:, None]
  # Calculate the log of the sum of the exponentiated shifted logits.
  log_sum_exp = max_logits + jnp.log(jnp.sum(jnp.exp(shifted_logits), axis=1))

  # 2. Gather the logits corresponding to the correct target classes.
  #    We need to select predictions[i, targets[i]] for each row i.
  row_indices = jnp.arange(predictions.shape[0])
  target_logits = predictions[row_indices, targets]

  # 3. Compute the final loss for each example in the block.
  losses = log_sum_exp - target_logits

  # 4. Write the computed losses back to the output buffer.
  losses_ref[...] = losses


losses = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size,), predictions.dtype),
  grid=(batch_size // block_size,),
  in_specs=[
    pl.BlockSpec(block_shape=(block_size, num_classes), index_map=lambda i: (i, 0)),
    pl.BlockSpec(block_shape=(block_size,), index_map=lambda i: (i,)),
  ],
  out_specs=pl.BlockSpec(block_shape=(block_size,), index_map=lambda i: (i,)),
)(predictions, targets)
loss = losses.mean().block_until_ready()
