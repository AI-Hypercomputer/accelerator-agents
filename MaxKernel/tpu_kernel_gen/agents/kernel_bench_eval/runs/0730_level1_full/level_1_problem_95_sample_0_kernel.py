# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl
from jax.scipy.special import logsumexp

# Initialization
batch_size = 4096
num_classes = 10
key = random.PRNGKey(0)
key_preds, key_targets = random.split(key)

predictions = random.normal(key_preds, (batch_size, num_classes))
targets = random.randint(key_targets, (batch_size,), 0, num_classes)

b_size = 512


# Computation
def kernel(predictions_ref, targets_ref, loss_ref):
  """Pallas kernel for softmax cross-entropy with integer labels.

  This kernel computes the cross-entropy loss for a block of predictions and
  integer targets. The formula used is log(sum(exp(logits))) - logits[correct_class],
  which is numerically stable.

  Args:
    predictions_ref: A reference to a block of logits of shape
      [b_size, num_classes].
    targets_ref: A reference to a block of integer labels of shape [b_size,].
    loss_ref: An output reference to store the computed losses, of shape [b_size,].
  """
  # Load the input blocks from SRAM into registers.
  predictions = predictions_ref[...]
  targets = targets_ref[...]

  # Calculate log(sum(exp(logits))) for each example in the block.
  # This is the log-normalizer term of the softmax.
  lse = logsumexp(predictions, axis=1)

  # Gather the logit corresponding to the correct integer label for each example.
  # We create a one-hot encoding of the targets via broadcasting, which is
  # more compatible with Pallas on TPU than direct indexing or `one_hot`.
  targets_one_hot = jnp.arange(num_classes, dtype=targets.dtype) == targets[:, None]
  correct_class_logits = jnp.sum(predictions * targets_one_hot.astype(predictions.dtype), axis=1)

  # Calculate the cross-entropy loss for each example in the block.
  loss = lse - correct_class_logits

  # Write the resulting block of losses to the output reference.
  loss_ref[...] = loss


loss = (
  pl.pallas_call(
    kernel,
    out_shape=jax.ShapeDtypeStruct((batch_size,), predictions.dtype),
    grid=(batch_size // b_size,),
    in_specs=[
      pl.BlockSpec(block_shape=(b_size, num_classes), index_map=lambda i: (i, 0)),
      pl.BlockSpec(block_shape=(b_size,), index_map=lambda i: (i,)),
    ],
    out_specs=pl.BlockSpec(block_shape=(b_size,), index_map=lambda i: (i,)),
  )(predictions, targets)
  .mean()
  .block_until_ready()
)
