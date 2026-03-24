# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
input_shape = (1,)
key = random.PRNGKey(0)
key_pred, key_targets = random.split(key)

predictions = random.normal(key_pred, (batch_size, *input_shape))
targets = random.randint(key_targets, (batch_size, 1), 0, 2) * 2 - 1


# Computation
def kernel(predictions_ref, targets_ref, out_ref):
  # Perform the element-wise computation
  result = 1 - predictions_ref[...] * targets_ref[...]
  # Clip the result at 0
  clipped_result = jnp.clip(result, a_min=0)
  # Write the element-wise result to the output
  out_ref[...] = clipped_result


# The pallas call now computes the element-wise clipped result
clipped_result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(predictions.shape, predictions.dtype),
  grid=(1,),
  in_specs=[
    pl.BlockSpec(block_shape=predictions.shape, index_map=lambda i: (0, 0)),
    pl.BlockSpec(block_shape=targets.shape, index_map=lambda i: (0, 0)),
  ],
  out_specs=pl.BlockSpec(block_shape=predictions.shape, index_map=lambda i: (0, 0)),
)(predictions, targets)

# The final reduction is done outside the pallas kernel
result = jnp.mean(clipped_result).block_until_ready()
