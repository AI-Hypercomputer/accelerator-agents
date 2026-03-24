# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl
from jax.experimental.pallas.ops import tpu as tpu_ops
from jax.experimental.pallas.ops.tpu import RmwOp

# Initialization
batch_size = 128
input_shape = (1,)
key = random.PRNGKey(0)
key_preds, key_targets = random.split(key)

predictions = random.normal(key_preds, (batch_size, *input_shape))
targets = random.randint(key_targets, (batch_size, 1), 0, 2) * 2 - 1


# Computation
def kernel(predictions_ref, targets_ref, out_ref):
  # Initialize the output reference to 0.0. This is done only by the first program.
  @pl.when(pl.program_id(0) == 0)
  def _():
    out_ref[:] = jnp.zeros(out_ref.shape, out_ref.dtype)

  # Compute the clipped values for the current block.
  clipped_values = jnp.clip(1 - predictions_ref[...] * targets_ref[...], a_min=0)
  # Sum the clipped values locally for the block.
  local_sum = jnp.sum(clipped_values)
  # Atomically add the local sum to the single output element.
  tpu_ops.atomic_rmw(out_ref, (0,), local_sum, op=RmwOp.ADD)


# The kernel would compute the sum of clipped values, which is then divided by the total number of elements.
# The pallas_call computes the sum into a single-element array, which is then used to compute the mean.
sum_val = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((1,), predictions.dtype),
  grid=(16,),
  in_specs=[
    pl.BlockSpec(block_shape=(8, 1), index_map=lambda i: (i * 8, 0)),
    pl.BlockSpec(block_shape=(8, 1), index_map=lambda i: (i * 8, 0)),
  ],
  out_specs=pl.BlockSpec(block_shape=(1,), index_map=lambda i: (0,)),
)(predictions, targets).block_until_ready()
result = sum_val[0] / predictions.size
