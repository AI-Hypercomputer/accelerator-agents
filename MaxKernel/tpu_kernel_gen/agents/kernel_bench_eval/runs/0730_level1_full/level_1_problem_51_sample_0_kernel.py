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
  """Pallas kernel for computing argmax along a specified axis.

  Args:
    x_ref: A reference to a slice of the input tensor x. For each program,
      this slice corresponds to a block of shape (8, dim1, dim2).
    out_ref: A reference to the output slice. For each program, this corresponds
      to a block of shape (8, dim2).
  """
  # The original operation is jnp.argmax(x, axis=1).
  # Each kernel instance processes a block of 8 batches.
  # We iterate through each batch in the block.
  for b in range(x_ref.shape[0]):
    # Manual implementation of argmax as it's not supported on TPU.
    # We find the argmax for each slice x_ref[b, :, :].
    max_vals = jnp.full(x_ref.shape[2], -jnp.inf, dtype=x_ref.dtype)
    idxs = jnp.zeros(x_ref.shape[2], jnp.int32)
    for i in range(x_ref.shape[1]):
      row = x_ref[b, i, :]
      is_new_max = row > max_vals
      idxs = jnp.where(is_new_max, i, idxs)
      max_vals = jnp.maximum(max_vals, row)
    out_ref[b, :] = idxs


# Each kernel instance handles a block of 8 batches to satisfy TPU layout constraints.
grid_size = batch_size // 8
result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, dim2), jnp.int32),
  grid=(grid_size,),
  in_specs=[pl.BlockSpec(block_shape=(8, dim1, dim2), index_map=lambda i: (i * 8, 0, 0))],
  out_specs=pl.BlockSpec(block_shape=(8, dim2), index_map=lambda i: (i * 8, 0)),
)(x).block_until_ready()
