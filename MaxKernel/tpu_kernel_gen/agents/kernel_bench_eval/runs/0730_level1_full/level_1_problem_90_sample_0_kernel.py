# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax import lax
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
input_shape = (4000,)
dim = 1
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, *input_shape))


# Computation
def kernel(x_ref, out_ref):
  """Pallas kernel for row-wise cumulative product.

  Args:
    x_ref: A reference to a block of the input array.
    out_ref: A reference to a block of the output array for storing the result.
  """
  # The `cumprod` primitive and `lax.scan` are not supported in Pallas TPU.
  # `lax.associative_scan` is a good alternative, but it can fail internally
  # if the scan dimension is not a power of two, leading to an unsupported
  # internal padding operation.
  # We can fix this by manually padding the input to a power of two
  # before the scan, and then slicing the output.
  x_block = x_ref[...]

  orig_len = x_block.shape[1]
  # Pad to the next power of 2, e.g., 4000 -> 4096
  next_pow2_len = 1 << orig_len.bit_length()

  # Pad with 1s, the identity for multiplication.
  x_padded = jnp.pad(
    x_block,
    pad_width=((0, 0), (0, next_pow2_len - orig_len)),
    mode="constant",
    constant_values=1.0,
  )

  # Now `lax.associative_scan` should not need to do internal padding.
  out_padded = lax.associative_scan(jnp.multiply, x_padded, axis=1)

  # Slice the result back to the original shape.
  out_ref[...] = out_padded[:, :orig_len]


result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(batch_size // 8,),
  in_specs=[pl.BlockSpec(block_shape=(8, input_shape[0]), index_map=lambda i: (i * 8, 0))],
  out_specs=pl.BlockSpec(block_shape=(8, input_shape[0]), index_map=lambda i: (i * 8, 0)),
)(x).block_until_ready()
