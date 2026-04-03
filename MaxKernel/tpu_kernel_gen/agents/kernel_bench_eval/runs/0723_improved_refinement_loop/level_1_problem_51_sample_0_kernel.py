# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
dim1 = 256
dim2 = 256
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, dim1, dim2))
b_dim1_out = 8
b_dim2_out = 128


# Computation
def kernel(x_ref, out_ref):
  """Pallas kernel to compute argmax along axis=1.

  Args:
    x_ref: A reference to a block of the input array. The shape is expected to
      be (b_dim1_out, dim1, b_dim2_out).
    out_ref: A reference to a block of the output array for writing the
      results. The shape is expected to be (b_dim1_out, b_dim2_out).
  """
  # The input block x_ref has a shape of (b_dim1_out, dim1, b_dim2_out).
  # We compute argmax over axis=1, which results in a shape of (b_dim1_out, b_dim2_out).
  # jnp.argmax is a supported JAX primitive within Pallas kernels.
  argmax_result = jnp.argmax(x_ref[...], axis=1)

  # The output reference out_ref has a shape of (b_dim1_out, b_dim2_out).
  # The result of argmax already matches the output shape.
  out_ref[...] = argmax_result


result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, dim2), jnp.int32),
  grid=(batch_size // b_dim1_out, dim2 // b_dim2_out),
  in_specs=[pl.BlockSpec((b_dim1_out, dim1, b_dim2_out), lambda i, j: (i * b_dim1_out, 0, j * b_dim2_out))],
  out_specs=[pl.BlockSpec((b_dim1_out, b_dim2_out), lambda i, j: (i * b_dim1_out, j * b_dim2_out))],
)(x).block_until_ready()
