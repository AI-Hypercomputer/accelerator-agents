# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
dim1 = 256
dim2 = 256
reduce_dim = 1
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, dim1, dim2))


# Computation
def kernel(x_ref, out_ref):
  """Computes the sum of x along the first dimension.

  Args:
    x_ref: A reference to a slice of the input tensor x. The slice has the
      shape (1, dim1, dim2).
    out_ref: A reference to the output tensor. The kernel will write the result
      of the summation to this reference. The slice has the shape (1, 1, dim2).
  """
  out_ref[...] = jnp.sum(x_ref[...], axis=1, keepdims=True)


result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, 1, dim2), x.dtype),
  grid=(batch_size,),
  in_specs=[pl.BlockSpec(block_shape=(1, dim1, dim2), index_map=lambda i: (i, 0, 0))],
  out_specs=pl.BlockSpec(block_shape=(1, 1, dim2), index_map=lambda i: (i, 0, 0)),
)(x).block_until_ready()
