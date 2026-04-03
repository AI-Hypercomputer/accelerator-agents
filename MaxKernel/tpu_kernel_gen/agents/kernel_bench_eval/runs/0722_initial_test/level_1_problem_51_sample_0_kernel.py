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
batch_block_size = 8


# Computation
def kernel(x_ref, out_ref):
  """Pallas kernel to compute argmax along axis=1.

  Args:
    x_ref: A reference to a block of the input tensor. The shape is
      (batch_block_size, 256, 256).
    out_ref: A reference to a block of the output tensor. The shape is
      (batch_block_size, 256).
  """
  # The original computation is jnp.argmax(x, axis=1).
  # The pallas_call invocation provides a block of x as x_ref.
  # We perform the argmax operation on this block along the same axis.
  # The result of jnp.argmax(x_ref[...], axis=1) will have the shape
  # (batch_block_size, 256), which matches the shape of out_ref.
  out_ref[...] = jnp.argmax(x_ref[...], axis=1)


result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((16, 256), jnp.int32),
  grid=(16 // batch_block_size,),
  in_specs=[
    pl.BlockSpec(
      block_shape=(batch_block_size, 256, 256),
      index_map=lambda i: (i * batch_block_size, 0, 0),
    )
  ],
  out_specs=pl.BlockSpec(
    block_shape=(batch_block_size, 256),
    index_map=lambda i: (i * batch_block_size, 0),
  ),
)(x).block_until_ready()
