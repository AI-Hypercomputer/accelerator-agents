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

b_bs = 8
b_d2 = 128


def kernel(x_ref, out_ref):
  out_ref[...] = jnp.min(x_ref[...], axis=1)


# Computation
result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, dim2), x.dtype),
  grid=(batch_size // b_bs, dim2 // b_d2),
  in_specs=[
    pl.BlockSpec((b_bs, dim1, b_d2), lambda i, j: (i, 0, j)),
  ],
  out_specs=pl.BlockSpec((b_bs, b_d2), lambda i, j: (i, j)),
)(x).block_until_ready()
