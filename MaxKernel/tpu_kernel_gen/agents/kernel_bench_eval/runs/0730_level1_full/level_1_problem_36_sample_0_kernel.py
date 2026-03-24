# Imports
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
eps = 1e-5
batch_size = 16
features = 64
dim1 = 256
dim2 = 256
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, features, dim1, dim2))


# Computation
def kernel(x_ref, out_ref, *, eps):
  """
  Pallas kernel for RMS normalization.

  This kernel computes the root mean square normalization for a slice of the input
  tensor. The normalization is performed across the feature dimension (axis=1).

  Args:
    x_ref: A reference to the input tensor slice. The slice corresponds to a
      single item in the batch and has a shape of
      (1, features, dim1, dim2).
    out_ref: A reference to the output tensor slice where the result is stored
      in-place.
    eps: A small floating-point scalar to avoid division by zero.
  """
  # Calculate the root mean square of the input slice across the feature axis.
  # The mean is taken over axis=1, and keepdims=True ensures that the rank of
  # the resulting tensor is maintained for broadcasting during the division.
  rms = jnp.sqrt(jnp.mean(x_ref[...] ** 2, axis=1, keepdims=True) + eps)

  # Normalize the input slice by dividing it by the RMS value and write the
  # result to the output reference.
  out_ref[...] = x_ref[...] / rms


output = pl.pallas_call(
  partial(kernel, eps=eps),
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(batch_size,),
  in_specs=[pl.BlockSpec(block_shape=(1, x.shape[1], x.shape[2], x.shape[3]), index_map=lambda i: (i, 0, 0, 0))],
  out_specs=pl.BlockSpec(block_shape=(1, x.shape[1], x.shape[2], x.shape[3]), index_map=lambda i: (i, 0, 0, 0)),
)(x).block_until_ready()
