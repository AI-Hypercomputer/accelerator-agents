# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
b = 16
i = 256
j = 512
l = 256
k = 768
bJ = 128
key = random.PRNGKey(0)
key_A, key_B = random.split(key)
A = random.normal(key_A, (b, i, j, l))
B = random.normal(key_B, (l, k))


# Computation
def kernel(x_ref, y_ref, out_ref):
  """Pallas kernel for the einsum operation.

  This kernel computes a block of the equivalent operation:
  C[b, i, j, k] = A[b, i, j, l] @ B[l, k]

  Args:
    x_ref: A reference to a block of the input tensor A.
      Shape: (1, 1, bJ, l)
    y_ref: A reference to the input tensor B.
      Shape: (l, k)
    out_ref: A reference to a block of the output tensor C, which will be
      written to.
      Shape: (1, 1, bJ, k)
  """
  # The einsum "bijl,lk->bijk" is equivalent to a batched matrix multiplication.
  # The pallas_call sets up a grid over the b, i, and blocked j dimensions.
  # Inside the kernel, we perform the matrix multiplication for a given block.
  # First, load the data from the references into local variables.
  x = x_ref[...]
  y = y_ref[...]
  # x has shape (1, 1, bJ, l) and y has shape (l, k).
  # jnp.matmul on TPU expects 2D inputs, so we squeeze the singleton dimensions
  # from x.
  x = jnp.squeeze(x, axis=(0, 1))
  # The matmul result has shape (bJ, k). We must reshape it to the
  # (1, 1, bJ, k) shape of out_ref before assignment.
  out_ref[...] = jnp.reshape(jnp.matmul(x, y), (1, 1, bJ, k))


C = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((b, i, j, k), jnp.float32),
  grid=(b, i, j // bJ),
  in_specs=[
    pl.BlockSpec(block_shape=(1, 1, bJ, l), index_map=lambda gb, gi, gj: (gb, gi, gj * bJ, 0)),
    pl.BlockSpec(block_shape=(l, k), index_map=lambda gb, gi, gj: ()),
  ],
  out_specs=pl.BlockSpec(block_shape=(1, 1, bJ, k), index_map=lambda gb, gi, gj: (gb, gi, gj * bJ, 0)),
)(A, B).block_until_ready()
