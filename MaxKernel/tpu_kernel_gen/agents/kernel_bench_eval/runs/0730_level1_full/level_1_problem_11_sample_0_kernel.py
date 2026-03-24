# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
b = 2
i = 4
j = 8
l = 4
k = 16
key = random.PRNGKey(0)
key_A, key_B = random.split(key)
A = random.normal(key_A, (b, i, j, l))
B = random.normal(key_B, (l, k))


# Computation
def kernel(x_ref, y_ref, out_ref):
  """
  Computes jnp.einsum("bijl,lk->bijk", A, B) for a block.

  Args:
    x_ref: A block of A with shape (1, 1, j, l).
    y_ref: The entire B tensor with shape (l, k).
    out_ref: An output block for C with shape (1, 1, j, k).
  """
  # The einsum "bijl,lk->bijk" is equivalent to a matrix multiplication
  # for each (b, i) index. The grid of the pallas_call iterates over
  # these b and i dimensions.
  # Inside the kernel, we perform the matmul for a given (b, i) slice.
  # x_ref[...] has shape (1, 1, j, l)
  # y_ref[...] has shape (l, k)
  # The matmul broadcasts y_ref and computes the (j, l) by (l, k) matmul,
  # resulting in a (1, 1, j, k) shape, which matches out_ref.
  out_ref[...] = jnp.matmul(x_ref[...], y_ref[...])


C = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((b, i, j, k), A.dtype),
  grid=(b, i),
  in_specs=[
    pl.BlockSpec(block_shape=(1, 1, j, l), index_map=lambda b_idx, i_idx: (b_idx, i_idx, 0, 0)),
    pl.BlockSpec(block_shape=(l, k), index_map=lambda b_idx, i_idx: (0, 0)),
  ],
  out_specs=pl.BlockSpec(block_shape=(1, 1, j, k), index_map=lambda b_idx, i_idx: (b_idx, i_idx, 0, 0)),
)(A, B).block_until_ready()
