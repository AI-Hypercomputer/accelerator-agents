# Imports
import jax
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
b = 16
i = 256
j = 512
l = 256
k = 768
key = random.PRNGKey(0)
key_A, key_B = random.split(key)
A = random.normal(key_A, (b, i, j, l))
B = random.normal(key_B, (l, k))


# Computation
def kernel(a_ref, b_ref, c_ref):
  """Computes jnp.einsum("bijl,lk->bijk", A, B) with pallas.

  Args:
    a_ref: A reference to a slice of A of shape [1, 1, j, l].
    b_ref: A reference to the entire B tensor of shape [l, k].
    c_ref: A reference to a slice of the output C of shape [1, 1, j, k],
      to be written to.
  """
  # For each program in the grid (b, i), we are given a slice of A,
  # A[b, i, :, :], which has an effective shape of (j, l).
  # We are also given the entire B tensor, which has shape (l, k).
  # The einsum "jl,lk->jk" is equivalent to a matrix multiplication.
  # The inputs refs have leading singleton dimensions due to the BlockSpec,
  # so we slice them to 2D before the matmul.
  c_ref[0, 0] = a_ref[0, 0] @ b_ref[...]


C = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(shape=(A.shape[0], A.shape[1], A.shape[2], B.shape[1]), dtype=A.dtype),
  grid=(A.shape[0], A.shape[1]),
  in_specs=[
    pl.BlockSpec(block_shape=(1, 1, A.shape[2], A.shape[3]), index_map=lambda b, i: (b, i, 0, 0)),
    pl.BlockSpec(block_shape=(B.shape[0], B.shape[1]), index_map=lambda b, i: (0, 0)),
  ],
  out_specs=pl.BlockSpec(block_shape=(1, 1, A.shape[2], B.shape[1]), index_map=lambda b, i: (b, i, 0, 0)),
)(A, B).block_until_ready()
