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
key = random.PRNGKey(0)
key_A, key_B = random.split(key)
A = random.normal(key_A, (b, i, j, l))
B = random.normal(key_B, (l, k))

k_block = 128


def kernel(A_ref, B_ref, C_ref):
  """Computes a tile of the batched matrix multiplication."""
  # A_ref is a slice of A with shape (1, 1, j, l)
  # B_ref is a tile of B with shape (l, k_block)
  # C_ref is the output tile with shape (1, 1, j, k_block)
  C_ref[...] = jnp.matmul(A_ref, B_ref)


# Computation
C = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((b, i, j, k), A.dtype),
  grid=(b, i, k // k_block),
  in_specs=[
    pl.BlockSpec(block_shape=(1, 1, j, l), index_map=lambda b_idx, i_idx, k_idx: (b_idx, i_idx, 0, 0)),
    pl.BlockSpec(block_shape=(l, k_block), index_map=lambda b_idx, i_idx, k_idx: (0, k_idx * k_block)),
  ],
  out_specs=pl.BlockSpec(
    block_shape=(1, 1, j, k_block), index_map=lambda b_idx, i_idx, k_idx: (b_idx, i_idx, 0, k_idx * k_block)
  ),
)(A, B).block_until_ready()
