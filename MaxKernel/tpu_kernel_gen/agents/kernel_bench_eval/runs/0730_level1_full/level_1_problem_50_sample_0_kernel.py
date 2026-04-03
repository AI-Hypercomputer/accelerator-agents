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
b_b = 8
b_d2 = 128


# Computation
def kernel(x_ref, out_ref):
  """
  Pallas kernel to compute the product of elements along axis 1.

  Args:
    x_ref: A reference to the input tensor `x`. The BlockSpec for this ref
      is configured to give a view into HBM, but not necessarily to pre-load
      the whole thing.
    out_ref: A reference to a block of the output tensor. The shape of this
      block is (b_b, b_d2).
  """
  # Initialize an accumulator in registers.
  acc = jnp.ones_like(out_ref)
  # Iterate over the reduction dimension.
  for k in range(dim1):
    # x_ref has a shape of (b_b, 1, b_d2) and points to the k=0 slice of the
    # data needed for this kernel. We load subsequent slices (k=1, 2, ...)
    # by providing an offset to `pl.load`. The offset is applied to the
    # base pointer of `x_ref` in HBM.
    # The loaded slice will have the same shape as x_ref: (b_b, 1, b_d2).
    current_slice = pl.load(x_ref, (0, k, 0), eviction_policy="evict_last")
    # We squeeze out the singleton dimension before the multiplication.
    acc *= current_slice.squeeze(axis=1)
  # Store the final result from registers to the output block in SRAM.
  out_ref[...] = acc


result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, dim2), x.dtype),
  grid=(batch_size // b_b, dim2 // b_d2),
  in_specs=[
    # The block_shape for x_ref is now (b_b, 1, b_d2). This means the ref
    # passed to the kernel has this shape. The index_map points to the
    # start of the data for the whole reduction (k=0).
    pl.BlockSpec(block_shape=(b_b, 1, b_d2), index_map=lambda i, j: (i * b_b, 0, j * b_d2)),
  ],
  out_specs=pl.BlockSpec(block_shape=(b_b, b_d2), index_map=lambda i, j: (i * b_b, j * b_d2)),
)(x).block_until_ready()
