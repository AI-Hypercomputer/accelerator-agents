# Imports
import jax
import jax.nn as nn
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
input_size = 512
output_size = 1024
divisor = 10.0
block_size = 128
key = random.PRNGKey(0)
key, x_key, weight_key, bias_key = random.split(key, 4)
x = random.normal(x_key, (batch_size, input_size))
weight = random.normal(weight_key, (output_size, input_size))
bias = random.normal(bias_key, (output_size,))


# Computation
def kernel(x_ref, weight_ref, bias_ref, out_ref, divisor):
  # Load the data from HBM to SRAM.
  x = x_ref[...]
  weight = weight_ref[...]
  bias = bias_ref[...]

  # Perform the matrix multiplication between the input block and the transposed weight block.
  # x: [block_size, input_size]
  # weight: [block_size, input_size] -> weight.T: [input_size, block_size]
  # y: [block_size, block_size]
  y = jnp.dot(x, weight.T)

  # Add the bias vector. It will be broadcasted to the shape of y.
  # bias: [block_size,]
  y = y + bias

  # Divide by the divisor.
  y = y / divisor

  # Apply the GELU activation function.
  y = nn.gelu(y)

  # Write the final result to the output block.
  out_ref[...] = y


# The kernel for this computation would fuse the matmul, division, and GELU activation.
# The grid is set up to parallelize the computation over the output matrix.
# Each kernel instance computes a 128x128 block of the final output.
result = pl.pallas_call(
  lambda x_ref, weight_ref, bias_ref, o_ref: kernel(x_ref, weight_ref, bias_ref, o_ref, divisor),
  # The output shape is (batch_size, output_size)
  out_shape=jax.ShapeDtypeStruct((batch_size, output_size), x.dtype),
  # Grid is (batch_size / 128, output_size / 128) -> (1, 8)
  grid=(batch_size // block_size, output_size // block_size),
  in_specs=[
    # x_ref: (128, 512). For each output block, we need the full x.
    # The index_map depends only on the first grid index `i`.
    pl.BlockSpec(block_shape=(block_size, input_size), index_map=lambda i, j: (i * block_size, 0)),
    # weight_ref: (128, 512). For each output block (i, j), we need the j-th
    # block of rows from the weight matrix.
    pl.BlockSpec(block_shape=(block_size, input_size), index_map=lambda i, j: (j * block_size, 0)),
    # bias_ref: (128,). For each output block (i, j), we need the j-th
    # slice of the bias vector.
    pl.BlockSpec(block_shape=(block_size,), index_map=lambda i, j: (j * block_size,)),
  ],
  # out_specs maps each grid instance (i, j) to a unique (128, 128)
  # block in the output matrix.
  out_specs=pl.BlockSpec(block_shape=(block_size, block_size), index_map=lambda i, j: (i * block_size, j * block_size)),
)(x, weight, bias).block_until_ready()
