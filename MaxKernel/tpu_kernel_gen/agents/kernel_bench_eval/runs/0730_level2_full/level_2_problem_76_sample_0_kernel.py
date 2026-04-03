# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl
from jax.nn import relu

# Initialization
batch_size = 128
in_features = 1024
out_features = 512
bias_shape = (out_features,)

key = random.PRNGKey(0)
key_x, key_weight, key_bias = random.split(key, 3)

x = random.normal(key_x, (batch_size, in_features))
weight = random.normal(key_weight, (out_features, in_features))
bias = random.normal(key_bias, bias_shape)

# Block sizes for the kernel
bM = 128  # Block size for the batch dimension
bN = 128  # Block size for the output features dimension
bK = in_features  # Block size for the contracting dimension


# Computation
def kernel(x_ref, weight_ref, bias_ref, out_ref):
  """Pallas kernel for a fused GEMM, bias add, and ReLU activation.

  Args:
    x_ref: Input data block.
    weight_ref: Weight matrix block.
    bias_ref: Bias vector block.
    out_ref: Output block to write the result to.
  """
  j = pl.program_id(axis=1)
  # Perform the matrix multiplication: x @ weight.T
  gemm_output = jnp.matmul(x_ref[...], weight_ref[...].T)

  # Slice the full bias vector to get the relevant chunk for this tile
  bias_slice = jax.lax.slice(bias_ref[...], [j * bN], [j * bN + bN])

  # Add the bias vector (broadcasts automatically)
  bias_add_output = gemm_output + bias_slice

  # Apply the ReLU activation function and write to the output
  out_ref[...] = relu(bias_add_output)


output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, out_features), x.dtype),
  grid=(batch_size // bM, out_features // bN),
  in_specs=[
    pl.BlockSpec(block_shape=(bM, bK), index_map=lambda i, j: (i, 0)),
    pl.BlockSpec(block_shape=(bN, bK), index_map=lambda i, j: (j, 0)),
    pl.BlockSpec(block_shape=(out_features,), index_map=lambda i, j: (0,)),
  ],
  out_specs=pl.BlockSpec(block_shape=(bM, bN), index_map=lambda i, j: (i, j)),
)(x, weight, bias).block_until_ready()
