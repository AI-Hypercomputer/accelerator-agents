# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from flax import linen as nn
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
input_size = 1024
hidden_size = 512
scaling_factor = 2.0
key = random.PRNGKey(0)
key_x, key_params = random.split(key)
x = random.normal(key_x, (batch_size, input_size))
gemm = nn.Dense(features=hidden_size)
params = gemm.init(key_params, x)["params"]
bM = 128
bN = 128


# Computation
def kernel(x_ref, kernel_ref, bias_ref, out_ref, scaling_factor):
  j = pl.program_id(axis=1)

  # Load the slice of the bias for the current block.
  bias_slice = pl.load(bias_ref, (pl.dslice(j * bN, bN),))

  # Perform the matrix multiplication and add the bias.
  gemm_output = jnp.dot(x_ref[...], kernel_ref[...]) + bias_slice

  # Apply the sigmoid activation function.
  sigmoid_output = jax.nn.sigmoid(gemm_output)

  # Scale the result of the sigmoid activation.
  scaled_output = sigmoid_output * scaling_factor

  # Add the original gemm_output (residual connection) and write to the output.
  out_ref[...] = scaled_output + gemm_output


result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, hidden_size), x.dtype),
  grid=(batch_size // bM, hidden_size // bN),
  in_specs=[
    pl.BlockSpec(block_shape=(bM, x.shape[1]), index_map=lambda i, j: (i * bM, 0)),
    pl.BlockSpec(block_shape=(x.shape[1], bN), index_map=lambda i, j: (0, j * bN)),
    pl.BlockSpec(block_shape=(hidden_size,), index_map=lambda i, j: (0,)),
  ],
  out_specs=pl.BlockSpec(block_shape=(bM, bN), index_map=lambda i, j: (i * bM, j * bN)),
)(x, params["kernel"], params["bias"], scaling_factor).block_until_ready()
