# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
in_features = 1024
out_features = 512
scaling_factor = 0.5
hardtanh_min = -2
hardtanh_max = 2
b_size = 128

key = random.PRNGKey(0)
key_x, key_gemm = random.split(key)

x = random.normal(key_x, (batch_size, in_features))
gemm = nn.Dense(features=out_features)
params = gemm.init(key_gemm, x)["params"]


# Computation
def kernel(x_ref, kernel_ref, bias_ref, out_ref):
  # Get the current column index from the grid
  j = pl.program_id(axis=1)

  # Perform the matrix multiplication
  out = jnp.dot(x_ref[...], kernel_ref[...])

  # Slice the bias vector statically based on the column index
  bias_slice = jax.lax.slice(bias_ref, (j * b_size,), ((j + 1) * b_size,))

  # Add the bias
  out = out + bias_slice

  # Apply the scaling factor
  out = out * scaling_factor

  # Apply the hardtanh activation
  out = jnp.clip(out, a_min=hardtanh_min, a_max=hardtanh_max)

  # Apply the GELU activation
  out_ref[...] = nn.gelu(out)


x = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, out_features), x.dtype),
  grid=(batch_size // b_size, out_features // b_size),
  in_specs=[
    pl.BlockSpec(block_shape=(b_size, in_features), index_map=lambda i, j: (i, 0)),
    pl.BlockSpec(block_shape=(in_features, b_size), index_map=lambda i, j: (0, j)),
    pl.BlockSpec(block_shape=(out_features,), index_map=lambda i, j: (0,)),
  ],
  out_specs=pl.BlockSpec(block_shape=(b_size, b_size), index_map=lambda i, j: (i, j)),
)(x, params["kernel"], params["bias"]).block_until_ready()
