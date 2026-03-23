# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
input_size = 512
hidden_size = 256
num_groups = 8
eps = 1e-5
negative_slope = 0.01

key = random.PRNGKey(0)
key, x_key, fc_key, gn_key = random.split(key, 4)

x = random.normal(x_key, (batch_size, input_size))

fc = nn.Dense(features=hidden_size)
fc_params = fc.init(fc_key, x)["params"]

gn = nn.GroupNorm(num_groups=num_groups, epsilon=eps)
dummy_gn_input = jnp.ones((batch_size, hidden_size))
gn_params = gn.init(gn_key, dummy_gn_input)["params"]


# Computation
def kernel(x_ref, fc_kernel_ref, fc_bias_ref, gn_scale_ref, gn_bias_ref, out_ref):
  # Constants from the original computation
  num_groups = 8
  eps = 1e-5
  negative_slope = 0.01
  hidden_size = 256
  input_size = 512

  # 1. Fully connected layer
  # x_ref: (input_size,)
  # fc_kernel_ref: (input_size, hidden_size)
  # fc_bias_ref: (hidden_size,)
  x = x_ref[...].reshape(1, input_size) @ fc_kernel_ref[...] + fc_bias_ref[...]

  # 2. Group Normalization
  # Reshape to compute statistics over groups
  x_reshaped = x.reshape(num_groups, hidden_size // num_groups)

  # Compute mean and variance per group
  mean = jnp.mean(x_reshaped, axis=1, keepdims=True)
  var = jnp.var(x_reshaped, axis=1, keepdims=True)

  # Normalize within groups
  x_normalized_reshaped = (x_reshaped - mean) / jnp.sqrt(var + eps)

  # Reshape back to original feature dimension
  x_normalized = x_normalized_reshaped.reshape(1, hidden_size)

  # Apply scale and bias
  # gn_scale_ref: (hidden_size,)
  # gn_bias_ref: (hidden_size,)
  x = x_normalized * gn_scale_ref[...] + gn_bias_ref[...]

  # 3. Leaky ReLU activation
  x = jnp.where(x > 0, x, x * negative_slope)

  # 4. Element-wise addition
  x = x + x

  # Write the final result to the output reference
  out_ref[...] = x.reshape(
    hidden_size,
  )


x_flat = x.reshape(-1)
result_flat = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size * hidden_size,), x.dtype),
  grid=(batch_size,),
  in_specs=[
    pl.BlockSpec(block_shape=(input_size,), index_map=lambda i: (i * input_size,)),
    pl.BlockSpec(block_shape=(input_size, hidden_size), index_map=lambda i: (0, 0)),
    pl.BlockSpec(block_shape=(hidden_size,), index_map=lambda i: (0,)),
    pl.BlockSpec(block_shape=(hidden_size,), index_map=lambda i: (0,)),
    pl.BlockSpec(block_shape=(hidden_size,), index_map=lambda i: (0,)),
  ],
  out_specs=pl.BlockSpec(block_shape=(hidden_size,), index_map=lambda i: (i * hidden_size,)),
)(
  x_flat,
  fc_params["kernel"],
  fc_params["bias"],
  gn_params["scale"],
  gn_params["bias"],
)
x = result_flat.reshape(batch_size, hidden_size).block_until_ready()
