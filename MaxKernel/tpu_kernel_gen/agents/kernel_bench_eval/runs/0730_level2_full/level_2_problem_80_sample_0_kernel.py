# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
in_features = 512
out_features = 1024
max_dim = 1
b_size = 8

key = random.PRNGKey(0)
key_x, key_weight, key_bias = random.split(key, 3)

x = random.normal(key_x, (batch_size, in_features))
weight = random.normal(key_weight, (out_features, in_features))
bias = random.normal(key_bias, (out_features,))


# Computation
def kernel(x_ref, weight_ref, bias_ref, out_ref):
  # Perform the linear transformation (dot product + bias)
  # x_ref: [b_size, in_features]
  # weight_ref: [out_features, in_features]
  # bias_ref: [out_features]
  # y: [b_size, out_features]
  y = jnp.dot(x_ref[...], weight_ref[...].T) + bias_ref[...]

  # Apply the max reduction along the feature dimension
  # y: [b_size, out_features] -> max_val: [b_size, 1]
  max_val = jnp.max(y, axis=1, keepdims=True)

  # Center the values by subtracting the mean.
  # Since max_val is of shape [b_size, 1], its mean along axis 1 is itself.
  # This operation effectively results in a tensor of zeros.
  # centered_val: [b_size, 1]
  centered_val = max_val - max_val.mean(axis=1, keepdims=True)

  # Apply the GELU activation function
  # activated_val: [b_size, 1]
  activated_val = nn.gelu(centered_val)

  # Write the final result to the output reference
  out_ref[...] = activated_val


x = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, 1), x.dtype),
  grid=(batch_size // b_size,),
  in_specs=[
    pl.BlockSpec((b_size, in_features), lambda i: (i, 0)),
    pl.BlockSpec((out_features, in_features), lambda i: (0, 0)),
    pl.BlockSpec((out_features,), lambda i: (0,)),
  ],
  out_specs=pl.BlockSpec((b_size, 1), lambda i: (i, 0)),
)(x, weight, bias).block_until_ready()
