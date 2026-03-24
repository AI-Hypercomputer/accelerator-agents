# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
in_features = 10
out_features = 5
subtract_value = 2.0
multiply_value = 1.5

key = random.PRNGKey(0)
x_key, params_key = random.split(key)

x = random.normal(x_key, (batch_size, in_features))
linear = nn.Dense(features=out_features)
params = linear.init(params_key, x)["params"]

block_b = 32


# Computation
def kernel(x_ref, kernel_ref, bias_ref, out_ref):
  # Perform the dense layer computation (matmul + bias)
  y = x_ref[...] @ kernel_ref[...] + bias_ref[...]

  # Apply the element-wise transformations
  y = y - subtract_value
  y = y * multiply_value

  # Apply the ReLU activation
  y = jnp.maximum(y, 0)

  # Write the final result to the output buffer
  out_ref[...] = y


x = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, out_features), x.dtype),
  grid=(batch_size // block_b,),
  in_specs=[
    pl.BlockSpec(block_shape=(block_b, in_features), index_map=lambda i: (i, 0)),
    pl.BlockSpec(block_shape=(in_features, out_features), index_map=lambda i: (0, 0)),
    pl.BlockSpec(block_shape=(out_features,), index_map=lambda i: (0,)),
  ],
  out_specs=pl.BlockSpec(block_shape=(block_b, out_features), index_map=lambda i: (i, 0)),
)(x, params["kernel"], params["bias"]).block_until_ready()
