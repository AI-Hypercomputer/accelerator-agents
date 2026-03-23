# Imports
import flax.linen as nn
import jax
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
in_features = 10
out_features = 20

key = random.PRNGKey(0)
key, x_key, params_key = random.split(key, 3)

x = random.normal(x_key, (batch_size, in_features))
linear = nn.Dense(features=out_features)
params = linear.init(params_key, x)["params"]


def kernel(x_ref, kernel_ref, bias_ref, out_ref):
  # Apply the dense layer: y = x @ w + b
  y = x_ref[...] @ kernel_ref[...] + bias_ref[...]
  # Apply the first mish activation
  y = jax.nn.mish(y)
  # Apply the second mish activation
  y = jax.nn.mish(y)
  # Write the result to the output
  out_ref[...] = y


# Computation
x = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, out_features), x.dtype),
  grid=(1,),
  in_specs=[
    pl.BlockSpec(block_shape=(batch_size, in_features), index_map=lambda i: (0, 0)),
    pl.BlockSpec(block_shape=(in_features, out_features), index_map=lambda i: (0, 0)),
    pl.BlockSpec(block_shape=(out_features,), index_map=lambda i: (0,)),
  ],
  out_specs=pl.BlockSpec(block_shape=(batch_size, out_features), index_map=lambda i: (0, 0)),
)(x, params["kernel"], params["bias"]).block_until_ready()
