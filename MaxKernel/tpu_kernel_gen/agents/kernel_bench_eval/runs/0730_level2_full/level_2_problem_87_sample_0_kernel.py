# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from flax import linen as nn
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
subtract_value_1 = 0.5
subtract_value_2 = 0.2

key = random.PRNGKey(0)
params_key, data_key = random.split(key)

conv = nn.Conv(features=out_channels, kernel_size=(kernel_size, kernel_size))
# JAX uses channels-last (NHWC) convention
input_shape = (batch_size, height, width, in_channels)
params = conv.init(params_key, jnp.ones(input_shape))["params"]
x = random.normal(data_key, input_shape)

output_shape = (batch_size, height, width, out_channels)
grid = (batch_size,)

# Perform the convolution using standard JAX.
# The `apply` method handles both the convolution and bias addition.
conv_output = conv.apply({"params": params}, x)


# Computation
def kernel(conv_out_ref, out_ref):
  """
  Pallas kernel for a sequence of element-wise operations:
  subtract, subtract, Mish activation.
  """
  # The bias is already added by the nn.Conv layer.
  # Perform the two subtractions.
  result = conv_out_ref[...] - subtract_value_1 - subtract_value_2

  # Apply the Mish activation function element-wise.
  result = jax.nn.mish(result)

  # Write the final result to the output buffer.
  out_ref[...] = result


# Apply the fused element-wise operations using the Pallas kernel.
fused_output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(output_shape, x.dtype),
  grid=grid,
  in_specs=[pl.BlockSpec(block_shape=(1, height, width, out_channels), index_map=lambda i: (i, 0, 0, 0))],
  out_specs=pl.BlockSpec(block_shape=(1, height, width, out_channels), index_map=lambda i: (i, 0, 0, 0)),
)(conv_output).block_until_ready()
