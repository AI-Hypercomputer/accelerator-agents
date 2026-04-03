# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
divisor = 2
key = random.PRNGKey(0)
key_input, key_params = random.split(key)
x = random.normal(key_input, (batch_size, height, width, in_channels))
conv = nn.Conv(features=out_channels, kernel_size=(kernel_size, kernel_size))
params = conv.init(key_params, x)["params"]

# Computation
# The conv_general_dilated primitive is not supported in Pallas TPU kernels.
# We perform the convolution outside of Pallas and use Pallas for the fusion
# of the subsequent element-wise operations.
conv_out = conv.apply({"params": params}, x)


def fused_kernel(conv_out_ref, bias_ref, divisor_ref, out_ref):
  # Add the bias term.
  biased_out = conv_out_ref[...] + bias_ref[...].reshape(1, 1, 1, -1)

  # Divide the result by the divisor.
  divided_out = biased_out / divisor_ref[0]

  # Apply the leaky ReLU activation function.
  activated_out = nn.leaky_relu(divided_out, negative_slope=0.01)

  # Write the final result to the output reference.
  out_ref[...] = activated_out


x = pl.pallas_call(
  fused_kernel,
  out_shape=jax.ShapeDtypeStruct(conv_out.shape, conv_out.dtype),
  grid=(batch_size,),
  in_specs=[
    pl.BlockSpec(
      block_shape=(1, height, width, out_channels),
      index_map=lambda i: (i, 0, 0, 0),
    ),
    pl.BlockSpec(block_shape=params["bias"].shape, index_map=lambda i: (0,)),
    pl.BlockSpec(block_shape=(1,), index_map=lambda i: (0,)),
  ],
  out_specs=pl.BlockSpec(
    block_shape=(1, height, width, out_channels),
    index_map=lambda i: (i, 0, 0, 0),
  ),
)(conv_out, params["bias"], jnp.array([divisor], dtype=x.dtype)).block_until_ready()
