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
D, H, W = 16, 32, 32
kernel_size = 3
dim = 1  # Corresponds to the 'D' dimension in JAX's N,D,H,W,C format
key = random.PRNGKey(0)
key_x, key_params = random.split(key)

x = random.normal(key_x, (batch_size, D, H, W, in_channels))
conv = nn.Conv(features=out_channels, kernel_size=(kernel_size, kernel_size, kernel_size))
params = conv.init(key_params, x)["params"]

# Computation
# The convolution is performed in JAX since conv primitives are not supported in Pallas.
conv_out = conv.apply({"params": params}, x)


def reduction_softmax_kernel(x_conv_ref, out_ref):
  """Pallas kernel for a sequence of reduction and softmax.

  Args:
    x_conv_ref: Input tensor after convolution with shape (1, D, H, W, out_channels).
    out_ref: Output tensor with shape (1, H, W, out_channels).
  """
  # Apply the reduction operation (min) along the depth dimension (axis=1).
  # The output shape becomes (1, H, W, out_channels).
  reduced_out = jnp.min(x_conv_ref[...], axis=1)

  # Apply the softmax activation function along the last dimension (channels).
  softmax_out = nn.softmax(reduced_out, axis=-1)

  # Write the final result to the output buffer.
  out_ref[...] = softmax_out


x = pl.pallas_call(
  reduction_softmax_kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, H, W, out_channels), x.dtype),
  grid=(batch_size,),
  in_specs=[
    pl.BlockSpec(block_shape=(1, D, H, W, out_channels), index_map=lambda i: (i, 0, 0, 0, 0)),
  ],
  out_specs=pl.BlockSpec(block_shape=(1, H, W, out_channels), index_map=lambda i: (i, 0, 0, 0)),
)(conv_out).block_until_ready()
