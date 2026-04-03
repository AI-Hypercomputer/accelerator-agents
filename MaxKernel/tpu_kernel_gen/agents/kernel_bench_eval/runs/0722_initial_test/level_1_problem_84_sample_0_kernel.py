# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
in_channels = 3
out_channels = 3
kernel_size = 3
width_in = 256
height_in = 128
stride = 1
padding = "VALID"  # padding=0 in PyTorch is equivalent to 'VALID' in JAX/Flax
key = random.PRNGKey(0)
key_x, key_params = random.split(key)
x = random.normal(key_x, (batch_size, height_in, width_in, in_channels))
conv2d = nn.Conv(
  features=out_channels,
  kernel_size=(kernel_size, kernel_size),
  strides=stride,
  padding=padding,
  feature_group_count=in_channels,
  use_bias=False,
)
params = conv2d.init(key_params, x)["params"]
w = params["kernel"]
height_out = (height_in - kernel_size) // stride + 1
width_out = (width_in - kernel_size) // stride + 1
out_shape = (batch_size, height_out, width_out, out_channels)


# Computation
def kernel(x_ref, w_ref, o_ref):
  """Pallas kernel for depthwise convolution."""
  # x_ref is a view into the input of shape (1, kernel_size, width_in, in_channels)
  # w_ref are the weights of shape (kernel_size, kernel_size, 1, out_channels)
  # o_ref is the output slice of shape (1, 1, width_out, out_channels)
  # We iterate over the output width to compute one output pixel at a time.
  for j in range(width_out):
    # Extract a patch from the input corresponding to the current output pixel.
    # We use standard array slicing, which Pallas can handle, instead of
    # lax.dynamic_slice which is not supported on TPU in this context.
    in_patch = x_ref[0, :, j * stride : j * stride + kernel_size, :]
    # The depthwise convolution operation:
    # Sum over the spatial dimensions (0, 1) after element-wise multiplication.
    out_pixel = jnp.sum(in_patch * jnp.squeeze(w_ref[...], axis=2), axis=(0, 1))
    # Write the computed pixel to the output.
    o_ref[0, 0, j, :] = out_pixel


output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(out_shape, x.dtype),
  grid=(batch_size, height_out),
  in_specs=[
    pl.BlockSpec(block_shape=(1, kernel_size, width_in, in_channels), index_map=lambda b, h: (b, h * stride, 0, 0)),
    pl.BlockSpec(block_shape=w.shape, index_map=lambda b, h: (0, 0, 0, 0)),
  ],
  out_specs=pl.BlockSpec(block_shape=(1, 1, width_out, out_channels), index_map=lambda b, h: (b, h, 0, 0)),
)(x, w).block_until_ready()
