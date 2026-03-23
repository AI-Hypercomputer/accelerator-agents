# Imports
import flax.linen as nn
import jax
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
in_channels = 3
kernel_size = 3
width = 256
height = 256
stride = 1
padding = 0
dilation = 1

key = random.PRNGKey(0)
key_x, key_params = random.split(key)

# JAX uses channels-last convention by default
x = random.normal(key_x, (batch_size, height, width, in_channels))

conv2d = nn.Conv(
  features=in_channels,
  kernel_size=(kernel_size, 1),
  strides=stride,
  padding=padding,
  kernel_dilation=dilation,
  feature_group_count=in_channels,
  use_bias=False,
)
params = conv2d.init(key_params, x)["params"]

# The output height for a 'VALID' convolution.
out_height = height - kernel_size + 1


# Computation
def kernel(x_ref, kernel_ref, out_ref):
  """Pallas kernel for a depthwise convolution with a (kernel_size, 1) kernel.

  This kernel computes one output row at a time. It assumes that x_ref contains
  the necessary receptive field from the input tensor to compute one full
  output row.

  Args:
    x_ref: A reference to the input tensor slice. The kernel assumes this slice
      has a shape corresponding to (1, kernel_size, width, in_channels) and
      contains the receptive field needed for the convolution.
    kernel_ref: A reference to the convolution kernel. Expected shape is
      (kernel_size, 1, 1, in_channels).
    out_ref: A reference to the output tensor slice, where the result for the
      current row will be written. Expected shape is (1, 1, width, in_channels).
  """
  # Squeeze the singleton dimensions to simplify the computation.
  # x_ref shape: (1, kernel_size, width, in_channels) -> (kernel_size, width, in_channels)
  x_block = x_ref[0]
  # kernel_ref shape: (kernel_size, 1, 1, in_channels) -> (kernel_size, in_channels)
  k_block = kernel_ref[:, 0, 0, :]

  # Reshape kernel for broadcasting across the width dimension.
  k_block_reshaped = k_block.reshape(kernel_size, 1, in_channels)

  # Perform element-wise multiplication and sum over the kernel dimension.
  # This is equivalent to the depthwise convolution.
  out_block = (x_block * k_block_reshaped).sum(axis=0, dtype=x_ref.dtype)

  # Reshape the result to match the 4D output shape and write it out.
  out_ref[...] = out_block.reshape(1, 1, width, in_channels)


result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, out_height, width, in_channels), x.dtype),
  grid=(batch_size, out_height),
  in_specs=[
    pl.BlockSpec(block_shape=(1, kernel_size, width, in_channels), index_map=lambda b, h: (b, h, 0, 0)),
    pl.BlockSpec(block_shape=(kernel_size, 1, 1, in_channels), index_map=lambda b, h: (0, 0, 0, 0)),
  ],
  out_specs=pl.BlockSpec(block_shape=(1, 1, width, in_channels), index_map=lambda b, h: (b, h, 0, 0)),
)(x, params["kernel"]).block_until_ready()
