# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = (3, 5, 7)
depth = 16
height = 256
width = 256
stride = (1, 1, 1)
padding = "VALID"
dilation = (1, 1, 1)
groups = 1
bias = False
key = random.PRNGKey(0)
key_params, key_x = random.split(key)
conv3d = nn.Conv(
  features=out_channels,
  kernel_size=kernel_size,
  strides=stride,
  padding=padding,
  kernel_dilation=dilation,
  feature_group_count=groups,
  use_bias=bias,
)
x = random.normal(key_x, (batch_size, depth, height, width, in_channels))
params = conv3d.init(key_params, x)["params"]


def kernel(x_ref, kernel_ref, out_ref):
  """Pallas kernel for 3D convolution.

  This kernel computes one output pixel (a vector of size out_channels) per
  program instance. A program instance is identified by the full 4D index
  (batch, output_depth, output_height, output_width).

  Args:
    x_ref: A reference to the input tensor slice (patch). The slice is the
      exact patch from the input `x` needed to compute one output pixel.
      Shape: (1, kernel_depth, kernel_height, kernel_width, input_channels).
    kernel_ref: A reference to the complete kernel weights tensor.
      Shape: (kernel_depth, kernel_height, kernel_width, input_channels, output_channels).
    out_ref: A reference to the output tensor slice. The kernel will write
      the single computed output pixel here.
      Shape: (1, 1, 1, 1, output_channels).
  """
  # Extract shape information from the kernel weights.
  kd, kh, kw, cin, cout = kernel_ref.shape

  # Load the input patch and kernel weights from SRAM into registers.
  # x_ref is a view of the larger input tensor, corresponding to the patch
  # needed for a single convolution operation.
  x_patch = x_ref[...]
  kernel_w = kernel_ref[...]

  # Reshape the input patch for the dot product.
  # The dimensions that are contracted in the convolution are flattened.
  # New shape: (1, kd * kh * kw * cin)
  x_reshaped = x_patch.reshape(1, -1)

  # Reshape the kernel for the dot product.
  # New shape: (kd * kh * kw * cin, cout)
  kernel_reshaped = kernel_w.reshape(-1, cout)

  # Perform the dot product to compute the output pixel.
  # (1, kd*kh*kw*cin) @ (kd*kh*kw*cin, cout) -> (1, cout)
  out_pixel = jnp.dot(x_reshaped, kernel_reshaped)

  # Write the result to the output buffer, reshaping to match the expected
  # 5D output shape of (1, 1, 1, 1, cout).
  out_ref[...] = out_pixel.reshape(out_ref.shape)


# Computation
# Calculate output dimensions based on 'VALID' padding and stride of 1
out_depth = depth - kernel_size[0] + 1
out_height = height - kernel_size[1] + 1
out_width = width - kernel_size[2] + 1
output_shape = (batch_size, out_depth, out_height, out_width, out_channels)

# The grid now iterates over every single output pixel
grid = (batch_size, out_depth, out_height, out_width)

output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(output_shape, x.dtype),
  grid=grid,
  in_specs=[
    # Spec for input 'x': read a patch of size (kd, kh, kw) for each pixel
    pl.BlockSpec(
      block_shape=(1, *kernel_size, in_channels),
      index_map=lambda b, d, h, w: (b, d, h, w, 0),
    ),
    # Spec for kernel weights: read the whole kernel every time
    pl.BlockSpec(
      block_shape=params["kernel"].shape,
      index_map=lambda *_: tuple([0] * params["kernel"].ndim),
    ),
  ],
  out_specs=pl.BlockSpec(
    # Spec for output: write one pixel at a time
    block_shape=(1, 1, 1, 1, out_channels),
    index_map=lambda b, d, h, w: (b, d, h, w, 0),
  ),
  compiler_params=dict(mosaic=dict(dimension_semantics=("parallel", "parallel", "parallel", "parallel"))),
)(x, params["kernel"]).block_until_ready()
