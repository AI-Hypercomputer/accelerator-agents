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
kernel_size = 3
depth = 64
width = 64
height = 64
stride = 1
padding = "VALID"
dilation = 1
groups = 1
bias = False
key = random.PRNGKey(0)
key_params, key_input = random.split(key)

conv3d = nn.Conv(
  features=out_channels,
  kernel_size=(kernel_size, kernel_size, kernel_size),
  strides=stride,
  padding=padding,
  kernel_dilation=dilation,
  feature_group_count=groups,
  use_bias=bias,
)
# JAX uses channels-last convention (N, D, H, W, C)
x = random.normal(key_input, (batch_size, depth, width, height, in_channels))
variables = conv3d.init(key_params, x)


# Computation
def kernel(x_ref, kernel_ref, out_ref):
  """Pallas kernel for 3D convolution."""
  # The kernel now computes a full spatial plane for a given batch and channel.
  # We iterate over the output spatial dimensions (d, h, w) manually.
  for d in range(out_ref.shape[0]):
    for h in range(out_ref.shape[1]):
      for w in range(out_ref.shape[2]):
        # Extract the input patch for the current output coordinate.
        x_patch = jax.lax.dynamic_slice(
          x_ref,
          (d * stride, h * stride, w * stride, 0),
          (kernel_size, kernel_size, kernel_size, in_channels),
        )
        # Compute the dot product and write to the corresponding output location.
        out_ref[d, h, w] = jnp.sum(x_patch * kernel_ref)


# The pallas_call replaces the original convolution computation.
# We define the output shape first.
output_shape = (
  batch_size,
  (depth - kernel_size) // stride + 1,
  (height - kernel_size) // stride + 1,
  (width - kernel_size) // stride + 1,
  out_channels,
)
out_d, out_h, out_w = output_shape[1:4]

# The grid is now 2D, parallelizing over batch and output channels.
# Spatial dimensions are handled inside the kernel.
output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(output_shape, x.dtype),
  grid=(batch_size, out_channels),
  # Input specs define how the full input arrays are chunked for each kernel.
  in_specs=[
    # For the input `x`, each kernel instance gets the full spatial volume
    # for its corresponding batch index `b`. The batch dimension is indexed,
    # not part of the block.
    pl.BlockSpec(
      (None, depth, height, width, in_channels),
      lambda b, c: (b, 0, 0, 0, 0),
    ),
    # For the kernel weights, each instance gets the weights for its
    # target output channel `c`. The out_channel dimension is indexed.
    pl.BlockSpec(
      (kernel_size, kernel_size, kernel_size, in_channels, None),
      lambda b, c: (0, 0, 0, 0, c),
    ),
  ],
  # Output spec defines how the output is written. Each kernel instance writes
  # to a full spatial plane for its batch `b` and channel `c`. These two
  # dimensions are indexed.
  out_specs=pl.BlockSpec((None, out_d, out_h, out_w, None), lambda b, c: (b, 0, 0, 0, c)),
)(x, variables["params"]["kernel"]).block_until_ready()
