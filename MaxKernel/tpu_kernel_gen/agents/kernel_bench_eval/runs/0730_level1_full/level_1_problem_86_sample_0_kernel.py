# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from flax import linen as nn
from jax.experimental import pallas as pl
from jax.experimental.pallas.ops import tpu

# Initialization
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 3
width = 256
height = 256
stride = 1
padding = "VALID"  # PyTorch padding=0 is 'VALID' in JAX
dilation = 1
bias = False

key = random.PRNGKey(0)
key, x_key, depthwise_key, pointwise_key = random.split(key, 4)

# JAX uses channels-last convention
x = random.normal(x_key, (batch_size, height, width, in_channels))

depthwise = nn.Conv(
  features=in_channels,
  kernel_size=(kernel_size, kernel_size),
  strides=(stride, stride),
  padding=padding,
  kernel_dilation=(dilation, dilation),
  feature_group_count=in_channels,
  use_bias=bias,
)
pointwise = nn.Conv(features=out_channels, kernel_size=(1, 1), use_bias=bias)

depthwise_params = depthwise.init(depthwise_key, x)["params"]

# To initialize the pointwise layer, we need the output shape of the depthwise layer
# For 'VALID' padding: output_dim = input_dim - (kernel_dim - 1) * dilation
intermediate_height = height - (kernel_size - 1) * dilation
intermediate_width = width - (kernel_size - 1) * dilation
intermediate_shape = (batch_size, intermediate_height, intermediate_width, in_channels)
dummy_intermediate = jnp.zeros(intermediate_shape)

pointwise_params = pointwise.init(pointwise_key, dummy_intermediate)["params"]


# Computation
def kernel(x_ref, depthwise_kernel_ref, pointwise_kernel_ref, out_ref):
  """Pallas kernel for depthwise separable convolution.

  This kernel performs a depthwise convolution followed by a pointwise
  convolution. It operates on tiles of the input data.

  Args:
    x_ref: A reference to a tile of the input tensor.
    depthwise_kernel_ref: A reference to the depthwise convolution kernel.
    pointwise_kernel_ref: A reference to the pointwise convolution kernel.
    out_ref: A reference to the output tile to be written to.
  """
  # Get dimension sizes from the shapes of the references. This makes the
  # kernel more general and less dependent on hardcoded values.
  bH = out_ref.shape[1]
  bW = out_ref.shape[2]
  in_channels = x_ref.shape[-1]
  # The kernel size can be inferred from the difference in the height of the
  # input tile and the output tile for a 'VALID' convolution.
  kernel_size = x_ref.shape[1] - bH + 1

  # --- Stage 1: Depthwise Convolution ---

  # The input tile `x_ref` has its width dimension padded for memory alignment.
  # We slice it back to its logical, unpadded size for the convolution.
  x_tile_logical_width = bW + kernel_size - 1
  x_tile = jax.lax.slice_in_dim(x_ref[...], 0, x_tile_logical_width, axis=2)

  # The depthwise kernel has its input feature dimension (axis=2) padded.
  # For a depthwise convolution, this dimension is logically 1. We slice it.
  dw_kernel = jax.lax.slice_in_dim(depthwise_kernel_ref[...], 0, 1, axis=2)

  # Perform the depthwise convolution. `feature_group_count` is set to
  # `in_channels` to apply a different filter to each input channel.
  intermediate = tpu.convolution(
    x_tile, dw_kernel, window_strides=(1, 1), padding="VALID", feature_group_count=in_channels
  )

  # --- Stage 2: Pointwise Convolution (1x1) ---

  # The pointwise kernel has its input channel dimension (axis=2) padded.
  # We slice it back to its logical size, which is `in_channels`.
  pw_kernel = jax.lax.slice_in_dim(pointwise_kernel_ref[...], 0, in_channels, axis=2)

  # Perform the pointwise convolution. This is a standard 1x1 convolution
  # that mixes the channels from the depthwise stage.
  output = tpu.convolution(intermediate, pw_kernel, window_strides=(1, 1), padding="VALID")

  # --- Stage 3: Write Output ---

  # Write the computed output tile to the output reference.
  out_ref[...] = output


# Define block sizes for tiling the output computation.
bH = 32
bW = 32

# Calculate intermediate output dimensions after 'VALID' convolution
intermediate_height = height - (kernel_size - 1) * dilation
intermediate_width = width - (kernel_size - 1) * dilation

# Define the output shape for the pallas kernel
out_shape = jax.ShapeDtypeStruct((batch_size, intermediate_height, intermediate_width, out_channels), x.dtype)

# Define the execution grid. We parallelize across the batch and spatial tiles.
grid = (
  batch_size,
  (intermediate_height + bH - 1) // bH,
  (intermediate_width + bW - 1) // bW,
)

# Replace the computation with a pallas_call to the kernel
x = pl.pallas_call(
  kernel,
  out_shape=out_shape,
  grid=grid,
  in_specs=[
    # Input 'x': A block of the input image. The size is determined by the
    # receptive field needed for an output block. The second-to-last dimension
    # is padded from 34 to 40 to be divisible by 8 for TPU compatibility.
    pl.BlockSpec(
      block_shape=(1, bH + kernel_size - 1, (bW + kernel_size - 1 + 7) & ~7, in_channels),
      index_map=lambda b, i, j: (b, i * bH, j * bW, 0),
    ),
    # Depthwise kernel: Replicated for all grid instances. The third dimension
    # is padded from 1 to 8 for TPU compatibility.
    pl.BlockSpec(block_shape=(kernel_size, kernel_size, 8, in_channels), index_map=lambda b, i, j: (0, 0, 0, 0)),
    # Pointwise kernel: Replicated for all grid instances. The third dimension
    # is padded from 3 to 8 for TPU compatibility.
    pl.BlockSpec(block_shape=(1, 1, 8, out_channels), index_map=lambda b, i, j: (0, 0, 0, 0)),
  ],
  out_specs=pl.BlockSpec(block_shape=(1, bH, bW, out_channels), index_map=lambda b, i, j: (b, i * bH, j * bW, 0)),
)(x, depthwise_params["kernel"], pointwise_params["kernel"]).block_until_ready()
