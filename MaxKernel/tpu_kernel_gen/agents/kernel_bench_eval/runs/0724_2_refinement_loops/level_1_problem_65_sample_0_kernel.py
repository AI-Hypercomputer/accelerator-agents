# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = (3, 5)
width = 128
height = 128
stride = 1
bias = False

key = random.PRNGKey(0)
key_init, key_x = random.split(key)

# JAX uses channel-last convention (N, H, W, C)
x = random.normal(key_x, (batch_size, height, width, in_channels))

# In Flax, padding=0 is specified as 'VALID'.
# output_padding=0 and groups=1 from the original are default behaviors.
conv_transpose2d = nn.ConvTranspose(
  features=out_channels, kernel_size=kernel_size, strides=(stride, stride), padding="VALID", use_bias=bias
)
params = conv_transpose2d.init(key_init, x)["params"]


# Computation
def kernel(x_ref, kernel_ref, out_ref):
  """Pallas kernel for 2D transposed convolution with stride 1.

  This kernel processes a single item from the input batch. It iterates over
  each pixel of the input feature map. For each input pixel, it scales the
  convolution kernel by the input pixel's channel values and adds the result
  to the corresponding region in the output feature map.

  Args:
    x_ref: A reference to a single input image with shape
      (1, in_height, in_width, in_channels).
    kernel_ref: A reference to the convolution kernel weights with shape
      (kernel_h, kernel_w, in_channels, out_channels).
    out_ref: A reference to the output feature map buffer, which is modified
      in-place. It has shape (1, out_height, out_width, out_channels).
  """
  # Get shapes for defining loop bounds and slicing.
  _, in_height, in_width, _ = x_ref.shape
  kh, kw, _, _ = kernel_ref.shape

  # Load the kernel weights from the memory reference into a JAX array.
  # This is necessary because jnp.einsum cannot operate on a mix of
  # JAX arrays (from slicing x_ref) and memory references (kernel_ref).
  kernel_val = kernel_ref[...]

  # Initialize the output buffer to zeros. This is crucial because we will be
  # accumulating results into it.
  out_ref[...] = jnp.zeros_like(out_ref)

  # Iterate over each pixel of the input feature map.
  for h in range(in_height):
    for w in range(in_width):
      # For each input pixel (h, w), we "scatter" the kernel onto the output.
      # The input pixel vector x_ref[0, h, w] (shape: in_channels) scales the
      # kernel (shape: kh, kw, in_channels, out_channels).
      # We reshape the pixel vector to (1, in_channels) to work around a
      # TPU lowering issue with dot_general where the LHS operand cannot
      # consist solely of a contracting dimension.
      pixel_val = x_ref[0, h, w]
      pixel_val_reshaped = jnp.expand_dims(pixel_val, axis=0)

      # The einsum performs scaling and summation over the input channels,
      # producing an update tensor of shape (1, kh, kw, out_channels).
      update_reshaped = jnp.einsum("ac,ijco->aijo", pixel_val_reshaped, kernel_val)
      update = jnp.squeeze(update_reshaped, axis=0)

      # Add the computed update to the corresponding slice in the output.
      # The top-left corner of this slice is at (h, w) because the stride is 1.
      # Since the loops are sequential, a standard read-modify-write is sufficient
      # and avoids the use of atomic operations, which are not supported on TPU.
      out_slice = (0, pl.dslice(h, kh), pl.dslice(w, kw), slice(None))
      current_val = out_ref[out_slice]
      out_ref[out_slice] = current_val + update


# The pallas_call replaces the original nn.ConvTranspose computation.
# The parallelization strategy is to map each item in the batch to a separate
# kernel instance. The grid is one-dimensional, with size equal to the batch size.
#
# - Grid: (16,) corresponding to the batch_size.
# - Input `x`: Each kernel instance `i` receives the i-th full image from the batch.
#   This is achieved by slicing `x` along the first (batch) dimension.
# - Input `kernel`: The kernel weights are needed by all instances, so the full
#   tensor is passed to each without slicing.
# - Output: Each kernel instance `i` is responsible for computing the i-th full
#   output feature map.
output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(
    (batch_size, height + kernel_size[0] - 1, width + kernel_size[1] - 1, out_channels), x.dtype
  ),
  grid=(batch_size,),
  in_specs=[
    pl.BlockSpec(block_shape=(1, height, width, in_channels), index_map=lambda i: (i, 0, 0, 0)),
    pl.BlockSpec(block_shape=params["kernel"].shape, index_map=lambda i: (0, 0, 0, 0)),
  ],
  out_specs=pl.BlockSpec(
    block_shape=(1, height + kernel_size[0] - 1, width + kernel_size[1] - 1, out_channels),
    index_map=lambda i: (i, 0, 0, 0),
  ),
)(x, params["kernel"]).block_until_ready()
