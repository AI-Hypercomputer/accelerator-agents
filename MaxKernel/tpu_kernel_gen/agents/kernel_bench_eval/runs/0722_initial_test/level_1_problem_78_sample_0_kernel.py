# Imports
import flax.linen as nn
import jax
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = (3, 5)
height = 128
width = 256
stride = (1, 1)
# In PyTorch, padding=(1, 2) with this kernel/stride configuration
# keeps spatial dimensions the same, which is equivalent to 'SAME' in JAX.
padding = "SAME"
bias = False

key = random.PRNGKey(0)
key_params, key_x = random.split(key)

# JAX expects channels-last format: (N, H, W, C)
x = random.normal(key_x, (batch_size, height, width, in_channels))

# We use the Flax layer just to get the correctly initialized kernel weights.
# The actual computation will be done by our Pallas kernel.
conv_transpose2d = nn.ConvTranspose(
  features=out_channels, kernel_size=kernel_size, strides=stride, padding=padding, use_bias=bias
)
params = conv_transpose2d.init(key_params, x)["params"]
kernel_param = params["kernel"]

# Define block sizes for tiling the computation
# We tile over the output height and width dimensions.
# bH and bW must be chosen to satisfy TPU constraints.
# For a 2D block, the inner dimension (bW) must be a multiple of 128.
bH = 32
bW = 128


# Computation
def kernel(x_ref, kernel_ref, out_ref):
  """Pallas kernel for 2D transposed convolution.

  This kernel computes a single block of the output of a transposed convolution.
  The `pallas_call` is configured to handle the 'transposed' aspect by feeding
  this kernel appropriately sliced input data (`x_ref`). As a result, the
  computation within the kernel itself is a standard 'VALID' convolution.

  Args:
    x_ref: A reference to a block of the input tensor, already padded
      as needed for the convolution.
    kernel_ref: A reference to the full convolution kernel weights.
    out_ref: A reference to the output block to be computed and written to.
  """
  # Define the dimension numbers for the convolution. JAX uses these to
  # interpret the layout of the input, kernel, and output tensors.
  # 'NHWC': Batch, Height, Width, Channels for input/output.
  # 'HWIO': Height, Width, Input Channels, Output Channels for the kernel.
  dn = ("NHWC", "HWIO", "NHWC")

  # The input x_ref may be larger than logically needed due to padding
  # for memory alignment. We must slice it to the exact size required
  # for a 'VALID' convolution that produces an output of shape out_ref.shape.
  # The formula for 'VALID' conv output size is:
  #   out_size = (in_size - kernel_size) + 1
  # Rearranging for in_size:
  #   in_size = out_size + kernel_size - 1
  in_height = out_ref.shape[1] + kernel_ref.shape[0] - 1
  in_width = out_ref.shape[2] + kernel_ref.shape[1] - 1

  # Perform the convolution using jax.lax.conv_general_dilated.
  # - The input is a slice of `x_ref` to undo alignment padding.
  # - The kernel is `kernel_ref[...]`.
  # - `window_strides` is (1, 1) as specified in the original problem.
  # - `padding` is 'VALID' because the input is sliced to the exact size
  #   needed to produce an output of `out_ref`'s shape.
  # The result is written directly into the output buffer `out_ref`.
  out_ref[...] = jax.lax.conv_general_dilated(
    x_ref[:, :in_height, :in_width, :],
    kernel_ref[...],
    window_strides=(1, 1),
    padding="VALID",
    dimension_numbers=dn,
  )


# The input block shape must conform to TPU constraints. Specifically, the
# second-to-last dimension must be a multiple of 8, and the last a multiple
# of 128 (or equal to the full dimension size).
# We calculate the required input block size for a 'VALID' convolution,
# then pad it up to the next multiple that satisfies the constraints.
in_block_h = bH + kernel_size[0] - 1
in_block_w = bW + kernel_size[1] - 1
# Pad the height to be a multiple of 8
padded_in_block_h = (in_block_h + 7) & ~7
# Pad the width to be a multiple of 128
padded_in_block_w = (in_block_w + 127) & ~127

output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, height, width, out_channels), x.dtype),
  grid=(batch_size, height // bH, width // bW),
  in_specs=[
    pl.BlockSpec(
      (1, padded_in_block_h, padded_in_block_w, in_channels),
      lambda n, h, w: (
        n,
        h * bH - (kernel_size[0] // 2),
        w * bW - (kernel_size[1] // 2),
        0,
      ),
    ),
    pl.BlockSpec(kernel_param.shape, lambda *_: (0, 0, 0, 0)),
  ],
  out_specs=pl.BlockSpec((1, bH, bW, out_channels), lambda n, h, w: (n, h * bH, w * bW, 0)),
  compiler_params=dict(mosaic=dict(dimension_semantics=("parallel", "parallel", "parallel"))),
)(x, kernel_param).block_until_ready()
