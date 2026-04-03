# Imports
import flax.linen as nn
import jax
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 3
width = 256
height = 256
depth = 10
stride = 1
padding = "VALID"
dilation = 1
groups = 1
bias = False

key = random.PRNGKey(0)
key, x_key, init_key = random.split(key, 3)

# Note: JAX uses channels-last convention (NHWDC)
x = random.normal(x_key, (batch_size, height, width, depth, in_channels))

conv3d = nn.Conv(
  features=out_channels,
  kernel_size=(kernel_size, kernel_size, 1),
  strides=stride,
  padding=padding,
  kernel_dilation=dilation,
  feature_group_count=groups,
  use_bias=bias,
)
params = conv3d.init(init_key, x)["params"]

# Calculate output dimensions for 'VALID' padding
out_height = height - kernel_size + 1
out_width = width - kernel_size + 1

# The kernel weight 'w' is extracted from the initialized parameters
w = params["kernel"]


# Computation
def kernel(x_ref, w_ref, out_ref):
  """Pallas kernel for a 3D convolution.

  This kernel computes one output pixel for a given batch, height, and width,
  across all depth slices and output channels.

  Args:
    x_ref: A reference to the input patch.
      Shape: (1, kernel_size, kernel_size, depth, in_channels)
             (1, 3, 3, 10, 3)
    w_ref: A reference to the convolution kernel weights.
      Shape: (kernel_size, kernel_size, 1, in_channels, out_channels)
             (3, 3, 1, 3, 64)
    out_ref: A reference to the output slice to be written to.
      Shape: (1, 1, 1, depth, out_channels)
             (1, 1, 1, 10, 64)
  """
  # Load the data from the references into local variables.
  x = x_ref[...]
  w = w_ref[...]

  # The convolution can be expressed as a tensor contraction (tensordot).
  # We first slice out the singleton dimension (kernel_depth) from the weights.
  # w shape: (3, 3, 1, 3, 64) -> (3, 3, 3, 64)
  w_sliced = w[:, :, 0, :, :]

  # Reshape inputs for matmul.
  # x: (1, 3, 3, 10, 3) -> (10, 27)
  # This requires transposing the depth dimension to the front before reshaping.
  x_transposed = x.transpose(3, 0, 1, 2, 4)
  x_reshaped = x_transposed.reshape(depth, -1)
  # w: (3, 3, 3, 64) -> (27, 64)
  w_reshaped = w_sliced.reshape(-1, out_channels)

  # Perform the matmul.
  # (10, 27) @ (27, 64) -> (10, 64)
  conv_out = x_reshaped @ w_reshaped

  # Reshape the result to the expected output slice shape and write it.
  # (10, 64) -> (1, 1, 1, 10, 64)
  out_ref[...] = conv_out.reshape(1, 1, 1, depth, out_channels)


output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, out_height, out_width, depth, out_channels), x.dtype),
  grid=(batch_size, out_height, out_width),
  in_specs=[
    # Input 'x' is sliced. Each grid program gets a (1, 3, 3, 10, 3) patch.
    pl.BlockSpec(
      block_shape=(1, kernel_size, kernel_size, depth, in_channels),
      index_map=lambda b, h, w: (b, h, w, 0, 0),
    ),
    # Kernel 'w' is broadcast. Each grid program gets the full (3, 3, 1, 3, 64) kernel.
    pl.BlockSpec(
      block_shape=(kernel_size, kernel_size, 1, in_channels, out_channels),
      index_map=lambda b, h, w: (0, 0, 0, 0, 0),
    ),
  ],
  out_specs=pl.BlockSpec(
    # Each grid program writes to a (1, 1, 1, 10, 64) slice of the output.
    block_shape=(1, 1, 1, depth, out_channels),
    index_map=lambda b, h, w: (b, h, w, 0, 0),
  ),
)(x, w).block_until_ready()
