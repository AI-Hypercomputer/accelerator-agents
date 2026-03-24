# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
in_channels = 64
out_channels = 3
kernel_size = 3
length = 128
stride = 1
padding = "VALID"  # padding=0 in torch is 'VALID' in JAX/Flax
bias = False

key = random.PRNGKey(0)
key_x, key_params = random.split(key)

# Note: JAX/Flax expect channel-last layout: (batch, length, channels)
x = random.normal(key_x, (batch_size, length, in_channels))

# Use Flax to initialize kernel weights with the correct shape and values
conv1d_transpose = nn.ConvTranspose(
  features=out_channels, kernel_size=(kernel_size,), strides=(stride,), padding=padding, use_bias=bias
)
params = conv1d_transpose.init(key_params, x)["params"]
kernel_weights = params["kernel"]

# Calculate output shape for the Pallas call
output_length = (length - 1) * stride + kernel_size
output_shape = (batch_size, output_length, out_channels)


# Computation
def kernel(x_ref, kernel_ref, out_ref):
  """Pallas kernel for 1D transposed convolution with stride=1.

  This kernel implements a 1D transposed convolution by iterating through each
  element of the input sequence. For each input element (a vector of `in_channels`),
  it performs a batched dot product with the convolution kernel to produce an
  `update` tensor. This `update` tensor is then added ("scattered") onto the
  appropriate slice of the output tensor.

  Args:
    x_ref: A reference to the input tensor block, with shape
      (1, length, in_channels). The first dimension is 1 because each
      kernel instance processes one item from the batch.
    kernel_ref: A reference to the kernel weights, with shape
      (kernel_size, in_channels, out_channels). This is read by all kernel
      instances.
    out_ref: A reference to the output tensor block, with shape
      (1, output_length, out_channels). This is where the result is written
      in-place.
  """
  # Initialize the output buffer with zeros to prepare for accumulation.
  out_ref[...] = jnp.zeros_like(out_ref)

  # Extract dimensions from the shapes of the input references.
  # The stride is fixed to 1 based on the source computation.
  length = x_ref.shape[1]
  kernel_size = kernel_ref.shape[0]
  stride = 1

  # Iterate over each position `l` in the input sequence.
  for l in range(length):
    # Extract the input feature vector of shape (in_channels,) at position `l`.
    # Squeezing is not strictly necessary as JAX handles broadcasting, but it
    # makes the intended shape explicit.
    x_slice = jnp.squeeze(x_ref[0, l, :])

    # Compute the product of the input vector and the kernel.
    # jnp.einsum('i,kio->ko', ...) performs a batched dot product.
    # 'i' corresponds to in_channels.
    # 'k' corresponds to kernel_size.
    # 'o' corresponds to out_channels.
    # The result `update` has shape (kernel_size, out_channels).
    update = jnp.einsum("i,kio->ko", x_slice, kernel_ref[...])

    # Accumulate the `update` into the output buffer. This loop iterates
    # through the `kernel_size` dimension of the `update` tensor.
    for k in range(kernel_size):
      # The output position is calculated based on the input position `l`,
      # the stride, and the current position `k` within the kernel.
      output_index = l * stride + k
      # Add the computed update vector (shape: out_channels) to the
      # corresponding location in the output buffer in-place.
      out_ref[0, output_index, :] += update[k, :]


# Call the Pallas kernel
result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(output_shape, x.dtype),
  grid_spec=pl.GridSpec(
    grid=(batch_size,),
    block_mappings=[
      pl.BlockMapping(block_shape=(1, length, in_channels), index_map=lambda i: (i, 0, 0)),
      pl.BlockMapping(block_shape=kernel_weights.shape, index_map=lambda _: (0, 0, 0)),
      pl.BlockMapping(block_shape=(1, output_length, out_channels), index_map=lambda i: (i, 0, 0)),
    ],
  ),
)(x, kernel_weights).block_until_ready()
