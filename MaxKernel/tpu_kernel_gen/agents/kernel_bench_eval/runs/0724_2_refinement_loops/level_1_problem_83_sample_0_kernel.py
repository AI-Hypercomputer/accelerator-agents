# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
in_channels = 3
kernel_size = 3
width = 256
height = 256
stride = 1
padding = "VALID"
dilation = 1
key = random.PRNGKey(0)
key_x, key_params = random.split(key)
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


def kernel(x_ref, kernel_ref, out_ref):
  """Pallas kernel for a single row of a depthwise convolution.

  This kernel computes one row of the output feature map for a single item
  in the batch.

  Args:
    x_ref: A reference to a tile of the input tensor. The shape is
      (1, kernel_size, width, in_channels), representing the receptive
      field needed for the output row.
    kernel_ref: A reference to the entire convolution kernel tensor. The shape
      is (kernel_size, 1, 1, in_channels).
    out_ref: A reference to a tile of the output tensor. The shape is
      (1, 1, width, in_channels), representing the output row to be computed.
  """
  # Squeeze the batch dimension from the input tile.
  x = jnp.squeeze(x_ref[...], axis=0)
  # Squeeze the kernel to remove the singleton dimensions (width and features_out).
  # The shape becomes (kernel_size, in_channels).
  kernel = jnp.squeeze(kernel_ref[...], axis=(1, 2))

  # Perform the depthwise convolution by broadcasting and summing.
  # This is equivalent to `jnp.einsum('kwc,kc->wc', x, kernel)` but can be
  # more robustly compiled.
  # x shape:      (k, w, c)
  # kernel shape: (k, c) -> broadcasted to (k, 1, c)
  # product shape:(k, w, c)
  # sum over k -> (w, c)
  conv_out = jnp.sum(x * kernel[:, None, :], axis=0)

  # Write the result to the output buffer. The output buffer expects
  # leading singleton dimensions for batch and height, so we add them back.
  out_ref[...] = conv_out[None, None, :, :]


# Computation
variables = conv2d.init(key_params, x)
output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, height - kernel_size + 1, width, in_channels), x.dtype),
  grid=(batch_size, height - kernel_size + 1),
  in_specs=[
    # Spec for input 'x': each grid element gets a horizontal slice
    # corresponding to the receptive field needed for one output row.
    pl.BlockSpec(
      block_shape=(1, kernel_size, width, in_channels),
      index_map=lambda b, h: (b, h, 0, 0),
    ),
    # Spec for kernel weights: broadcast the entire kernel to all grid instances.
    pl.BlockSpec(
      block_shape=variables["params"]["kernel"].shape,
      index_map=lambda b, h: tuple([0] * variables["params"]["kernel"].ndim),
    ),
  ],
  out_specs=pl.BlockSpec(
    # Spec for output: each grid element writes to a single output row.
    block_shape=(1, 1, width, in_channels),
    index_map=lambda b, h: (b, h, 0, 0),
  ),
)(x, variables["params"]["kernel"]).block_until_ready()
