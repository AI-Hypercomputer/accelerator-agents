# Imports
import jax
import jax.lax
import jax.numpy as jnp
import jax.random as random
from flax import linen as nn
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
in_features = 3
out_features = 16
height, width = 32, 32
kernel_size = (3, 3)
strides = (2, 2)
# Using 'SAME' padding in Flax with stride=2 will double the input spatial dimensions.
padding = "SAME"
multiplier = 0.5

key = random.PRNGKey(0)
key_x, key_params = random.split(key)

# JAX uses channels-last (NHWC) by default
x = random.normal(key_x, (batch_size, height, width, in_features))


class ConvTransposeModel(nn.Module):
  @nn.compact
  def __call__(self, x):
    # Flax's ConvTranspose with use_bias=True (default) will create both
    # kernel and bias parameters. We explicitly name the layer for predictable
    # parameter dictionary keys.
    return nn.ConvTranspose(
      features=out_features, kernel_size=kernel_size, strides=strides, padding=padding, name="ConvTranspose_0"
    )(x)


conv_transpose = ConvTransposeModel()
params = conv_transpose.init(key_params, x)["params"]

# Computation
# Step 1: Perform the transposed convolution using standard Flax/JAX, as this
# operation is not supported inside a Pallas kernel on TPU.
conv_transpose_layer = ConvTransposeModel()
y_conv = conv_transpose_layer.apply({"params": params}, x)

# The output spatial dimensions are doubled by the convolution.
out_height, out_width = height * strides[0], width * strides[1]


def bias_mul_mean_kernel(y_ref, bias_ref, out_ref):
  """
  Pallas kernel for the sequence of bias add, scalar multiplication,
  and mean reduction.
  """
  # Add the bias term.
  y = y_ref[...] + bias_ref[...]

  # Apply the scalar multiplier.
  y = y * multiplier

  # Compute the mean over the spatial dimensions (height and width).
  mean_y = jnp.mean(y, axis=(1, 2), keepdims=True)

  # Write the final result to the output buffer.
  out_ref[...] = mean_y


# Step 2: Use pallas_call to execute the kernel for the remaining operations.
# The scalar 'multiplier' is captured by the kernel's closure via the lambda.
x = pl.pallas_call(
  bias_mul_mean_kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, 1, 1, out_features), x.dtype),
  grid=(batch_size,),
  in_specs=[
    pl.BlockSpec(block_shape=(1, out_height, out_width, out_features), index_map=lambda i: (i, 0, 0, 0)),
    pl.BlockSpec(block_shape=(out_features,), index_map=lambda i: (0,)),
  ],
  out_specs=pl.BlockSpec(block_shape=(1, 1, 1, out_features), index_map=lambda i: (i, 0, 0, 0)),
)(y_conv, params["ConvTranspose_0"]["bias"]).block_until_ready()
