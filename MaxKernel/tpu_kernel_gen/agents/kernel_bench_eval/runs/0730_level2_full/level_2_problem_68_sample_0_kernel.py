# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
in_features = 10
out_features = 5
constant = 2.0
key = random.PRNGKey(0)
key_x, key_weight, key_bias = random.split(key, 3)

x = random.normal(key_x, (batch_size, in_features))
weight = random.normal(key_weight, (out_features, in_features))
bias = random.normal(key_bias, (out_features,))
# The constant is passed to the kernel via closure, so we don't need a JAX array for it.


# Computation
def kernel(x_ref, weight_ref, bias_ref, out_ref):
  """Pallas kernel for a simple neural network layer.

  Args:
    x_ref: A reference to a slice of the input tensor `x`.
    weight_ref: A reference to the weight matrix.
    bias_ref: A reference to the bias vector.
    out_ref: A reference to a slice of the output tensor for storing the result.
  """
  # Unpack the references to load data from HBM into SRAM.
  x_val = x_ref[...]
  weight_val = weight_ref[...]
  bias_val = bias_ref[...]

  # Perform the dot product of the input slice with the transposed weight matrix.
  # x_val shape: (batch_block_size, in_features)
  # weight_val.T shape: (in_features, out_features)
  # The result will have a shape of (batch_block_size, out_features).
  y = jnp.dot(x_val, jnp.transpose(weight_val))

  # Add the bias vector. Broadcasting handles the addition across the batch dim.
  # y shape: (batch_block_size, out_features)
  # bias_val shape: (out_features,)
  y = y + bias_val

  # Apply the element-wise minimum operation with the constant.
  # The scalar constant is broadcast to the shape of y.
  y = jnp.minimum(y, constant)

  # Subtract the constant from the result.
  y = y - constant

  # Store the final result in the output buffer.
  out_ref[...] = y


# For TPU compatibility, the block size for the batch dimension must be a multiple of 8.
batch_block_size = 8

# The kernel is parallelized across the batch dimension. Each kernel instance
# processes a block of `batch_block_size` rows from 'x' to produce a
# corresponding block in the output.
# - grid: A 1D grid of size `batch_size // batch_block_size`.
# - in_specs:
#   - x: For grid instance `i`, we need a block of `batch_block_size` rows.
#   - weight, bias: The entire tensors are needed for each computation.
# - out_specs:
#   - For grid instance `i`, we write to a block of `batch_block_size` rows in the output.
x = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, out_features), x.dtype),
  grid=(batch_size // batch_block_size,),
  in_specs=[
    pl.BlockSpec(block_shape=(batch_block_size, in_features), index_map=lambda i: (i * batch_block_size, 0)),
    pl.BlockSpec(block_shape=(out_features, in_features), index_map=lambda i: (0, 0)),
    pl.BlockSpec(block_shape=(out_features,), index_map=lambda i: (0,)),
  ],
  out_specs=pl.BlockSpec(block_shape=(batch_block_size, out_features), index_map=lambda i: (i * batch_block_size, 0)),
)(x, weight, bias).block_until_ready()
