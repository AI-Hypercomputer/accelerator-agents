# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
in_features = 10
out_features = 5
subtract_value = 2.0
multiply_value = 1.5

key = random.PRNGKey(0)
key, x_key, params_key = random.split(key, 3)

x = random.normal(x_key, (batch_size, in_features))
linear_layer = nn.Dense(features=out_features)
variables = linear_layer.init(params_key, x)

# We define a block size for the batch dimension. To be compliant with TPU hardware
# constraints, this size should ideally be a multiple of 8.
block_size = 8

# The output of the computation.
out_shape = jax.ShapeDtypeStruct((batch_size, out_features), x.dtype)

# We extract the kernel and bias from the Flax variables dictionary to pass them
# as distinct arguments to the Pallas kernel.
kernel_weights = variables["params"]["kernel"]
bias = variables["params"]["bias"]


# Computation
def kernel(x_ref, kernel_ref, bias_ref, out_ref):
  """
  Pallas kernel for a dense layer followed by element-wise operations.

  Args:
    x_ref: A reference to a block of the input data.
    kernel_ref: A reference to the kernel weights of the dense layer.
    bias_ref: A reference to the bias of the dense layer.
    out_ref: A reference to the output block to write the results to.
  """
  # These values are from the original source code.
  subtract_value = 2.0
  multiply_value = 1.5

  # Perform the dense layer computation: x @ W + b
  # The result 'y' is a temporary value held in registers.
  y = jnp.dot(x_ref[...], kernel_ref[...]) + bias_ref[...]

  # Apply the subsequent element-wise operations
  y = y - subtract_value
  y = y * multiply_value
  y = nn.relu(y)

  # Write the final result to the output buffer in SRAM.
  out_ref[...] = y


result = pl.pallas_call(
  kernel,
  out_shape=out_shape,
  # The grid is 1D, iterating over blocks of the batch dimension.
  grid=(batch_size // block_size,),
  in_specs=[
    # For the input `x`, each kernel instance `i` gets a block of `block_size` rows.
    pl.BlockSpec(block_shape=(block_size, in_features), index_map=lambda i: (i * block_size, 0)),
    # The kernel weights are needed by all instances, so we pass the full block.
    pl.BlockSpec(block_shape=(in_features, out_features), index_map=lambda i: (0, 0)),
    # The bias is also needed by all instances. Its index_map must return a tuple
    # of the same rank as the array. Bias is 1D, so we return a 1-tuple.
    pl.BlockSpec(block_shape=(out_features,), index_map=lambda i: (0,)),
  ],
  # The output is chunked similarly to the input `x`, where each instance `i`
  # writes to its corresponding block of rows.
  out_specs=pl.BlockSpec(block_shape=(block_size, out_features), index_map=lambda i: (i * block_size, 0)),
)(x, kernel_weights, bias).block_until_ready()
