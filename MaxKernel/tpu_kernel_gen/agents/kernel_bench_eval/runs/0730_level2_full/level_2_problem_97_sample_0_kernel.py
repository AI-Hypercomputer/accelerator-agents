# Imports
from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
in_features = 1024
out_features = 512
bn_eps = 1e-5
bn_momentum = 0.1
bias_shape = (1,)
divide_value = 1.0
block_n = 128

key = random.PRNGKey(0)
key_x, key_matmul, key_bn, key_bias = random.split(key, 4)

x = random.normal(key_x, (batch_size, in_features))

matmul = nn.Dense(features=out_features)
matmul_params = matmul.init(key_matmul, x)["params"]

bn = nn.BatchNorm(use_running_average=False, momentum=bn_momentum, epsilon=bn_eps)
bn_vars = bn.init(key_bn, jnp.ones((batch_size, out_features)))

bias = random.normal(key_bias, bias_shape)


# Computation
def kernel(x_ref, matmul_params_ref, bn_vars_ref, bias_ref, x_out_ref, *, divide_value, bn_momentum, bn_eps):
  """Pallas kernel for a sequence of fused operations.

  This kernel performs the following steps:
  1. Dense layer (matrix multiplication + bias).
  2. Batch normalization.
  3. Adds an additional bias.
  4. Divides by a scalar value.
  5. Applies a Swish activation function.

  Args:
    x_ref: Reference to the input data.
    matmul_params_ref: PyTree of references to the dense layer's weights and bias.
    bn_vars_ref: PyTree of references to the batch normalization parameters and
      statistics. The batch statistics are updated in-place.
    bias_ref: Reference to the additional bias vector.
    x_out_ref: Reference to the output buffer.
    divide_value: A scalar value to divide the result by.
    bn_momentum: The momentum for the running average in batch norm.
    bn_eps: A small epsilon value to avoid division by zero in batch norm.
  """
  # Step 1: Dense layer (Matmul + Bias)
  y = x_ref[...] @ matmul_params_ref["kernel"][...]
  y = y + matmul_params_ref["bias"][...]

  # Step 2: Batch Normalization (updates stats in-place)
  # Calculate mean and variance for the current batch
  mean = jnp.mean(y, axis=0)
  var = jnp.var(y, axis=0)

  # Update running mean and variance (leaky average)
  new_mean = (1 - bn_momentum) * bn_vars_ref["batch_stats"]["mean"][...] + bn_momentum * mean
  new_var = (1 - bn_momentum) * bn_vars_ref["batch_stats"]["var"][...] + bn_momentum * var
  bn_vars_ref["batch_stats"]["mean"][...] = new_mean
  bn_vars_ref["batch_stats"]["var"][...] = new_var

  # Normalize using the batch's statistics
  y_norm = (y - mean) / jnp.sqrt(var + bn_eps)

  # Scale and shift
  y_bn = y_norm * bn_vars_ref["params"]["scale"][...] + bn_vars_ref["params"]["bias"][...]

  # Step 3: Add bias
  y_biased = y_bn + bias_ref[...]

  # Step 4: Divide
  y_divided = y_biased / divide_value

  # Step 5: Swish activation function
  y_swish = y_divided * jax.nn.sigmoid(y_divided)

  # Write final result to the output reference
  x_out_ref[...] = y_swish


x, updated_bn_vars = pl.pallas_call(
  partial(kernel, divide_value=divide_value, bn_momentum=bn_momentum, bn_eps=bn_eps),
  # Define the shapes and dtypes of the output arrays
  out_shape=[
    jax.ShapeDtypeStruct(x.shape, x.dtype),
    bn_vars,
  ],
  # The grid is 1D, parallelizing over the output features
  grid=(out_features // block_n,),
  # Specify how input arrays are sliced into blocks for each kernel instance
  in_specs=[
    # x: Load the full (128, 1024) array for each kernel
    pl.BlockSpec(block_shape=(batch_size, in_features), index_map=lambda j: (0, 0)),
    # matmul_params: Load a (1024, 128) slice of weights and a (128,) slice of bias
    # corresponding to the j-th block of output features.
    {
      "kernel": pl.BlockSpec(block_shape=(in_features, block_n), index_map=lambda j: (0, j * block_n)),
      "bias": pl.BlockSpec(block_shape=(block_n,), index_map=lambda j: (j * block_n,)),
    },
    # bn_vars: Load slices of batch stats and params for the j-th block.
    # These are also outputs, so they will be updated in place.
    {
      "batch_stats": {
        "mean": pl.BlockSpec(block_shape=(block_n,), index_map=lambda j: (j * block_n,)),
        "var": pl.BlockSpec(block_shape=(block_n,), index_map=lambda j: (j * block_n,)),
      },
      "params": {
        "scale": pl.BlockSpec(block_shape=(block_n,), index_map=lambda j: (j * block_n,)),
        "bias": pl.BlockSpec(block_shape=(block_n,), index_map=lambda j: (j * block_n,)),
      },
    },
    # bias: Load the full (1,) array for each kernel
    pl.BlockSpec(block_shape=(1,), index_map=lambda j: (0,)),
  ],
  # Specify how the output arrays are constructed from the blocks computed by each kernel
  out_specs=[
    # x: Each kernel j writes to a (128, 128) block of the output.
    pl.BlockSpec(block_shape=(batch_size, block_n), index_map=lambda j: (0, j * block_n)),
    # bn_vars: The updated stats and params are written back to the corresponding slices.
    {
      "batch_stats": {
        "mean": pl.BlockSpec(block_shape=(block_n,), index_map=lambda j: (j * block_n,)),
        "var": pl.BlockSpec(block_shape=(block_n,), index_map=lambda j: (j * block_n,)),
      },
      "params": {
        "scale": pl.BlockSpec(block_shape=(block_n,), index_map=lambda j: (j * block_n,)),
        "bias": pl.BlockSpec(block_shape=(block_n,), index_map=lambda j: (j * block_n,)),
      },
    },
  ],
)(x, matmul_params, bn_vars, bias).block_until_ready()
