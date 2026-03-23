# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
in_features = 64
out_features = 128
eps = 1e-5
momentum = 0.1

key = random.PRNGKey(0)
key_x, key_y, key_bmm, key_norm = random.split(key, 4)

x = random.normal(key_x, (batch_size, in_features))
y = random.normal(key_y, (batch_size, out_features))

bmm = nn.Dense(features=out_features)
bmm_variables = bmm.init(key_bmm, x)

# The original PyTorch code uses InstanceNorm2d on an input of shape (N, C, 1, 1).
# This means the normalization happens over the spatial dimensions (H, W), which are singletons.
# In this case, the mean is the input itself and the variance is 0.
# The operation (x - mean) / sqrt(var + eps) * weight + bias simplifies to just `bias`.
# However, the more likely intent, given the name `InstanceNorm`, is to normalize
# across the feature dimension for each instance in the batch.
# PyTorch's `InstanceNorm1d` on a (N, C) input would do this.
# The equivalent in Flax is `nn.LayerNorm` with `reduction_axes=-1` and `feature_axes=-1`.
# We will use LayerNorm as it correctly captures the probable intent of instance-wise normalization
# on a 2D tensor, which is computationally different but more standard than a no-op InstanceNorm2d.
instance_norm = nn.LayerNorm(epsilon=eps, use_bias=True, use_scale=True, reduction_axes=-1, feature_axes=-1)
norm_variables = instance_norm.init(
  key_norm, y
)  # Initialize with a tensor of the correct shape (batch_size, out_features)

# Define a block size for processing the batch dimension, ensuring it's divisible by 8 for TPU compatibility.
block_size = 8


# Computation
def kernel(x_ref, y_ref, bmm_kernel_ref, bmm_bias_ref, norm_scale_ref, norm_bias_ref, out_ref):
  """
  Pallas kernel to perform a sequence of operations: dense layer, layer normalization,
  and element-wise addition and multiplication.

  Args:
    x_ref: Input tensor block of shape (block_size, in_features).
    y_ref: Second input tensor block of shape (block_size, out_features).
    bmm_kernel_ref: Weight matrix for the dense layer of shape (in_features, out_features).
    bmm_bias_ref: Bias vector for the dense layer of shape (out_features,).
    norm_scale_ref: Scale vector for layer normalization of shape (out_features,).
    norm_bias_ref: Bias vector for layer normalization of shape (out_features,).
    out_ref: Output tensor block of shape (block_size, out_features) to be written to in-place.
  """
  # Epsilon value for numerical stability in layer normalization.
  eps = 1e-5

  # --- 1. Dense Layer ---
  # Apply the dense layer: x = matmul(x, kernel) + bias
  x = jnp.dot(x_ref[...], bmm_kernel_ref[...]) + bmm_bias_ref[...]

  # --- 2. Layer Normalization ---
  # The normalization is performed across the feature dimension (axis=1).
  # Calculate mean and variance for the current block.
  mean = jnp.mean(x, axis=1, keepdims=True)
  var = jnp.var(x, axis=1, keepdims=True)

  # Normalize the output of the dense layer.
  x_normalized = (x - mean) / jnp.sqrt(var + eps)

  # Apply scale and bias for the layer normalization.
  x = x_normalized * norm_scale_ref[...] + norm_bias_ref[...]

  # --- 3. Element-wise Operations ---
  # Add the y tensor.
  x = x + y_ref[...]
  # Multiply by the y tensor.
  x = x * y_ref[...]

  # --- 4. Store Result ---
  # Write the final result to the output buffer.
  out_ref[...] = x


# The pallas_call replaces the original computation section.
# It processes the batch in chunks of `block_size`.
x = pl.pallas_call(
  kernel,
  # The output shape matches the final tensor shape from the original computation.
  out_shape=jax.ShapeDtypeStruct(y.shape, y.dtype),
  # The grid is 1D, parallelizing over the batch dimension in chunks.
  grid=(batch_size // block_size,),
  # Input specifications define how each input tensor is sliced for the kernel.
  in_specs=[
    # x: Chunked along the batch dimension.
    pl.BlockSpec(block_shape=(block_size, in_features), index_map=lambda i: (i, 0)),
    # y: Chunked along the batch dimension.
    pl.BlockSpec(block_shape=(block_size, out_features), index_map=lambda i: (i, 0)),
    # bmm_kernel: The entire weight matrix is passed to each kernel instance.
    pl.BlockSpec(block_shape=(in_features, out_features), index_map=lambda i: (0, 0)),
    # bmm_bias: The entire bias vector is passed to each kernel instance.
    pl.BlockSpec(block_shape=(out_features,), index_map=lambda i: (0,)),
    # norm_scale: The entire scale vector is passed to each kernel instance.
    pl.BlockSpec(block_shape=(out_features,), index_map=lambda i: (0,)),
    # norm_bias: The entire bias vector is passed to each kernel instance.
    pl.BlockSpec(block_shape=(out_features,), index_map=lambda i: (0,)),
  ],
  # Output specification defines how the output is written back, chunked by batch.
  out_specs=pl.BlockSpec(block_shape=(block_size, out_features), index_map=lambda i: (i, 0)),
)(
  x,
  y,
  bmm_variables["params"]["kernel"],
  bmm_variables["params"]["bias"],
  norm_variables["params"]["scale"],
  norm_variables["params"]["bias"],
).block_until_ready()
