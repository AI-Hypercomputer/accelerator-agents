# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy.special
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
in_features = 1024
out_features = 512

key = random.PRNGKey(0)
key_x, key_w, key_b, key_sub = random.split(key, 4)

x = random.normal(key_x, (batch_size, in_features))
gemm_weight = random.normal(key_w, (out_features, in_features))
gemm_bias = random.normal(key_b, (out_features,))
subtract_tensor = random.normal(key_sub, (out_features,))


# Computation
def kernel(x_ref, gemm_weight_ref, gemm_bias_ref, subtract_tensor_ref, out_ref):
  # Load the input slice and keep a copy of the original.
  original_x = x_ref[...]

  # Perform the GEMM operation, add bias, and subtract the tensor.
  x = jnp.dot(original_x, gemm_weight_ref[...].T) + gemm_bias_ref[...]
  x = x - subtract_tensor_ref[...]

  # Perform the reduction and activation functions.
  x = jnp.mean(x, axis=1, keepdims=True)
  x = jax.scipy.special.logsumexp(x, axis=1, keepdims=True)
  x = nn.gelu(x)

  # Add the result back to the original input slice (with broadcasting).
  result = x + original_x

  # Write the final result to the output buffer.
  out_ref[...] = result


# Define a block size for the batch dimension.
# This value must be divisible by 8 for TPU compatibility.
bM = 8

result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(batch_size // bM,),
  in_specs=[
    pl.BlockSpec(block_shape=(bM, in_features), index_map=lambda i: (i, 0)),
    pl.BlockSpec(block_shape=(out_features, in_features), index_map=lambda i: (0, 0)),
    pl.BlockSpec(block_shape=(out_features,), index_map=lambda i: (0,)),
    pl.BlockSpec(block_shape=(out_features,), index_map=lambda i: (0,)),
  ],
  out_specs=pl.BlockSpec(block_shape=(bM, in_features), index_map=lambda i: (i, 0)),
)(x, gemm_weight, gemm_bias, subtract_tensor).block_until_ready()
