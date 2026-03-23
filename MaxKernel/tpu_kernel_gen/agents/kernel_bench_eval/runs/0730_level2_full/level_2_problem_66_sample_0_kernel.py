# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from flax import linen as nn
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
in_features = 100
out_features = 50
dropout_p = 0.2

key = random.PRNGKey(0)
key, x_key, params_key, dropout_key = random.split(key, 4)

x = random.normal(x_key, (batch_size, in_features))

matmul = nn.Dense(features=out_features)
params = matmul.init(params_key, x)["params"]


# Computation
def kernel(x_ref, kernel_ref, bias_ref, dropout_key_ref, out_ref):
  # Define constants from the source computation
  dropout_p = 0.2
  keep_p = 1.0 - dropout_p

  # Apply the dense layer (matrix multiplication + bias)
  # x_ref is a block of shape (32, in_features)
  # kernel_ref is the full weights matrix of shape (in_features, out_features)
  # bias_ref is the full bias vector of shape (out_features,)
  y = x_ref[...] @ kernel_ref[...] + bias_ref[...]

  # Apply dropout
  # To ensure each parallel execution of the kernel uses a unique dropout mask,
  # we fold the program_id into the base dropout key.
  program_id = pl.program_id(0)
  dropout_key = random.fold_in(dropout_key_ref[...], program_id)

  # Generate a dropout mask and apply it.
  # The rate `p` for bernoulli is the keep probability.
  dropout_mask = random.bernoulli(dropout_key, p=keep_p, shape=y.shape)
  # The output is scaled by 1/keep_p to account for the dropped units.
  y = jnp.where(dropout_mask, y / keep_p, 0.0)

  # Compute the mean across the feature dimension
  y = jnp.mean(y, axis=1, keepdims=True)

  # Apply softmax. Note: for a tensor of shape (N, 1), softmax along axis=1
  # is an identity operation that returns a tensor of ones.
  y = jax.nn.softmax(y, axis=1)

  # Write the final result to the output buffer
  out_ref[...] = y


x = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, 1), x.dtype),
  grid=(batch_size // 32,),
  in_specs=[
    pl.BlockSpec(block_shape=(32, in_features), index_map=lambda i: (i, 0)),
    pl.BlockSpec(block_shape=(in_features, out_features), index_map=lambda i: (0, 0)),
    pl.BlockSpec(block_shape=(out_features,), index_map=lambda i: (0,)),
    pl.BlockSpec(block_shape=(2,), index_map=lambda i: (0,)),
  ],
  out_specs=pl.BlockSpec(block_shape=(32, 1), index_map=lambda i: (i, 0)),
)(x, params["kernel"], params["bias"], dropout_key).block_until_ready()
