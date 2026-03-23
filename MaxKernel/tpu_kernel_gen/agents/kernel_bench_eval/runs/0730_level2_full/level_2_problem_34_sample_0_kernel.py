# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
import jax.tree_util
import optax
from jax.experimental import pallas as pl

# Initialization
batch_size = 64
in_features = 784
out_features = 10
learning_rate = 0.01

key = random.PRNGKey(0)
key_x, key_y, key_init = random.split(key, 3)

x = random.normal(key_x, (batch_size, in_features))
y = random.randint(key_y, (batch_size,), 0, out_features)


class Model(nn.Module):
  @nn.compact
  def __call__(self, x):
    return nn.Dense(features=out_features)(x)


model = Model()
params = model.init(key_init, x)["params"]

optimizer = optax.sgd(learning_rate)
opt_state = optimizer.init(params)


# Computation
def kernel(params_ref, opt_state_ref, x_ref, y_ref, grad_params_ref, params_out_ref, opt_state_out_ref):
  # Phase 1: Parallel Gradient Calculation
  # Forward pass
  logits = x_ref[...] @ params_ref["Dense_0"]["kernel"][...] + params_ref["Dense_0"]["bias"][...]

  # Gradient of softmax cross-entropy loss
  d_logits = (jax.nn.softmax(logits) - y_ref[...]) / batch_size

  # Gradient w.r.t. weights (d_kernel) and bias (d_bias)
  d_kernel = x_ref[...].T @ d_logits
  d_bias = jnp.sum(d_logits, axis=0)

  # Atomically aggregate gradients from all programs.
  pl.atomic_add(grad_params_ref["Dense_0"]["kernel"], (), d_kernel)
  pl.atomic_add(grad_params_ref["Dense_0"]["bias"], (), d_bias)

  # Synchronize programs to ensure all gradients are accumulated.
  pl.sync_threads()

  # Phase 2: Parameter Update (executed by a single program)
  @pl.when(pl.program_id(0) == 0)
  def _update_params():
    # Apply SGD update rule
    params_out_ref["Dense_0"]["kernel"][...] = (
      params_ref["Dense_0"]["kernel"][...] - learning_rate * grad_params_ref["Dense_0"]["kernel"][...]
    )
    params_out_ref["Dense_0"]["bias"][...] = (
      params_ref["Dense_0"]["bias"][...] - learning_rate * grad_params_ref["Dense_0"]["bias"][...]
    )

    # Update optimizer state
    opt_state_out_ref["count"][...] = opt_state_ref["count"][...] + 1


# Prepare inputs for the kernel
grad_params = jax.tree_util.tree_map(jnp.zeros_like, params)
y_one_hot = jax.nn.one_hot(y, out_features)


# JIT-compile and execute the Pallas kernel for one training step.
def full_block_spec(x):
  return pl.BlockSpec(x.shape, lambda *_: (0,) * x.ndim)


params, opt_state = pl.pallas_call(
  kernel,
  out_shape=(params, opt_state),
  grid=(batch_size // 8,),
  in_specs=(
    jax.tree_util.tree_map(full_block_spec, params),
    jax.tree_util.tree_map(full_block_spec, opt_state),
    pl.BlockSpec((8, in_features), lambda i: (i * 8, 0)),
    pl.BlockSpec((8, out_features), lambda i: (i * 8, 0)),
    jax.tree_util.tree_map(full_block_spec, grad_params),
  ),
  out_specs=jax.tree_util.tree_map(full_block_spec, (params, opt_state)),
)(params, opt_state, x, y_one_hot, grad_params)
