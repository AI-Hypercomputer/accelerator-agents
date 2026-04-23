"""Training loop components for the SimpleMLP."""

from examples.jax_mlp import model
from flax.training import train_state
import jax
import jax.numpy as jnp
import optax


SimpleMLP = model.SimpleMLP


def create_train_state(
    rng: jax.Array, learning_rate: float
) -> train_state.TrainState:
  """Creates initial `TrainState`."""
  mlp_model = SimpleMLP()
  params = mlp_model.init(rng, jnp.ones([1, 10]))['params']
  tx = optax.adam(learning_rate)
  return train_state.TrainState.create(
      apply_fn=mlp_model.apply, params=params, tx=tx
  )


@jax.jit
def train_step(
    state: train_state.TrainState,
    batch_images: jax.Array,
    batch_labels: jax.Array,
):
  """Train for a single step."""

  def loss_fn(params):
    logits = state.apply_fn({'params': params}, batch_images)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch_labels
    ).mean()
    return loss

  grad_fn = jax.value_and_grad(loss_fn)
  loss, grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  return state, loss
