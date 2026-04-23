"""Main entry point to run a dummy training loop."""

from absl import app
from examples.jax_mlp import train
import jax

train_step = train.train_step
create_train_state = train.create_train_state


def main(argv):
  del argv  # Unused.
  rng = jax.random.PRNGKey(0)
  _, init_rng, data_rng = jax.random.split(rng, 3)

  learning_rate = 0.01
  state = create_train_state(init_rng, learning_rate)
  num_steps = 10
  batch_size = 32
  input_features = 10
  num_classes = 2

  print("Starting training...")
  for i in range(num_steps):
    step_rng = jax.random.fold_in(data_rng, i)
    batch_images = jax.random.uniform(step_rng, (batch_size, input_features))
    batch_labels = jax.random.randint(step_rng, (batch_size,), 0, num_classes)
    state, loss = train_step(state, batch_images, batch_labels)
    if i % 1 == 0:
      print(f"Step {i+1}, Loss: {loss:.4f}")
  print("Training finished.")


if __name__ == "__main__":
  app.run(main)
