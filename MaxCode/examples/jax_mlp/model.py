"""A simple MLP model defined in Flax."""

import flax.linen as nn


class SimpleMLP(nn.Module):
  """A simple MLP with one hidden layer."""

  num_classes: int = 2

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(features=64, name="hidden")(x)
    x = nn.relu(x)
    x = nn.Dense(features=self.num_classes, name="output")(x)
    return x
