"""A simple MLP model defined in PyTorch."""

from torch import nn


class SimpleMLP(nn.Module):
  """A simple MLP with one hidden layer."""

  def __init__(self):
    super(SimpleMLP, self).__init__()
    self.layers = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 2))

  def forward(self, x):
    return self.layers(x)
