"""PyTorch reference implementation of GenerativeAdversarialNetwork."""

import numpy as np
from torch import nn


# Generator Network
class Generator(nn.Module):
  """Generator network in PyTorch."""

  def __init__(self, latent_dim: int, img_shape: tuple[int, ...]):
    """Initializes the Generator."""
    super(Generator, self).__init__()
    self.model = nn.Sequential(
        nn.Linear(latent_dim, 128),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(128, 256),
        nn.BatchNorm1d(256, momentum=0.8),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(256, 512),
        nn.BatchNorm1d(512, momentum=0.8),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(512, int(np.prod(img_shape))),
        nn.Tanh(),
    )

  def forward(self, z):
    """Forward pass of the generator."""
    return self.model(z)


# Discriminator Network
class Discriminator(nn.Module):
  """Discriminator network in PyTorch."""

  def __init__(self, img_shape: tuple[int, ...]):
    """Initializes the Discriminator."""
    super(Discriminator, self).__init__()
    self.model = nn.Sequential(
        nn.Linear(int(np.prod(img_shape)), 512),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(512, 256),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(256, 1),
        nn.Sigmoid(),
    )

  def forward(self, img):
    """Forward pass of the discriminator."""
    return self.model(img)
