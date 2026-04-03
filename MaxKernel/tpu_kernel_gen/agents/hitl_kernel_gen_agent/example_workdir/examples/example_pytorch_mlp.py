"""
Example PyTorch MLP with CUDA operations
Demonstrates PyTorch code that should be converted to JAX
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMLP(nn.Module):
  """Simple multi-layer perceptron."""

  def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
    super(SimpleMLP, self).__init__()
    self.fc1 = nn.Linear(input_dim, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, output_dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass through the network."""
    x = self.fc1(x)
    x = F.relu(x)
    x = self.fc2(x)
    return x


def main():
  """Test the MLP on random data."""
  # Configuration
  batch_size = 128
  input_dim = 784
  hidden_dim = 256
  output_dim = 10

  # Create model and move to GPU
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = SimpleMLP(input_dim, hidden_dim, output_dim).to(device)

  # Create random input
  x = torch.randn(batch_size, input_dim).to(device)

  # Forward pass
  output = model(x)

  print(f"Input shape: {x.shape}")
  print(f"Output shape: {output.shape}")
  print(f"Output mean: {output.mean().item():.4f}")
  print(f"Output std: {output.std().item():.4f}")


if __name__ == "__main__":
  main()
