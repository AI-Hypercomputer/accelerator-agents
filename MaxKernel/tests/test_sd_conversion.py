# ----------------- PyTorch side -----------------
import torch
import torch.nn as nn
from flax import nnx
from jax import numpy as jnp
from torch.distributed._state_dict_utils import _flatten_state_dict

TORCH_TO_JAX = {
  "kernel": "weight",
}


class TorchModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.linear1 = nn.Linear(3, 4)
    self.relu = nn.ReLU()
    self.linear2 = nn.Linear(4, 2)

  def forward(self, x):
    x = self.linear1(x)
    x = self.relu(x)
    x = self.linear2(x)
    return x


torch_model = TorchModel()
torch_state_dict = torch_model.state_dict()

torch_state_dict, map = _flatten_state_dict(torch_state_dict)

# ----------------- JAX/Flax side -----------------


class FlaxModel(nnx.Module):
  def __init__(self, rngs: nnx.Rngs):
    super().__init__()
    self.linear1 = nnx.Linear(3, 4, rngs=rngs)
    self.relu = nnx.relu
    self.linear2 = nnx.Linear(4, 2, rngs=rngs)

  def __call__(self, x):
    x = self.linear1(x)
    x = self.relu(x)
    x = self.linear2(x)
    return x


# Initialize JAX model
jax_model = FlaxModel(nnx.Rngs(0))

# Confirm outputs do not match:
torch_input = torch.randn(1, 3)
jax_input = jnp.array(torch_input.numpy())

# Get outputs from both models
with torch.no_grad():
  torch_output = torch_model(torch_input)

jax_output = jax_model(jax_input)

# Print the initial outputs to show the difference before conversion
print("Before conversion:")
print("--PyTorch output:", torch_output)
print("--JAX output:", jax_output)
print("--Output difference:", jnp.abs(jnp.array(torch_output.numpy()) - jax_output).max())

# Get jax state dict
jax_graphdef, jax_params, jax_batch_stats = nnx.split(jax_model, nnx.Param, nnx.BatchStat)

jax_state_dict = jax_params.flat_state()

for jk, jv in jax_state_dict:
  jk_concat = ".".join(jk)

  # Convert JAX terms to PyTorch terms based on TERM_DICT
  for torch_term, jax_term in TORCH_TO_JAX.items():
    if torch_term in jk_concat:
      jk_concat = jk_concat.replace(torch_term, jax_term)

  if jk_concat not in torch_state_dict:
    raise ValueError(f"Mismatch in keys: JAX key {jk} does not exist in PyTorch state_dict.")

  # Check if the JAX tensor needs to be transposed
  torch_shape = torch_state_dict[jk_concat].shape
  jax_shape = jv.value.shape

  # If shapes don't match but have same elements, try to transpose
  if torch_shape != jax_shape and torch_shape[::-1] == jax_shape:
    # If weight tensor in linear layer, transpose for PyTorch -> JAX conversion
    torch_tensor = torch_state_dict[jk_concat].numpy().T
    jv.value = jnp.array(torch_tensor)
  elif torch_shape != jax_shape:
    # More complex case: dimensions might be permuted differently
    if torch_shape[0] * torch_shape[1] == jax_shape[0] * jax_shape[1]:
      # Try different permutations if dimensions are compatible
      try:
        # Try simple transpose first
        torch_tensor = torch_state_dict[jk_concat].numpy().T
        if torch_tensor.shape == jax_shape:
          jv.value = jnp.array(torch_tensor)
        else:
          # If still doesn't match, try reshape then transpose
          torch_tensor = torch_state_dict[jk_concat].numpy().reshape(jax_shape)
          jv.value = jnp.array(torch_tensor)
      except Exception as e:
        raise ValueError(f"Cannot reconcile shapes: {torch_shape} vs {jax_shape}. Error: {str(e)}")
    else:
      raise ValueError(f"Incompatible shapes: {torch_shape} vs {jax_shape}")
  else:
    # Shapes match, direct copy
    jv.value = jnp.array(torch_state_dict[jk_concat].numpy())


jax_model = nnx.merge(jax_graphdef, jax_params, jax_batch_stats)

# Verify by comparing the output of both models
torch_input = torch.randn(1, 3)
jax_input = jnp.array(torch_input.numpy())

# Get outputs from both models
with torch.no_grad():
  torch_output = torch_model(torch_input)

jax_output = jax_model(jax_input)

print("After conversion:")
print("--PyTorch output:", torch_output)
print("--JAX output:", jax_output)
print("--Output difference:", jnp.abs(jnp.array(torch_output.numpy()) - jax_output).max())
