"""Compilability, Shape, and Equivalence tests for SimpleRNN."""

import os
import sys

from absl import app
# Correct Imports
from evaluation.code_agent.pytorch_references.level2.simple_RNN import SimpleRNN as SimpleRNN_Torch
from generated_code.code_agent.v1.Gemini_2_5_pro.SimpleRNN.model_corrected import SimpleRNN as SimpleRNN_Jax
from flax import core
import jax
import jax.numpy as jnp
import numpy as np
import torch


os.environ['JAX_PLATFORMS'] = 'cpu'


def test_pytorch_independent():
  """Sanity check for PyTorch model execution."""
  model = SimpleRNN_Torch(
      input_size=4, hidden_size=8, num_layers=2, output_size=3
  )
  x = torch.randn(2, 5, 4)
  y = model(x)
  assert y.shape == (2, 3)
  assert not torch.isnan(y).any()


def test_jax_independent():
  """Sanity check for JAX model execution."""
  model = SimpleRNN_Jax(hidden_size=8, num_layers=2, output_size=3)
  x = jnp.ones((2, 5, 4))
  variables = model.init(jax.random.PRNGKey(42), x)
  y = model.apply(variables, x)
  assert y.shape == (2, 3)
  assert not jnp.isnan(y).any()


def test_rnn_equivalence(
    batch_size, seq_len, input_size, hidden_size, num_layers, output_size
):
  """Validates that the JAX implementation matches the PyTorch implementation."""
  # --- 1. Initialize PyTorch Model ---
  pt_model = SimpleRNN_Torch(input_size, hidden_size, num_layers, output_size)
  pt_model.eval()

  # --- 2. Initialize JAX Model ---
  jax_model = SimpleRNN_Jax(
      hidden_size=hidden_size, num_layers=num_layers, output_size=output_size
  )

  # Create dummy input for initialization
  dummy_input = jnp.zeros((batch_size, seq_len, input_size))
  rng = jax.random.PRNGKey(0)
  variables = jax_model.init(rng, dummy_input)

  # --- 3. Weight Transfer ---
  new_params = core.unfreeze(variables['params'])

  for i in range(num_layers):
    # Retrieve PyTorch parameters
    pt_w_ih = getattr(pt_model.rnn, f'weight_ih_l{i}').detach().numpy()
    pt_w_hh = getattr(pt_model.rnn, f'weight_hh_l{i}').detach().numpy()
    pt_b_ih = getattr(pt_model.rnn, f'bias_ih_l{i}').detach().numpy()
    pt_b_hh = getattr(pt_model.rnn, f'bias_hh_l{i}').detach().numpy()

    # JAX params are (In, Out), PyTorch are (Out, In)
    new_params[f'w_ih_l{i}'] = pt_w_ih.T
    new_params[f'w_hh_l{i}'] = pt_w_hh.T
    # Sum both PyTorch biases to match single bias term
    new_params[f'bias_l{i}'] = pt_b_ih + pt_b_hh

  new_params['fc']['kernel'] = pt_model.fc.weight.detach().numpy().T
  new_params['fc']['bias'] = pt_model.fc.bias.detach().numpy()

  final_params = {'params': core.freeze(new_params)}

  # --- 4. Generate Random Input ---
  np_input = np.random.randn(batch_size, seq_len, input_size).astype(np.float32)
  pt_input = torch.from_numpy(np_input)
  jax_input = jnp.array(np_input)

  # --- 5. Run Inference ---
  with torch.no_grad():
    pt_out = pt_model(pt_input).numpy()

  jax_out = jax_model.apply(final_params, jax_input)

  # --- 6. Compare Outputs ---
  jax_numpy = np.array(jax_out)
  diff = np.abs(pt_out - jax_numpy)
  max_diff = np.max(diff)
  print(f'\n   >> Max Absolute Difference: {max_diff:.2e}')
  np.testing.assert_allclose(pt_out, jax_out, rtol=1e-5, atol=1e-5)


# pylint: disable=unused-argument
def main(argv):
  # validity test (shape)
  try:
    test_pytorch_independent()
    print('PyTorch Model Shape: VALID (True)')
  except AssertionError as e:
    print(f'PyTorch Model FAILED: {e}')
    sys.exit(1)

  try:
    test_jax_independent()
    print('JAX Model Shape: VALID (True)')
  except AssertionError as e:
    print(f'JAX Model FAILED: {e}')
    sys.exit(1)

  # numerical equivalence test
  try:
    test_rnn_equivalence(
        batch_size=5,
        seq_len=10,
        input_size=4,
        hidden_size=8,
        num_layers=2,
        output_size=2,
    )
    print('Equivalence Test:    VALID (True)')
  except AssertionError as e:
    print(f'Equivalence Test FAILED: {e}')
    sys.exit(1)


if __name__ == '__main__':
  app.run(main)
