"""Compilability, Shape, and Equivalence tests for SimpleCNN."""

import os
import traceback

from absl import app
from absl import flags
import evaluation.code_agent.pytorch_references.level2.simple_CNN as SimpleCNN_Torch
from evaluation.proto import result_pb2
import generated_code.code_agent.v1.Gemini_2_5_pro.SimpleCNN.model as model_jax
import flax
import jax
import jax.numpy as jnp
import numpy as np
import torch

from google3.net.proto2.python.public import text_format


os.environ['JAX_PLATFORMS'] = 'cpu'

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'metrics_output_path',
    None,
    'Path to write the evaluation metrics.',
)


def test_pytorch_model_validity():
  """Validates that the PyTorch model runs without errors."""
  model = SimpleCNN_Torch.SimpleCNN()
  input_tensor = torch.randn(4, 3, 32, 32)
  output = model(input_tensor)

  assert output.shape == (
      4,
      10,
  ), f'Expected output shape (4, 10), got {output.shape}'
  assert not torch.isnan(output).any(), 'Output contains NaNs'


def test_jax_model_validity():
  """Validates that the JAX model runs without errors."""
  model = model_jax.SimpleCNN()
  key = jax.random.PRNGKey(0)
  # Flax Conv expects NHWC by default
  input_shape = (4, 32, 32, 3)
  dummy_input = jnp.ones(input_shape)

  params = model.init(key, dummy_input)
  output = model.apply(params, dummy_input)

  assert output.shape == (
      4,
      10,
  ), f'Expected output shape (4, 10), got {output.shape}'
  assert not jnp.isnan(output).any(), 'Output contains NaNs'


def test_pytorch_jax_equivalence():
  """Compares PyTorch and JAX model outputs.

  Compares the outputs of PyTorch and JAX models given the same weights and
  inputs.
  """
  # 1. Initialize PyTorch Model
  torch_model = SimpleCNN_Torch.SimpleCNN()
  torch_model.eval()

  # 2. Initialize JAX Model
  jax_model = model_jax.SimpleCNN()
  key = jax.random.PRNGKey(42)

  # JAX expects NHWC
  dummy_input_jax = jnp.zeros((1, 32, 32, 3))
  params = jax_model.init(key, dummy_input_jax)

  # 3. Transfer Weights from PyTorch to JAX
  new_params = params

  def get_w(layer):
    return layer.weight.detach().numpy()

  def get_b(layer):
    return layer.bias.detach().numpy()

  # --- Conv1 ---
  new_params['params']['Conv_0']['kernel'] = get_w(torch_model.conv1).transpose(
      2, 3, 1, 0
  )
  new_params['params']['Conv_0']['bias'] = get_b(torch_model.conv1)

  # --- Conv2 ---
  new_params['params']['Conv_1']['kernel'] = get_w(torch_model.conv2).transpose(
      2, 3, 1, 0
  )
  new_params['params']['Conv_1']['bias'] = get_b(torch_model.conv2)

  # --- FC1 ---
  w_fc1 = get_w(torch_model.fc1)
  w_fc1 = w_fc1.reshape(256, 32, 8, 8)
  w_fc1 = w_fc1.transpose(0, 2, 3, 1)
  w_fc1 = w_fc1.reshape(256, -1)
  new_params['params']['Dense_0']['kernel'] = w_fc1.T
  new_params['params']['Dense_0']['bias'] = get_b(torch_model.fc1)

  # --- FC2 ---
  new_params['params']['Dense_1']['kernel'] = get_w(torch_model.fc2).T
  new_params['params']['Dense_1']['bias'] = get_b(torch_model.fc2)

  params = flax.core.freeze(new_params)

  # 4. Generate Input
  batch_size = 5
  input_np = np.random.randn(batch_size, 3, 32, 32).astype(np.float32)

  input_torch = torch.from_numpy(input_np)
  input_jax = jnp.array(input_np.transpose(0, 2, 3, 1))

  # 5. Run Inference
  with torch.no_grad():
    out_torch = torch_model(input_torch).numpy()

  out_jax = jax_model.apply(params, input_jax)

  # 6. Compare
  jax_numpy = np.array(out_jax)
  diff = np.abs(out_torch - jax_numpy)
  max_diff = np.max(diff)
  print(f'\n   >> Max Absolute Difference: {max_diff:.2e}')
  np.testing.assert_allclose(out_torch, out_jax, rtol=1e-5, atol=1e-5)


# pylint: disable=unused-argument
def main(argv):
  # validity test (shape)
  try:
    test_pytorch_model_validity()
    pytorch_result = result_pb2.Result(valid=True)
    print('PyTorch Model Shape: VALID (True)')
  except (AssertionError, RuntimeError, TypeError, ValueError) as e:
    error_message = ''.join(traceback.format_exception_only(type(e), e))
    pytorch_result = result_pb2.Result(valid=False, error_message=error_message)
    print(f'PyTorch Model FAILED: {e}')

  try:
    test_jax_model_validity()
    jax_result = result_pb2.Result(valid=True)
    print('JAX Model Shape: VALID (True)')
  except (AssertionError, RuntimeError, TypeError, ValueError) as e:
    error_message = ''.join(traceback.format_exception_only(type(e), e))
    jax_result = result_pb2.Result(valid=False, error_message=error_message)
    print(f'JAX Model FAILED: {e}')

  # numerical equivalence test
  try:
    test_pytorch_jax_equivalence()
    print('Equivalence Test:    VALID (True)')
  except (AssertionError, RuntimeError, TypeError, ValueError) as e:
    print(f'Equivalence Test FAILED: {e}')

  if FLAGS.metrics_output_path:
    eval_result = result_pb2.EvaluationResult(
        pytorch_result=pytorch_result,
        jax_result=jax_result,
    )
    with open(FLAGS.metrics_output_path, 'w') as f:
      f.write(text_format.MessageToString(eval_result))


if __name__ == '__main__':
  app.run(main)
