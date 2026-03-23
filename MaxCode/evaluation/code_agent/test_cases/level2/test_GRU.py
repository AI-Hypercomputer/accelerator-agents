"""Compilability and shape checks for GRUForecaster."""

from collections.abc import Sequence
import os
import traceback

# pylint: disable=g-importing-member
from absl import app
from absl import flags
from evaluation.code_agent.pytorch_references.level2.GRU import GRUForecaster as GRUForecaster_Torch
from evaluation.proto import result_pb2
from generated_code.code_agent.v1.Gemini_2_5_pro.GRU.model import GRUForecaster as GRUForecaster_Jax
import immutabledict
import jax
from jax import config as jax_config
import jax.numpy as jnp
import torch

from google3.net.proto2.python.public import text_format


os.environ['JAX_PLATFORMS'] = 'cpu'
jax_config.update('jax_default_matmul_precision', 'highest')

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'metrics_output_path',
    None,
    'Path to write the evaluation metrics.',
)

CONFIG = immutabledict.immutabledict({
    'inputs': 5,
    'hidden_size': 10,
    'num_layers': 2,
    'output_size': 3,
    'batch_size': 4,
    'seq_len': 8,
})


def test_pytorch_independent():
  """Sanity check for PyTorch model execution."""
  model = GRUForecaster_Torch(
      input_size=CONFIG['inputs'],
      hidden_size=CONFIG['hidden_size'],
      num_layers=CONFIG['num_layers'],
      output_size=CONFIG['output_size'],
  )
  model.eval()
  x = torch.ones(CONFIG['batch_size'], CONFIG['seq_len'], CONFIG['inputs'])
  y = model(x)

  expected_shape = (CONFIG['batch_size'], CONFIG['output_size'])
  assert (
      y.shape == expected_shape
  ), f'PyTorch Shape Mismatch: Expected {expected_shape}, got {y.shape}'
  assert not torch.isnan(y).any(), 'PyTorch Output contains NaNs'


def test_jax_independent():
  """Sanity check for JAX model execution."""
  model = GRUForecaster_Jax(
      input_size=CONFIG['inputs'],
      hidden_size=CONFIG['hidden_size'],
      num_layers=CONFIG['num_layers'],
      output_size=CONFIG['output_size'],
  )
  x = jnp.ones((CONFIG['batch_size'], CONFIG['seq_len'], CONFIG['inputs']))

  # Initialize variables
  rng = jax.random.PRNGKey(42)
  variables = model.init(rng, x, deterministic=True)

  # Run inference
  y = model.apply(variables, x, deterministic=True)

  expected_shape = (CONFIG['batch_size'], CONFIG['output_size'])
  assert (
      y.shape == expected_shape
  ), f'JAX Shape Mismatch: Expected {expected_shape}, got {y.shape}'
  assert not jnp.isnan(y).any(), 'JAX Output contains NaNs'


def main(argv: Sequence[str]):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  try:
    test_pytorch_independent()
    pytorch_result = result_pb2.Result(valid=True)
    print('PyTorch Model Shape: VALID (True)')
  except (AssertionError, TypeError, ValueError, RuntimeError) as e:
    error_message = ''.join(traceback.format_exception_only(type(e), e))
    pytorch_result = result_pb2.Result(valid=False, error_message=error_message)
    print(f'PyTorch Model FAILED: {e}')

  try:
    test_jax_independent()
    jax_result = result_pb2.Result(valid=True)
    print('JAX Model Shape:     VALID (True)')
  except (AssertionError, TypeError, ValueError, RuntimeError) as e:
    error_message = ''.join(traceback.format_exception_only(type(e), e))
    jax_result = result_pb2.Result(valid=False, error_message=error_message)
    print(f'JAX Model FAILED: {e}')

  if FLAGS.metrics_output_path:
    eval_result = result_pb2.EvaluationResult(
        pytorch_result=pytorch_result,
        jax_result=jax_result,
    )
    with open(FLAGS.metrics_output_path, 'w') as f:
      f.write(text_format.MessageToString(eval_result))


if __name__ == '__main__':
  app.run(main)
