"""Compilability, Shape, and Equivalence tests for GenerativeAdversarialNetwork."""

import os
import traceback

from absl import app
from absl import flags
import evaluation.code_agent.pytorch_references.level2.generative_adversarial_network as GAN_Torch
from evaluation.proto import result_pb2
import generated_code.code_agent.v1.Gemini_2_5_pro.GenerativeAdversarialNetwork.model as model_jax
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


def test_pytorch_validity():
  """Validates that the PyTorch models run without errors."""
  latent_dim = 100
  img_shape = (1, 28, 28)
  batch_size = 4

  # Test Generator
  generator = GAN_Torch.Generator(latent_dim, img_shape)
  generator.eval()
  z = torch.randn(batch_size, latent_dim)
  gen_out = generator(z)
  expected_gen_shape = (batch_size, int(np.prod(img_shape)))
  assert gen_out.shape == expected_gen_shape, (
      f'Generator Shape Mismatch: Expected {expected_gen_shape}, got'
      f' {gen_out.shape}'
  )
  assert not torch.isnan(gen_out).any(), 'Generator Output contains NaNs'

  # Test Discriminator
  discriminator = GAN_Torch.Discriminator(img_shape)
  discriminator.eval()
  img_flat = torch.randn(batch_size, int(np.prod(img_shape)))
  disc_out = discriminator(img_flat)
  expected_disc_shape = (batch_size, 1)
  assert disc_out.shape == expected_disc_shape, (
      f'Discriminator Shape Mismatch: Expected {expected_disc_shape}, got'
      f' {disc_out.shape}'
  )
  assert not torch.isnan(disc_out).any(), 'Discriminator Output contains NaNs'


def test_jax_validity():
  """Validates that the JAX models run without errors."""
  latent_dim = 100
  img_shape = (1, 28, 28)
  batch_size = 4

  key = jax.random.PRNGKey(0)

  # Test Generator
  generator = model_jax.Generator(img_shape=img_shape)
  dummy_z = jnp.zeros((batch_size, latent_dim))
  gen_params = generator.init(key, dummy_z)
  gen_out = generator.apply(gen_params, dummy_z, train=False)
  expected_gen_shape = (batch_size, int(np.prod(img_shape)))
  assert gen_out.shape == expected_gen_shape, (
      f'JAX Generator Shape Mismatch: Expected {expected_gen_shape}, got'
      f' {gen_out.shape}'
  )
  assert not jnp.isnan(gen_out).any(), 'JAX Generator Output contains NaNs'

  # Test Discriminator
  discriminator = model_jax.Discriminator()
  dummy_img = jnp.zeros((batch_size, *img_shape))
  disc_params = discriminator.init(key, dummy_img)
  disc_out = discriminator.apply(disc_params, dummy_img)
  expected_disc_shape = (batch_size, 1)
  assert disc_out.shape == expected_disc_shape, (
      f'JAX Discriminator Shape Mismatch: Expected {expected_disc_shape}, got'
      f' {disc_out.shape}'
  )
  assert not jnp.isnan(disc_out).any(), 'JAX Discriminator Output contains NaNs'


def test_generator_equivalence():
  """Tests equivalence of Generator between PyTorch and JAX."""
  latent_dim = 100
  img_shape = (1, 28, 28)
  batch_size = 4

  torch_model = GAN_Torch.Generator(latent_dim, img_shape)
  torch_model.eval()

  jax_model = model_jax.Generator(img_shape=img_shape)
  key = jax.random.PRNGKey(42)
  dummy_input = jnp.zeros((batch_size, latent_dim))
  params = jax_model.init(key, dummy_input, train=False)

  new_params = flax.core.unfreeze(params)

  def copy_linear(torch_layer, jax_layer_name):
    new_params['params'][jax_layer_name]['kernel'] = (
        torch_layer.weight.detach().numpy().T
    )
    new_params['params'][jax_layer_name][
        'bias'
    ] = torch_layer.bias.detach().numpy()

  def copy_bn(torch_layer, jax_layer_name):
    new_params['params'][jax_layer_name][
        'scale'
    ] = torch_layer.weight.detach().numpy()
    new_params['params'][jax_layer_name][
        'bias'
    ] = torch_layer.bias.detach().numpy()
    new_params['batch_stats'][jax_layer_name][
        'mean'
    ] = torch_layer.running_mean.detach().numpy()
    new_params['batch_stats'][jax_layer_name][
        'var'
    ] = torch_layer.running_var.detach().numpy()

  copy_linear(torch_model.model[0], 'Dense_0')
  copy_linear(torch_model.model[2], 'Dense_1')
  copy_bn(torch_model.model[3], 'BatchNorm_0')
  copy_linear(torch_model.model[5], 'Dense_2')
  copy_bn(torch_model.model[6], 'BatchNorm_1')
  copy_linear(torch_model.model[8], 'Dense_3')

  params = flax.core.freeze(new_params)

  z_np = np.random.randn(batch_size, latent_dim).astype(np.float32)
  z_torch = torch.from_numpy(z_np)
  z_jax = jnp.array(z_np)

  with torch.no_grad():
    out_torch = torch_model(z_torch).numpy()

  out_jax = jax_model.apply(params, z_jax, train=False)

  print(f'Generator Output Max Diff: {np.max(np.abs(out_torch - out_jax))}')
  np.testing.assert_allclose(out_torch, out_jax, rtol=1e-5, atol=1e-5)
  print('Generator Equivalence: PASS')


def test_discriminator_equivalence():
  """Tests equivalence of Discriminator between PyTorch and JAX."""
  img_shape = (1, 28, 28)
  batch_size = 4

  # 1. Initialize PyTorch Model
  torch_model = GAN_Torch.Discriminator(img_shape)
  torch_model.eval()

  # 2. Initialize JAX Model
  jax_model = model_jax.Discriminator()
  key = jax.random.PRNGKey(42)
  dummy_input = jnp.zeros((batch_size, *img_shape))
  params = jax_model.init(key, dummy_input)

  # 3. Transfer Weights
  new_params = flax.core.unfreeze(params)

  def copy_linear(torch_layer, jax_layer_name):
    new_params['params'][jax_layer_name]['kernel'] = (
        torch_layer.weight.detach().numpy().T
    )
    new_params['params'][jax_layer_name][
        'bias'
    ] = torch_layer.bias.detach().numpy()

  copy_linear(torch_model.model[0], 'Dense_0')
  copy_linear(torch_model.model[2], 'Dense_1')
  copy_linear(torch_model.model[4], 'Dense_2')

  params = flax.core.freeze(new_params)

  img_np = np.random.randn(batch_size, *img_shape).astype(np.float32)
  img_torch = torch.from_numpy(img_np)
  img_jax = jnp.array(img_np)

  img_torch_flat = img_torch.view(batch_size, -1)

  with torch.no_grad():
    out_torch = torch_model(img_torch_flat).numpy()

  out_jax = jax_model.apply(params, img_jax)

  print(f'Discriminator Output Max Diff: {np.max(np.abs(out_torch - out_jax))}')
  np.testing.assert_allclose(out_torch, out_jax, rtol=1e-5, atol=1e-5)
  print('Discriminator Equivalence: PASS')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  try:
    test_pytorch_validity()
    pytorch_result = result_pb2.Result(valid=True)
    print('PyTorch Model Shape: VALID (True)')
  except (AssertionError, RuntimeError, TypeError, ValueError) as e:
    error_message = ''.join(traceback.format_exception_only(type(e), e))
    pytorch_result = result_pb2.Result(valid=False, error_message=error_message)
    print(f'PyTorch Model FAILED: {e}')

  try:
    test_jax_validity()
    jax_result = result_pb2.Result(valid=True)
    print('JAX Model Shape: VALID (True)')
  except (AssertionError, RuntimeError, TypeError, ValueError) as e:
    error_message = ''.join(traceback.format_exception_only(type(e), e))
    jax_result = result_pb2.Result(valid=False, error_message=error_message)
    print(f'JAX Model FAILED: {e}')

  try:
    test_generator_equivalence()
    test_discriminator_equivalence()
    equivalence_result = result_pb2.Result(valid=True)
    print('Equivalence Test: VALID (True)')
  except (AssertionError, RuntimeError, TypeError, ValueError) as e:
    error_message = ''.join(traceback.format_exception_only(type(e), e))
    equivalence_result = result_pb2.Result(
        valid=False, error_message=error_message
    )
    print(f'Equivalence Test FAILED: {e}')

  if FLAGS.metrics_output_path:
    eval_result = result_pb2.EvaluationResult(
        pytorch_result=pytorch_result,
        jax_result=jax_result,
        equivalence_result=equivalence_result,
    )
    with open(FLAGS.metrics_output_path, 'w') as f:
      f.write(text_format.MessageToString(eval_result))


if __name__ == '__main__':
  app.run(main)
