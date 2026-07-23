# pylint: skip-file
# Imports
import jax
import jax.numpy as jnp


# Initialization
def get_inputs(dtype=jnp.bfloat16):
  CONFIG = {
      'name': 'alphafold_768_triangle_mult',
      'model': 'AlphaFold2',
      'operator': 'triangle_mult_outgoing',
      'N': 1536,
      'C': 128,
      'direction': 'outgoing',
  }
  key = jax.random.key(42)
  keys = jax.random.split(key, 9)
  N, C = CONFIG['N'], CONFIG['C']
  pair_act = jax.random.normal(keys[0], (N, N, C), dtype=dtype)
  mask = jnp.ones((N, N, 1), dtype=dtype)
  left_proj = jax.random.normal(keys[1], (C, C), dtype=dtype) * 0.02
  right_proj = jax.random.normal(keys[2], (C, C), dtype=dtype) * 0.02
  left_gate = jax.random.normal(keys[3], (C, C), dtype=dtype) * 0.02
  right_gate = jax.random.normal(keys[4], (C, C), dtype=dtype) * 0.02
  center_scale = jax.random.normal(keys[5], (C,), dtype=dtype) * 0.1 + 1.0
  out_proj = jax.random.normal(keys[6], (C, C), dtype=dtype) * 0.02
  out_gate = jax.random.normal(keys[7], (C, C), dtype=dtype) * 0.02
  dynamic_args = [
      pair_act,
      mask,
      left_proj,
      right_proj,
      left_gate,
      right_gate,
      center_scale,
      out_proj,
      out_gate,
  ]
  static_args = []
  return dynamic_args, static_args


# Computation
def computation(
    pair_act,
    mask,
    left_proj_w,
    right_proj_w,
    left_gate_w,
    right_gate_w,
    center_scale,
    out_proj_w,
    out_gate_w,
):
  act = pair_act * mask
  left_proj = jnp.dot(act, left_proj_w)
  right_proj = jnp.dot(act, right_proj_w)
  left_gate = jax.nn.sigmoid(jnp.dot(act, left_gate_w))
  right_gate = jax.nn.sigmoid(jnp.dot(act, right_gate_w))
  left_proj = left_proj * left_gate
  right_proj = right_proj * right_gate
  result = jnp.einsum('ikc,jkc->ijc', left_proj, right_proj)
  eps = 1e-6
  rms = jnp.sqrt(jnp.mean(result * result, axis=-1, keepdims=True) + eps)
  result = result / rms * center_scale
  output = jnp.dot(result, out_proj_w)
  gate = jax.nn.sigmoid(jnp.dot(pair_act, out_gate_w))
  return output * gate
