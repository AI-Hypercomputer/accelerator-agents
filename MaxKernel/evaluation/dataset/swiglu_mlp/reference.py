# Imports
import jax
import jax.numpy as jnp


# Initialization
def get_inputs():
  CONFIG = {
    "name": "llama3_70b_swiglu",
    "model": "Llama-3.1-70B",
    "operator": "swiglu_mlp",
    "batch": 1,
    "seq_len": 2048,
    "emb_dim": 8192,
    "mlp_dim": 28672,
  }
  dtype = jnp.bfloat16
  key = jax.random.PRNGKey(42)
  k1, k2, k3, k4 = jax.random.split(key, 4)
  B, S, E, M = (
    CONFIG["batch"],
    CONFIG["seq_len"],
    CONFIG["emb_dim"],
    CONFIG["mlp_dim"],
  )
  x = jax.random.normal(k1, (B, S, E), dtype=dtype)
  gate = jax.random.normal(k2, (E, M), dtype=dtype) * 0.02
  up = jax.random.normal(k3, (E, M), dtype=dtype) * 0.02
  down = jax.random.normal(k4, (M, E), dtype=dtype) * 0.02
  dynamic_args = [x, gate, up, down]
  static_args = []
  return dynamic_args, static_args


# Computation
def computation(x, gate_kernel, up_kernel, down_kernel):
  gate = jax.nn.silu(jnp.dot(x, gate_kernel))
  up = jnp.dot(x, up_kernel)
  return jnp.dot(gate * up, down_kernel)
