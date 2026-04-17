# Imports
import jax
import jax.numpy as jnp


# Initialization
def get_inputs():
  CONFIG = {
    "name": "mixtral_8x7b_moe",
    "model": "Mixtral-8x7B",
    "operator": "sparse_moe",
    "batch": 1,
    "seq_len": 2048,
    "emb_dim": 4096,
    "mlp_dim": 14336,
    "num_experts": 8,
    "num_experts_per_tok": 2,
  }
  dtype = jnp.bfloat16
  key = jax.random.PRNGKey(42)
  keys = jax.random.split(key, 5)
  B, S, E, M = (
    CONFIG["batch"],
    CONFIG["seq_len"],
    CONFIG["emb_dim"],
    CONFIG["mlp_dim"],
  )
  N = CONFIG["num_experts"]
  K = CONFIG["num_experts_per_tok"]
  x = jax.random.normal(keys[0], (B, S, E), dtype=dtype)
  router = jax.random.normal(keys[1], (E, N), dtype=dtype) * 0.02
  gate_k = jax.random.normal(keys[2], (N, E, M), dtype=dtype) * 0.02
  up_k = jax.random.normal(keys[3], (N, E, M), dtype=dtype) * 0.02
  down_k_t = jax.random.normal(keys[4], (N, E, M), dtype=dtype) * 0.02
  dynamic_args = [x, router, gate_k, up_k, down_k_t]
  static_args = [K]
  return dynamic_args, static_args


# Computation
def computation(
  x,
  router_weights,
  expert_gate_kernels,
  expert_up_kernels,
  expert_down_kernels_t,
  K,
):
  x = jax.lax.optimization_barrier(x)
  router_weights = jax.lax.optimization_barrier(router_weights)
  expert_gate_kernels = jax.lax.optimization_barrier(expert_gate_kernels)
  expert_up_kernels = jax.lax.optimization_barrier(expert_up_kernels)
  expert_down_kernels_t = jax.lax.optimization_barrier(expert_down_kernels_t)

  expert_down_kernels = expert_down_kernels_t.transpose(0, 2, 1)
  N = router_weights.shape[-1]
  logits = jnp.dot(x, router_weights)
  top_k_logits, top_k_indices = jax.lax.top_k(logits, K)
  router_probs = jax.nn.softmax(top_k_logits, axis=-1)
  gate_out = jax.nn.silu(jnp.einsum("bse,nem->bsnm", x, expert_gate_kernels))
  up_out = jnp.einsum("bse,nem->bsnm", x, expert_up_kernels)
  hidden = gate_out * up_out
  expert_outputs = jnp.einsum("bsnm,nme->bsne", hidden, expert_down_kernels)
  one_hot = jax.nn.one_hot(top_k_indices, N)
  weighted = one_hot * router_probs[..., None]
  expert_weights = weighted.sum(axis=2)
  output = jnp.einsum("bsne,bsn->bse", expert_outputs, expert_weights)
  return output
