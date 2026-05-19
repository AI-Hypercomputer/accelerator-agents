"""Sparse Mixture of Experts (MoE) — Mixtral 8x7B. Extracted from MaxText."""
import jax
import jax.numpy as jnp
from functools import partial

CONFIG = {
    'name': 'mixtral_8x7b_moe',
    'model': 'Mixtral-8x7B',
    'operator': 'sparse_moe',
    'batch': 2,
    'seq_len': 4096,
    'emb_dim': 4096,
    'mlp_dim': 14336,
    'num_experts': 8,
    'num_experts_per_tok': 2,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (x, router_weights, expert_gate, expert_up, expert_down)."""
    key = jax.random.key(42)
    keys = jax.random.split(key, 5)
    B, S, E, M = CONFIG['batch'], CONFIG['seq_len'], CONFIG['emb_dim'], CONFIG['mlp_dim']
    N = CONFIG['num_experts']
    x = jax.random.normal(keys[0], (B, S, E), dtype=dtype)
    router = jax.random.normal(keys[1], (E, N), dtype=dtype) * 0.02
    gate_k = jax.random.normal(keys[2], (N, E, M), dtype=dtype) * 0.02
    up_k = jax.random.normal(keys[3], (N, E, M), dtype=dtype) * 0.02
    down_k = jax.random.normal(keys[4], (N, M, E), dtype=dtype) * 0.02
    return x, router, gate_k, up_k, down_k


def workload(x, router_weights, expert_gate_kernels, expert_up_kernels, expert_down_kernels):
    """Sparse MoE with einsum-based batched expert computation."""
    B, S, E = x.shape
    N = router_weights.shape[-1]
    K = CONFIG['num_experts_per_tok']
    # Routing
    logits = jnp.dot(x, router_weights)
    top_k_logits, top_k_indices = jax.lax.top_k(logits, K)
    router_probs = jax.nn.softmax(top_k_logits, axis=-1)
    # All experts in parallel
    gate_out = jax.nn.silu(jnp.einsum('bse,nem->bsnm', x, expert_gate_kernels))
    up_out = jnp.einsum('bse,nem->bsnm', x, expert_up_kernels)
    hidden = gate_out * up_out
    expert_outputs = jnp.einsum('bsnm,nme->bsne', hidden, expert_down_kernels)
    # Weighted combination
    one_hot = jax.nn.one_hot(top_k_indices, N)
    weighted = one_hot * router_probs[..., None]
    expert_weights = weighted.sum(axis=2)
    output = jnp.einsum('bsne,bsn->bse', expert_outputs, expert_weights)
    return output


def benchmark(num_warmup=5, num_iters=100):
    """Benchmark and return results dict."""
    import time
    inputs = create_inputs()
    fn = jax.jit(workload)
    for _ in range(num_warmup):
        out = fn(*inputs)
        out.block_until_ready()
    times = []
    for _ in range(num_iters):
        t0 = time.perf_counter()
        out = fn(*inputs)
        out.block_until_ready()
        times.append(time.perf_counter() - t0)
    import numpy as np
    times = np.array(times) * 1000
    B, S, E, M = CONFIG['batch'], CONFIG['seq_len'], CONFIG['emb_dim'], CONFIG['mlp_dim']
    K, N = CONFIG['num_experts_per_tok'], CONFIG['num_experts']
    routing_flops = B * S * E * N * 2
    expert_flops = B * S * K * (E * M * 2 * 3)
    flops = routing_flops + expert_flops
    avg = float(np.mean(times))
    return {
        'name': CONFIG['name'],
        'model': CONFIG['model'],
        'operator': CONFIG['operator'],
        'config': {k: v for k, v in CONFIG.items() if k not in ('name', 'model', 'operator')},
        'time_ms': round(avg, 4),
        'std_ms': round(float(np.std(times)), 4),
        'tflops': round(flops / (avg / 1000) / 1e12, 2),
        'output_shape': list(out.shape),
        'status': 'success',
    }


if __name__ == '__main__':
    import json
    print(json.dumps(benchmark()))
