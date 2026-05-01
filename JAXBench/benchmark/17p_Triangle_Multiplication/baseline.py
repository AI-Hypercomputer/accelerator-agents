"""Triangle Multiplicative Update (Outgoing) — AlphaFold2 768.

Contracts over the second residue index with gated projections and
layer normalization. Core structural operation in AlphaFold2.
From openxla/tokamax triangle_mult benchmarks.
"""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'alphafold_768_triangle_mult',
    'model': 'AlphaFold2',
    'operator': 'triangle_mult_outgoing',
    'N': 1536,
    'C': 128,
    'direction': 'outgoing',
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (pair_act, mask, left_proj_w, right_proj_w, left_gate_w, right_gate_w, center_norm_scale, output_proj_w, output_gate_w)."""
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
    return pair_act, mask, left_proj, right_proj, left_gate, right_gate, center_scale, out_proj, out_gate


def workload(pair_act, mask, left_proj_w, right_proj_w, left_gate_w, right_gate_w, center_scale, out_proj_w, out_gate_w):
    """Triangle multiplicative update (outgoing): contracts over the second residue index."""
    act = pair_act * mask
    left_proj = jnp.dot(act, left_proj_w)
    right_proj = jnp.dot(act, right_proj_w)
    left_gate = jax.nn.sigmoid(jnp.dot(act, left_gate_w))
    right_gate = jax.nn.sigmoid(jnp.dot(act, right_gate_w))
    left_proj = left_proj * left_gate
    right_proj = right_proj * right_gate
    # Outgoing: contract over k dimension (second index)
    # pair_act[i,j] = sum_k left[i,k] * right[j,k]
    result = jnp.einsum('ikc,jkc->ijc', left_proj, right_proj)
    # Center layer norm (RMS)
    eps = 1e-6
    rms = jnp.sqrt(jnp.mean(result * result, axis=-1, keepdims=True) + eps)
    result = result / rms * center_scale
    # Output projection + gating
    output = jnp.dot(result, out_proj_w)
    gate = jax.nn.sigmoid(jnp.dot(pair_act, out_gate_w))
    return output * gate


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
    N, C = CONFIG['N'], CONFIG['C']
    proj_flops = 4 * N * N * C * C * 2
    einsum_flops = N * N * N * C * 2
    out_flops = 2 * N * N * C * C * 2
    flops = proj_flops + einsum_flops + out_flops
    avg = float(np.mean(times))
    return {
        'name': CONFIG['name'],
        'model': CONFIG['model'],
        'operator': CONFIG['operator'],
        'config': {k: v for k, v in CONFIG.items() if k not in ('name', 'model', 'operator')},
        'time_ms': round(avg, 4),
        'std_ms': round(float(np.std(times)), 4),
        'tflops': round(flops / (avg / 1000) / 1e12, 4),
        'output_shape': list(out.shape),
        'status': 'success',
    }


if __name__ == '__main__':
    import json
    print(json.dumps(benchmark()))
