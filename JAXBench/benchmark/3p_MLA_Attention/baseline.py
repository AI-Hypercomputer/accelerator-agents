"""Multi-head Latent Attention (MLA) — DeepSeek V3 671B. Extracted from MaxText."""
import jax
import jax.numpy as jnp
from functools import partial

CONFIG = {
    'name': 'deepseek_v3_mla',
    'model': 'DeepSeek-V3-671B',
    'operator': 'mla_attention',
    'batch': 4,
    'seq_len': 2048,
    'emb_dim': 7168,
    'num_heads': 128,
    'q_lora_rank': 1536,
    'kv_lora_rank': 512,
    'qk_nope_head_dim': 128,
    'qk_rope_head_dim': 64,
    'v_head_dim': 128,
    'rope_theta': 10000,
}


def _compute_rope(head_dim, seq_len, theta, dtype):
    freqs = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    pos = jnp.arange(seq_len, dtype=jnp.float32)
    angles = jnp.outer(pos, freqs)
    return jnp.cos(angles).astype(dtype), jnp.sin(angles).astype(dtype)


def _apply_rope(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    rotated = jnp.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)
    return rotated.reshape(x.shape)


def create_inputs(dtype=jnp.bfloat16):
    """Returns (x, q_down, q_up, kv_down, k_up, v_up, o_proj)."""
    key = jax.random.key(42)
    keys = jax.random.split(key, 8)
    C = CONFIG
    B, S, E = C['batch'], C['seq_len'], C['emb_dim']
    H = C['num_heads']
    ql, kvl = C['q_lora_rank'], C['kv_lora_rank']
    nope, rope, vd = C['qk_nope_head_dim'], C['qk_rope_head_dim'], C['v_head_dim']
    x = jax.random.normal(keys[0], (B, S, E), dtype=dtype)
    q_down = jax.random.normal(keys[1], (E, ql), dtype=dtype) * 0.02
    q_up = jax.random.normal(keys[2], (ql, H * (nope + rope)), dtype=dtype) * 0.02
    kv_down = jax.random.normal(keys[3], (E, kvl + rope), dtype=dtype) * 0.02
    k_up = jax.random.normal(keys[4], (kvl, H * nope), dtype=dtype) * 0.02
    v_up = jax.random.normal(keys[5], (kvl, H * vd), dtype=dtype) * 0.02
    o_proj = jax.random.normal(keys[6], (H * vd, E), dtype=dtype) * 0.02
    return x, q_down, q_up, kv_down, k_up, v_up, o_proj


def workload(x, q_down_proj, q_up_proj, kv_down_proj, k_up_proj, v_up_proj, o_proj):
    """MLA: low-rank KV compression with separated position/content attention."""
    C = CONFIG
    B, S, E = x.shape
    H = C['num_heads']
    nope, rope, vd = C['qk_nope_head_dim'], C['qk_rope_head_dim'], C['v_head_dim']
    kvl = C['kv_lora_rank']
    # Query
    q = jnp.dot(jnp.dot(x, q_down_proj), q_up_proj)
    q = q.reshape(B, S, H, nope + rope)
    q_nope, q_rope = q[..., :nope], q[..., nope:]
    # KV compression
    kv = jnp.dot(x, kv_down_proj)
    k_latent, k_rope_raw = kv[..., :kvl], kv[..., kvl:]
    k_nope = jnp.dot(k_latent, k_up_proj).reshape(B, S, H, nope)
    # RoPE
    cos, sin = _compute_rope(rope, S, C['rope_theta'], x.dtype)
    k_rope = jnp.broadcast_to(k_rope_raw[:, :, None, :], (B, S, H, rope))
    q_rope = _apply_rope(q_rope, cos, sin)
    k_rope = _apply_rope(k_rope, cos, sin)
    # Value
    v = jnp.dot(k_latent, v_up_proj).reshape(B, S, H, vd)
    # Attention
    q_full = jnp.concatenate([q_nope, q_rope], axis=-1).transpose(0, 2, 1, 3)
    k_full = jnp.concatenate([k_nope, k_rope], axis=-1).transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)
    hd = nope + rope
    attn = jnp.einsum('bhqd,bhkd->bhqk', q_full, k_full) * (hd ** -0.5)
    mask = jnp.tril(jnp.ones((S, S)))
    attn = jnp.where(mask, attn, -1e9)
    attn = jax.nn.softmax(attn, axis=-1)
    out = jnp.einsum('bhqk,bhkd->bhqd', attn, v)
    out = out.transpose(0, 2, 1, 3).reshape(B, S, H * vd)
    return jnp.dot(out, o_proj)


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
    C = CONFIG
    B, S, E, H = C['batch'], C['seq_len'], C['emb_dim'], C['num_heads']
    ql, kvl = C['q_lora_rank'], C['kv_lora_rank']
    hd = C['qk_nope_head_dim'] + C['qk_rope_head_dim']
    proj_flops = (
        B * S * E * ql * 2 +
        B * S * ql * (H * hd) * 2 +
        B * S * E * (kvl + C['qk_rope_head_dim']) * 2 +
        B * S * kvl * (H * C['qk_nope_head_dim']) * 2 +
        B * S * kvl * (H * C['v_head_dim']) * 2 +
        B * S * (H * C['v_head_dim']) * E * 2
    )
    attn_flops = B * H * S * S * hd * 4
    flops = proj_flops + attn_flops
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
