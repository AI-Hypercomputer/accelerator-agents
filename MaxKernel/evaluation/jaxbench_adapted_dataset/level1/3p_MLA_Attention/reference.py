# Imports
import jax
import jax.numpy as jnp
from functools import partial

# Initialization
def get_inputs(dtype=jnp.bfloat16):
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

    dynamic_args = [x, q_down, q_up, kv_down, k_up, v_up, o_proj]
    static_args = [H, nope, rope, vd, kvl, C['rope_theta']]
    return dynamic_args, static_args

# Computation
def _compute_rope(head_dim, seq_len, theta, dtype):
    freqs = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    pos = jnp.arange(seq_len, dtype=jnp.float32)
    angles = jnp.outer(pos, freqs)
    return jnp.cos(angles).astype(dtype), jnp.sin(angles).astype(dtype)

def _apply_rope(x, cos, sin):
    d = x.shape[-1]
    x1, x2 = x[..., :d // 2], x[..., d // 2:]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)

def computation(x, q_down_proj, q_up_proj, kv_down_proj, k_up_proj, v_up_proj, o_proj,
                num_heads, qk_nope_head_dim, qk_rope_head_dim, v_head_dim, kv_lora_rank, rope_theta):
    B, S, E = x.shape
    H = num_heads
    nope = qk_nope_head_dim
    rope = qk_rope_head_dim
    vd = v_head_dim
    kvl = kv_lora_rank
    q = jnp.dot(jnp.dot(x, q_down_proj), q_up_proj)
    q = q.reshape(B, S, H, nope + rope)
    q_nope, q_rope = q[..., :nope], q[..., nope:]
    kv = jnp.dot(x, kv_down_proj)
    k_latent, k_rope_raw = kv[..., :kvl], kv[..., kvl:]
    k_nope = jnp.dot(k_latent, k_up_proj).reshape(B, S, H, nope)
    cos, sin = _compute_rope(rope, S, rope_theta, x.dtype)
    k_rope = jnp.broadcast_to(k_rope_raw[:, :, None, :], (B, S, H, rope))
    q_rope = _apply_rope(q_rope, cos, sin)
    k_rope = _apply_rope(k_rope, cos, sin)
    v = jnp.dot(k_latent, v_up_proj).reshape(B, S, H, vd)
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