"""Sharding plan for HuggingFace Qwen3 on TPU under torchax + JAX.

Wildcard-by-layer-index map keyed on `_process_sharding_name(name)` so all
transformer layers share the same spec. The patterns ('fsdp', 'tp') etc. resolve
to `NamedSharding(mesh, P(...))` against the trainer's mesh
(typically `Mesh((fsdp, tp), ('fsdp', 'tp'))`).

For TP=1 we get pure FSDP (the default on v6e-8). For TP>1 the same map also
encodes Megatron-style tensor parallelism on attention/FFN projections.

Qwen3 vs Llama3 parameter naming — almost identical (GQA + SwiGLU + RMSNorm),
with ONE addition: per-head QK-norm.
- `model.layers.{i}.self_attn.q_norm.weight`  → RMSNorm over head_dim (NEW in
Qwen3)
- `model.layers.{i}.self_attn.k_norm.weight`  → RMSNorm over head_dim (NEW in
Qwen3)
Qwen3 also drops the QKV bias that Qwen2 carried (no `*_proj.bias`).

Everything else mirrors Llama3:
- `model.embed_tokens.weight`
- `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight`
- `model.layers.{i}.mlp.{gate,up,down}_proj.weight`
- `model.layers.{i}.{input_layernorm,post_attention_layernorm}.weight`
- `model.norm.weight`
- `lm_head.weight`  (Qwen3-8B has `tie_word_embeddings=False`, so this is a
  distinct parameter; the smaller Qwen3 sizes tie it to `embed_tokens`.)
"""

import jax
from jax.sharding import NamedSharding, PartitionSpec as P

# FSDP-only spec. For each row, the tuple is the PartitionSpec across mesh axes
# in the canonical mesh order ('fsdp', 'tp'). Empty tuple () = fully replicated.
SHARDING_MAP = {
    "model.embed_tokens.weight": ("fsdp", "tp"),  # (vocab, hidden)
    # Attention QKV/O.
    "model.layers.*.self_attn.q_proj.weight": (
        "tp",
        "fsdp",
    ),  # (num_heads*head_dim, hidden)
    "model.layers.*.self_attn.k_proj.weight": (
        "tp",
        "fsdp",
    ),  # (num_kv_heads*head_dim, hidden)
    "model.layers.*.self_attn.v_proj.weight": (
        "tp",
        "fsdp",
    ),  # (num_kv_heads*head_dim, hidden)
    "model.layers.*.self_attn.o_proj.weight": (
        "fsdp",
        "tp",
    ),  # (hidden, num_heads*head_dim)
    # Qwen3 QK-norm — RMSNorm over head_dim (small 1-D vectors). Replicate.
    "model.layers.*.self_attn.q_norm.weight": (),
    "model.layers.*.self_attn.k_norm.weight": (),
    # FFN (SwiGLU-style: gate, up, down).
    "model.layers.*.mlp.gate_proj.weight": (
        "tp",
        "fsdp",
    ),  # (intermediate, hidden)
    "model.layers.*.mlp.up_proj.weight": ("tp", "fsdp"),
    "model.layers.*.mlp.down_proj.weight": ("fsdp", "tp"),
    # RMSNorm (1-D).
    "model.layers.*.input_layernorm.weight": ("fsdp",),
    "model.layers.*.post_attention_layernorm.weight": ("fsdp",),
    "model.norm.weight": ("fsdp",),
    # LM head.
    "lm_head.weight": ("tp", "fsdp"),
}


def _process_sharding_name(name: str) -> str:
  """Replace integer tokens (layer indices) with `*` for wildcard matching."""

  def is_int(t):
    try:
      int(t)
      return True
    except ValueError:
      return False

  return ".".join("*" if is_int(t) else t for t in name.split("."))


def get_sharding(mesh, param_name: str) -> NamedSharding | None:
  """Return the canonical sharding for a HF Qwen3 parameter, or None if unknown."""
  spec = SHARDING_MAP.get(_process_sharding_name(param_name))
  if spec is None:
    return None
  return NamedSharding(mesh, P(*spec))
