---
title: 'Module: easydel/layers/rotary/_utils.py'
type: catalog
provenance: extracted
module: easydel/layers/rotary/_utils.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.layers.rotary._utils`/
symbols:
  yarn_get_mscale: yarn_get_mscale().
  _rotate_neox: _rotate_neox().
  _yarn_find_correction_range: _yarn_find_correction_range().
  _yarn_linear_ramp_mask: _yarn_linear_ramp_mask().
  _apply_rotary_emb: _apply_rotary_emb().
  _rotate_gptj: _rotate_gptj().
  _yarn_find_correction_dim: _yarn_find_correction_dim().
  _yarn_get_mscale: _yarn_get_mscale().
---
# Module: [`easydel/layers/rotary/_utils.py`](../../../../../../../raw/code/EasyDeL/easydel/layers/rotary/_utils.py)

## Functions
- `_apply_rotary_emb(x: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray, is_neox_style: bool)` — [`L203`](../../../../../../../raw/code/EasyDeL/easydel/layers/rotary/_utils.py#L203) — Applies rotary positional embedding to the input tensor.
- `_rotate_gptj(x: Float[Array, "... seq_len head_dim"])` — [`L184`](../../../../../../../raw/code/EasyDeL/easydel/layers/rotary/_utils.py#L184) — Applies the GPT-J-style rotation to the input array.
- `_rotate_neox(x: Float[Array, "... seq_len head_dim"])` — [`L165`](../../../../../../../raw/code/EasyDeL/easydel/layers/rotary/_utils.py#L165) — Applies the Neox-style rotation to the input array.
- `_yarn_find_correction_dim(num_rotations: int, dim: int, base: float = 10000, max_position_embeddings: int = 2048)` — [`L46`](../../../../../../../raw/code/EasyDeL/easydel/layers/rotary/_utils.py#L46) — Calculates the correction dimension for YaRN scaling.
- `_yarn_find_correction_range(low_rot: int, high_rot: int, dim: int, base: float = 10000, max_position_embeddings: int = 2048)` — [`L75`](../../../../../../../raw/code/EasyDeL/easydel/layers/rotary/_utils.py#L75) — Finds the correction range for YaRN scaling based on low and high rotation frequencies.
- `_yarn_get_mscale(scale: float = 1)` — [`L147`](../../../../../../../raw/code/EasyDeL/easydel/layers/rotary/_utils.py#L147) — Calculates the mscale factor for YaRN context extension method.
- `_yarn_linear_ramp_mask(low: float, high: float, dim: int, dtype: jnp.dtype)` — [`L118`](../../../../../../../raw/code/EasyDeL/easydel/layers/rotary/_utils.py#L118) — Creates a linear ramp mask for YaRN scaling.
- `yarn_get_mscale(scale: float = 1, mscale: float = 1)` — [`L246`](../../../../../../../raw/code/EasyDeL/easydel/layers/rotary/_utils.py#L246) — Calculates the mscale factor, potentially used by Deepseek-YaRN or similar methods.

