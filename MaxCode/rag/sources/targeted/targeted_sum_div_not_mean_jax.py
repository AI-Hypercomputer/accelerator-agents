"""
TARGETED JAX PATTERN: Preserve .sum() / divisor — Do Not Replace with .mean()

CRITICAL: When PyTorch source computes `.sum(dim=N) / some_constant`, the JAX
conversion must use `jnp.sum(x, axis=N) / some_constant` — NOT `.mean(axis=N)`.
These are only equivalent when the dimension size equals the constant, which is
not guaranteed.

## WRONG: Replacing .sum(dim=1) / num_heads with .mean(axis=1)

    # PyTorch source:
    #   attn_output = attn_weights.sum(dim=1) / self.num_heads

    # WRONG! .mean(axis=1) divides by the dimension size (dim_size),
    # but the source divides by num_heads. These differ when dim_size != num_heads.
    attn_output = jnp.mean(attn_weights, axis=1)

## WRONG: Replacing .sum(dim=-1) / divisor with .mean(axis=-1)

    # PyTorch source:
    #   normalized = scores.sum(dim=-1) / temperature

    # WRONG! .mean(axis=-1) divides by the last dimension size,
    # but the source divides by temperature (a scalar parameter).
    normalized = jnp.mean(scores, axis=-1)

## CORRECT: Preserve .sum() / constant exactly

    # PyTorch source:
    #   attn_output = attn_weights.sum(dim=1) / self.num_heads

    # CORRECT: Faithful translation — sum then divide by the same constant.
    attn_output = jnp.sum(attn_weights, axis=1) / self.num_heads

## CORRECT: Preserve .sum() / scalar parameter

    # PyTorch source:
    #   normalized = scores.sum(dim=-1) / temperature

    # CORRECT: Same reduction and same divisor.
    normalized = jnp.sum(scores, axis=-1) / temperature

## CORRECT: Use .mean() ONLY when the source uses .mean()

    # PyTorch source:
    #   avg_pool = features.mean(dim=1)

    # CORRECT: Source uses .mean(), so JAX uses .mean().
    avg_pool = jnp.mean(features, axis=1)

## Why this matters:

1. **Different denominators**: `.mean(axis=N)` divides by `x.shape[N]` (the
   dimension size). `.sum(axis=N) / C` divides by a constant C. These produce
   different results whenever `x.shape[N] != C`.
2. **Concrete example**: If `attn_weights` has shape `(batch, 8, seq, seq)` and
   `num_heads = 4`, then `.mean(axis=1)` divides by 8, but `.sum(axis=1) / 4`
   divides by 4 — the result is off by a factor of 2.
3. **Numerical equivalence is not guaranteed**: Even when the dimension happens
   to equal the constant for one model config, a different config (different
   num_heads, different seq_len) may break the equivalence.
4. **Faithfulness principle**: The conversion must preserve the source's exact
   arithmetic. If the source says "sum then divide by N", write "sum then divide
   by N" — do not simplify to "mean".
5. **Rule of thumb**: Only use `.mean()` in JAX when the PyTorch source uses
   `.mean()`. For `.sum() / constant`, always write `jnp.sum(...) / constant`.
"""
