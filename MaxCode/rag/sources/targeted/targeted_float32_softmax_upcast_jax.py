"""
TARGETED RAG: Float32 Softmax Upcast in JAX/Flax
==================================================

When converting attention code that uses `.float()` before softmax in PyTorch,
you MUST preserve the float32 upcast in JAX. This is critical for numerical
stability when the model runs in bfloat16 or float16.

WRONG -- No upcast before softmax:
------------------------------------
    attn_weights = jnp.matmul(q, k.transpose(0, 2, 1)) * scale
    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask
    attn_weights = jax.nn.softmax(attn_weights, axis=-1)  # WRONG: no upcast
    attn_probs = nn.Dropout(rate=self.attn_dropout)(
        attn_weights, deterministic=self.deterministic)

WHY THIS IS WRONG:
- In bfloat16, the exp() inside softmax can overflow or underflow
- PyTorch code explicitly does `attn_weights_float = attn_weights.float()`
  before softmax, then casts back with `.type_as(attn_weights)`
- Without the upcast, attention distributions become inaccurate, especially
  for long sequences where values can be very negative
- This causes subtle numerical errors that compound through layers

CORRECT -- Upcast to float32 before softmax, cast back after:
--------------------------------------------------------------
    attn_weights = jnp.matmul(q, k.transpose(0, 2, 1)) * scale
    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask
    # CORRECT: upcast to float32 before softmax for numerical stability
    attn_weights = jax.nn.softmax(attn_weights.astype(jnp.float32), axis=-1)
    attn_weights = attn_weights.astype(q.dtype)  # cast back to compute dtype
    attn_probs = nn.Dropout(rate=self.attn_dropout)(
        attn_weights, deterministic=self.deterministic)

PATTERN MATCHING:
-----------------
When you see ANY of these patterns in PyTorch source code, add the float32 upcast:

  PyTorch pattern 1: `attn_weights_float = attn_weights.float()`
  PyTorch pattern 2: `attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)`
  PyTorch pattern 3: `attn_weights.float().softmax(dim=-1).type_as(attn_weights)`

JAX equivalent for ALL of these:
  ```
  attn_weights = jax.nn.softmax(attn_weights.astype(jnp.float32), axis=-1)
  attn_weights = attn_weights.astype(q.dtype)
  ```

OTHER OPERATIONS THAT NEED FLOAT32 UPCAST:
-------------------------------------------
The same principle applies to:

1. Layer normalization variance:
   WRONG:  variance = jnp.mean(x ** 2, axis=-1, keepdims=True)
   CORRECT: variance = jnp.mean(x.astype(jnp.float32) ** 2, axis=-1, keepdims=True)

2. Loss functions with log:
   WRONG:  loss = -jnp.log(probs)
   CORRECT: loss = -jnp.log(probs.astype(jnp.float32))

3. Any operation with exp(), log(), or division where precision matters.

RULE: When in doubt, upcast to float32. The cost is negligible (XLA fuses the
cast with the computation) but the benefit is correct numerics.
"""
