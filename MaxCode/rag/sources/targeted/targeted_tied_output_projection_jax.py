"""
TARGETED JAX PATTERN: Tied Output Projection (Weight Tying)

When the PyTorch source uses explicit `x @ weight.T` for output projection,
the JAX conversion must use explicit matmul, not `.attend()`. Flax's
`nn.Embed.attend()` and framework-specific attend() methods (e.g., MaxText's
`Embed.attend()`) may internally match the matmul behavior, but explicit
`x @ embedding.T` guarantees numerical equivalence with the PyTorch source.

## WRONG approach (attend() -- DO NOT DO THIS):

    # WRONG! attend() is for embedding lookup, not linear projection
    token_embedding = nn.Embed(n_vocab, n_state, name='token_embedding')
    x_emb = token_embedding(tokens)
    # ... transformer layers ...
    logits = token_embedding.attend(x_out)  # <-- WRONG: may not match PyTorch

    # nn.Embed.attend() computes a dot product for attention-style lookup.
    # It may apply different scaling or normalization than a simple matmul.
    # The PyTorch source does `x @ weight.T` which is a plain linear projection.

## CORRECT approach (explicit matmul with embedding table):

    token_embedding = nn.Embed(n_vocab, n_state, name='token_embedding')
    x_emb = token_embedding(tokens)
    # ... transformer layers ...
    # Tied output projection: multiply by transpose of embedding table
    logits = (x_out @ token_embedding.embedding.T).astype(jnp.float32)

    # `token_embedding.embedding` is the [n_vocab, n_state] weight matrix.
    # `.T` transposes it to [n_state, n_vocab].
    # The matmul gives [B, T, n_vocab] logits -- exactly like PyTorch.

## WHY this matters:

1. **Faithfulness**: PyTorch `x @ weight.T` is a plain matrix multiplication.
   Using `token_embedding.embedding.T` in Flax does the exact same operation.
2. **Weight loading**: When loading PyTorch weights, the embedding weight is
   shared between input embedding and output projection. Using explicit matmul
   ensures the same weight is used for both, matching PyTorch exactly.
3. **Numerical equivalence**: `.attend()` may apply internal transformations
   that produce different logits than the simple transpose+matmul.
4. **Float32 cast**: Apply `.astype(jnp.float32)` after the matmul to match
   PyTorch's `.float()` call on the logits.
"""
