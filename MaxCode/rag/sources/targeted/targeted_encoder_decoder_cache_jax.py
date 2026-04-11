"""
TARGETED JAX PATTERN: Encoder-Decoder KV Cache with NamedTuple

When converting encoder-decoder models (e.g., Whisper, T5, BART), the decoder
has TWO types of KV cache:
  1. Self-attention cache: grows with each decode step (like decoder-only models)
  2. Cross-attention cache: computed ONCE from encoder output, reused every step

For migration output, use pure functional NamedTuple caches passed as arguments
and returned as outputs. Flax mutable variables (`self.variable('cache', ...)`)
are Flax's built-in approach but are not recommended for migration output because
they couple the code to Flax's variable management and complicate beam search.
Do NOT use init-flag protocols.

## WRONG approach (Flax mutable variables with init flag -- DO NOT DO THIS):

    class MultiHeadAttention(nn.Module):
        @nn.compact
        def __call__(self, x, xa=None, kv_cache=None):
            if xa is not None and kv_cache is not None:
                cross_k = self.variable('cache', 'cross_k', ...)
                cross_v = self.variable('cache', 'cross_v', ...)
                if kv_cache.get('init', False):   # <-- BAD: init flag protocol
                    k = key_proj(xa)
                    cross_k.value = k              # <-- BAD: mutable state
                else:
                    k = cross_k.value              # <-- BAD: reading mutable state
            # This couples caching logic to the attention module, breaks pure
            # functional JAX semantics, and makes beam search difficult.

## WRONG approach 2 (config dict with no actual caches -- DO NOT DO THIS):

    def install_kv_cache_hooks(self, max_length=448):
        cache_config = {'init': True, 'cache_index': 0, 'max_length': max_length}
        return cache_config, []
        # This returns flags but no pre-allocated cache tensors!
        # PyTorch hooks have no JAX equivalent -- replace with init function.

## CORRECT approach (NamedTuple caches, passed as args, returned as outputs):

    import jax
    import jax.numpy as jnp
    from typing import NamedTuple, Optional, Tuple

    class KVCache(NamedTuple):
        '''Pre-allocated KV cache buffer.'''
        key: jnp.ndarray      # [B, max_len, D]
        value: jnp.ndarray    # [B, max_len, D]
        index: jnp.ndarray    # scalar: next write position

    class MultiHeadAttention(nn.Module):
        n_state: int
        n_head: int

        @nn.compact
        def __call__(self, x, xa=None, mask=None, kv_cache=None):
            q = nn.Dense(self.n_state, name='query')(x)
            source = x if xa is None else xa

            if kv_cache is not None and xa is not None:
                # Cross-attention: K/V already cached from encoder output
                k = kv_cache.key
                v = kv_cache.value
                new_cache = kv_cache  # pass through unchanged
            elif kv_cache is not None:
                # Self-attention: update cache with new K/V
                k_new = nn.Dense(self.n_state, use_bias=False, name='key')(x)
                v_new = nn.Dense(self.n_state, name='value')(x)
                k = jax.lax.dynamic_update_slice(kv_cache.key, k_new, (0, kv_cache.index, 0))
                v = jax.lax.dynamic_update_slice(kv_cache.value, v_new, (0, kv_cache.index, 0))
                new_cache = KVCache(key=k, value=v, index=kv_cache.index + k_new.shape[1])
            else:
                # No cache: compute K/V from source
                k = nn.Dense(self.n_state, use_bias=False, name='key')(source)
                v = nn.Dense(self.n_state, name='value')(source)
                new_cache = None

            out, qk = self._qkv_attention(q, k, v, mask)
            return nn.Dense(self.n_state, name='out')(out), qk, new_cache

    # ResidualAttentionBlock accepts SEPARATE self and cross caches:
    class ResidualAttentionBlock(nn.Module):
        n_state: int
        n_head: int
        cross_attention: bool = False

        @nn.compact
        def __call__(self, x, xa=None, mask=None, self_attn_cache=None, cross_attn_cache=None):
            out, _, new_self_cache = MultiHeadAttention(
                self.n_state, self.n_head, name='attn'
            )(nn.LayerNorm(name='attn_ln')(x), mask=mask, kv_cache=self_attn_cache)
            x = x + out

            new_cross_cache = cross_attn_cache
            if self.cross_attention:
                cross_out, _, new_cross_cache = MultiHeadAttention(
                    self.n_state, self.n_head, name='cross_attn'
                )(nn.LayerNorm(name='cross_attn_ln')(x), xa=xa, kv_cache=cross_attn_cache)
                x = x + cross_out

            # MLP
            h = nn.Dense(self.n_state * 4)(nn.LayerNorm(name='mlp_ln')(x))
            h = jax.nn.gelu(h)
            h = nn.Dense(self.n_state)(h)
            x = x + h

            return x, new_self_cache, new_cross_cache

    # Pre-allocate all caches for decoder layers:
    def init_kv_caches(dims, batch_size, dtype=jnp.float32):
        '''Create pre-allocated KV caches for all decoder layers.'''
        self_caches = tuple(
            KVCache(
                key=jnp.zeros((batch_size, dims.n_text_ctx, dims.n_text_state), dtype=dtype),
                value=jnp.zeros((batch_size, dims.n_text_ctx, dims.n_text_state), dtype=dtype),
                index=jnp.array(0, dtype=jnp.int32),
            )
            for _ in range(dims.n_text_layer)
        )
        # Cross-attention caches: populated once from encoder output
        cross_caches = tuple(
            KVCache(
                key=jnp.zeros((batch_size, dims.n_audio_ctx, dims.n_text_state), dtype=dtype),
                value=jnp.zeros((batch_size, dims.n_audio_ctx, dims.n_text_state), dtype=dtype),
                index=jnp.array(0, dtype=jnp.int32),
            )
            for _ in range(dims.n_text_layer)
        )
        return self_caches, cross_caches

## WHY this pattern is correct:

1. **Pure functional**: Caches are inputs AND outputs. No hidden mutable state.
2. **Cross-attention reuse**: Encoder K/V computed once, stored in cross_attn_cache,
   passed through unchanged on every decode step. No init flag needed.
3. **JIT-safe**: All shapes static. dynamic_update_slice is traced, not Python mutation.
4. **Beam search**: Easy to duplicate/reorder NamedTuple caches by batch indexing.
5. **Replaces install_kv_cache_hooks**: PyTorch uses hooks to intercept projections.
   JAX replaces this with init_kv_caches() that pre-allocates all layer caches.
"""
