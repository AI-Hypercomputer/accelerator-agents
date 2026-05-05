"""
TARGETED JAX PATTERN: KV Cache — Pure Functional with Pre-Allocated Buffers

For migration output, use pre-allocated NamedTuple buffers instead of Flax mutable
variables. NamedTuples are framework-agnostic, JIT-safe with static shapes, and
beam-search friendly. Flax's `self.variable('cache', ...)` is the standard Flax API
and works for Flax-only codebases, but couples the conversion to Flax internals.
Do NOT use growing arrays (`jnp.concatenate`) -- they change shape each step and
break jax.jit. Use `dynamic_update_slice` for writes and `dynamic_slice` for reads,
with cache buffers passed as function arguments and returned as outputs.

## WRONG approach 1 (Flax mutable variables -- DO NOT DO THIS):

    # WRONG! Hidden mutable state breaks pure functional JAX semantics
    class Attention(nn.Module):
        @nn.compact
        def __call__(self, x, deterministic=True):
            k = nn.Dense(self.kv_dim)(x)
            v = nn.Dense(self.kv_dim)(x)

            # BAD: Flax mutable variables are hard to manage with jax.jit,
            # beam search, and custom training loops
            cached_key = self.variable('cache', 'cached_key',
                                        jnp.zeros, (batch, max_len, kv_dim))
            cached_key.value = jnp.concatenate([cached_key.value, k], axis=1)

## WRONG approach 2 (growing arrays -- DO NOT DO THIS):

    # WRONG! Concatenation creates new arrays each step, breaking jax.jit
    if cache is not None:
        k = jnp.concatenate([cache['key'], k], axis=1)  # Shape changes each step!
        v = jnp.concatenate([cache['value'], v], axis=1)

## CORRECT approach (pre-allocated buffers + dynamic_update_slice):

    import jax
    import jax.numpy as jnp
    from typing import NamedTuple

    class AttentionCache(NamedTuple):
        '''Pure functional cache for standard attention.'''
        key: jnp.ndarray    # [batch, max_seq_len, num_heads, head_dim]
        value: jnp.ndarray  # [batch, max_seq_len, num_heads, head_dim]
        index: jnp.ndarray  # [] scalar: next write position

    def init_attention_cache(batch_size, max_seq_len, num_heads, head_dim, dtype=jnp.bfloat16):
        '''Create an empty pre-allocated cache.'''
        return AttentionCache(
            key=jnp.zeros((batch_size, max_seq_len, num_heads, head_dim), dtype=dtype),
            value=jnp.zeros((batch_size, max_seq_len, num_heads, head_dim), dtype=dtype),
            index=jnp.array(0, dtype=jnp.int32),
        )

    def update_attention_cache(cache, new_key, new_value):
        '''
        Write new K/V into pre-allocated buffers at the current index.

        Args:
            cache: AttentionCache with pre-allocated buffers
            new_key: [batch, seq_len, num_heads, head_dim] new keys
            new_value: [batch, seq_len, num_heads, head_dim] new values

        Returns:
            updated_cache: AttentionCache with new K/V written in-place
            full_key: [batch, max_seq_len, num_heads, head_dim] (view for attention)
            full_value: [batch, max_seq_len, num_heads, head_dim]
        '''
        seq_len = new_key.shape[1]

        # Write new K/V at current index using dynamic_update_slice
        updated_key = jax.lax.dynamic_update_slice(
            cache.key, new_key,
            (0, cache.index, 0, 0)  # start indices: batch=0, time=index, head=0, dim=0
        )
        updated_value = jax.lax.dynamic_update_slice(
            cache.value, new_value,
            (0, cache.index, 0, 0)
        )

        updated_cache = AttentionCache(
            key=updated_key,
            value=updated_value,
            index=cache.index + seq_len,
        )

        return updated_cache, updated_key, updated_value

    def get_attention_mask(cache_index, new_seq_len, max_seq_len):
        '''
        Build causal mask for cached attention.

        Returns additive mask: 0.0 for allowed positions, -1e9 for blocked.
        '''
        # Positions of new queries: [cache_index, cache_index + new_seq_len)
        q_positions = jnp.arange(new_seq_len) + cache_index
        # Positions of all keys: [0, max_seq_len)
        k_positions = jnp.arange(max_seq_len)

        # Causal: query can attend to keys with position <= query position
        causal_mask = q_positions[:, None] >= k_positions[None, :]
        # Also mask out unfilled positions (beyond cache_index + new_seq_len)
        valid_mask = k_positions[None, :] < (cache_index + new_seq_len)

        mask = causal_mask & valid_mask
        return jnp.where(mask, 0.0, -1e9)

## For GatedDeltaNet linear attention (recurrent state cache):

    class GatedDeltaNetCache(NamedTuple):
        '''Cache for gated delta net linear attention layer.'''
        state: jnp.ndarray       # [batch, num_heads, head_k_dim, head_v_dim] recurrent state
        conv_state: jnp.ndarray  # [batch, channels, kernel_size-1] conv1d rolling state

    def init_gdn_cache(batch_size, num_heads, head_k_dim, head_v_dim,
                       conv_channels, kernel_size, dtype=jnp.bfloat16):
        return GatedDeltaNetCache(
            state=jnp.zeros((batch_size, num_heads, head_k_dim, head_v_dim), dtype=dtype),
            conv_state=jnp.zeros((batch_size, conv_channels, kernel_size - 1), dtype=dtype),
        )

## Full model cache as a NamedTuple of layer caches:

    class ModelCache(NamedTuple):
        '''Cache for the full model -- one entry per layer.'''
        layers: tuple  # tuple of (AttentionCache | GatedDeltaNetCache) per layer

    def init_model_cache(config, batch_size, max_seq_len, dtype=jnp.bfloat16):
        layers = []
        for i in range(config.num_hidden_layers):
            if config.layer_types[i] == 'attention':
                layers.append(init_attention_cache(
                    batch_size, max_seq_len,
                    config.num_attention_heads, config.head_dim, dtype
                ))
            else:
                layers.append(init_gdn_cache(
                    batch_size, config.num_attention_heads,
                    config.head_k_dim, config.head_v_dim,
                    config.hidden_size, config.conv_kernel_size, dtype
                ))
        return ModelCache(layers=tuple(layers))

## Why pure functional cache:

1. **JIT-compatible**: All shapes are static. `dynamic_update_slice` is a traced
   op, not a Python-level mutation.
2. **Pure functional**: Cache is an input and output of the model -- no hidden
   state. Works with `jax.jit`, `jax.vmap`, `jax.pmap`.
3. **Beam search**: Easy to duplicate/reorder caches for beam search by indexing
   into the batch dimension.
4. **No Flax coupling**: NamedTuple cache works with any JAX framework, not just
   Flax. No `self.variable('cache', ...)` magic.
5. **Efficient**: `dynamic_update_slice` is an O(seq_len) in-place XLA op, not
   O(max_seq_len) like concatenation.
"""
