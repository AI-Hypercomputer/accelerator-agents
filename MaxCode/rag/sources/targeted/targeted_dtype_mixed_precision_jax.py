"""
TARGETED JAX PATTERN: dtype and Mixed Precision on TPU/GPU

When converting PyTorch models to JAX, handle dtype carefully. TPU bfloat16 has
different precision characteristics than GPU float16, and certain operations
MUST be done in float32 for numerical stability.

## Operations that MUST use float32:

| Operation              | Why float32 is needed                              |
|------------------------|----------------------------------------------------|
| Softmax                | exp() overflows in bf16; sum of probs loses precision |
| Variance / RMS         | Squaring amplifies error; mean of squares needs range |
| Layer/RMS normalization| Uses variance internally                            |
| Loss computation       | Cross-entropy log() needs precision                 |
| Cumulative sum/prod    | Accumulation amplifies rounding errors              |
| Router logits (MoE)    | Small differences in routing matter                 |

## Pattern: Upcast before, cast back after

    import jax.numpy as jnp

    def stable_softmax(x, axis=-1):
        '''Softmax with float32 upcast for numerical stability.'''
        x_f32 = x.astype(jnp.float32)
        result = jax.nn.softmax(x_f32, axis=axis)
        return result.astype(x.dtype)

    def rms_norm(x, weight, eps=1e-6):
        '''RMS normalization with float32 upcast.'''
        orig_dtype = x.dtype
        x = x.astype(jnp.float32)
        rms = jax.lax.rsqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + eps)
        return (x * rms).astype(orig_dtype) * weight

## Flax param_dtype vs compute dtype:

    import flax.linen as nn

    class MyDense(nn.Module):
        features: int
        param_dtype: jnp.dtype = jnp.bfloat16   # Store weights in bf16
        compute_dtype: jnp.dtype = jnp.bfloat16  # Compute in bf16

        @nn.compact
        def __call__(self, x):
            kernel = self.param(
                'kernel',
                nn.initializers.normal(stddev=0.02),
                (x.shape[-1], self.features),
                self.param_dtype,  # Weight stored in this dtype
            )
            # Cast to compute dtype for matmul
            x = x.astype(self.compute_dtype)
            kernel = kernel.astype(self.compute_dtype)
            return x @ kernel

## TPU bfloat16 gotchas:

1. **No float16 on TPU**: TPU natively supports bf16 and f32. Using float16
   requires emulation and is slower. Always use bfloat16 on TPU.

2. **bf16 range vs precision**: bf16 has same exponent range as f32 (no overflow
   for typical values) but only 7 bits of mantissa (vs 23 for f32). This means
   additions of values with different magnitudes lose precision.

3. **Matmul accumulation**: `jnp.matmul` on TPU accumulates in float32 internally
   even with bf16 inputs, so matmuls are generally safe. But element-wise ops
   (add, multiply, square) do NOT auto-upcast.

4. **jnp.where dtype**: `jnp.where(cond, 0.0, -1e9)` -- the -1e9 must fit in
   the output dtype. For bf16, -1e9 is representable. For fp16, use
   `jnp.finfo(dtype).min` instead of a literal.

## Full pattern in a transformer layer:

    class TransformerLayer(nn.Module):
        config: ModelConfig

        @nn.compact
        def __call__(self, x):
            dtype = self.config.compute_dtype  # e.g., jnp.bfloat16

            # RMSNorm: upcast to f32 internally
            normed = rms_norm(x, self.param('norm', nn.initializers.ones_init(),
                              (self.config.hidden_size,)))

            # Attention: matmuls are safe in bf16
            q = nn.Dense(self.config.qk_dim, dtype=dtype)(normed)
            k = nn.Dense(self.config.qk_dim, dtype=dtype)(normed)
            v = nn.Dense(self.config.v_dim, dtype=dtype)(normed)

            # Attention scores: safe in bf16 (matmul accumulates in f32)
            attn = q @ k.swapaxes(-2, -1) / jnp.sqrt(self.config.head_dim)

            # Softmax: MUST upcast to f32
            attn = stable_softmax(attn)

            out = attn @ v
            return x + nn.Dense(self.config.hidden_size, dtype=dtype)(out)
"""
