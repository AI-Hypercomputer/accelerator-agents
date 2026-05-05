# Flax Linen Layers API Reference
# Source: https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/layers.html
"""
Flax Linen Layers API Reference
================================

Linear Modules
--------------

Dense(features, use_bias=True, dtype=None, param_dtype=float32,
      kernel_init=variance_scaling, bias_init=zeros)

    A linear transformation applied over the last dimension of the input.

    layer = nn.Dense(features=4)
    params = layer.init(jax.random.key(0), jnp.ones((1, 3)))
    output = layer.apply(params, x)  # x: [..., in_features] -> [..., 4]

DenseGeneral(features, axis=-1, batch_dims=(), use_bias=True, dtype=None,
             kernel_init=variance_scaling, bias_init=zeros)

    A linear transformation with flexible axes. Can contract over multiple axes.

    # Contract over axes 1 and -1, output features (4, 5)
    layer = nn.DenseGeneral(features=(4, 5), axis=(1, -1))
    params = layer.init(jax.random.key(0), jnp.ones((1, 3, 6, 7)))

Conv(features, kernel_size, strides=1, padding='SAME', input_dilation=1,
     kernel_dilation=1, feature_group_count=1, use_bias=True, dtype=None)

    Convolution layer wrapping lax.conv_general_dilated.

    # 1D convolution
    layer = nn.Conv(features=4, kernel_size=(3,), padding='VALID')
    out, variables = layer.init_with_output(jax.random.key(0), jnp.ones((1, 8, 3)))

    # Causal 1D convolution (pad left only)
    layer = nn.Conv(features=4, kernel_size=(3,), padding=((2, 0),))

Embedding Module
-----------------

Embed(num_embeddings, features, dtype=None, param_dtype=float32,
      embedding_init=variance_scaling)

    A parameterized function from integers [0, num_embeddings) to features-dimensional vectors.

    layer = nn.Embed(num_embeddings=50000, features=768)
    variables = layer.init(jax.random.key(0), jnp.array([[0, 1, 2]]))
    embeddings = layer.apply(variables, input_ids)  # [batch, seq_len, features]

    # attend() method for output projection (weight tying):
    logits = layer.attend(hidden_states)  # [batch, seq_len, num_embeddings]
    # Note: For exact PyTorch weight-tying equivalence, prefer explicit matmul: x @ embed.embedding.T

Normalization Layers
---------------------

LayerNorm(epsilon=1e-6, dtype=None, use_bias=True, use_scale=True,
          reduction_axes=-1, feature_axes=-1)

    Layer normalization. Normalizes over the last axis by default.

    norm = nn.LayerNorm()
    variables = norm.init(jax.random.key(0), x)
    y = norm.apply(variables, x)

RMSNorm(epsilon=1e-6, dtype=None, use_scale=True, scale_init=ones,
        reduction_axes=-1, feature_axes=-1)

    RMS Layer normalization. Normalizes by root mean square without re-centering.
    More efficient than LayerNorm as it skips the mean computation.

    norm = nn.RMSNorm()
    variables = norm.init(jax.random.key(0), x)
    y = norm.apply(variables, x)

    # Custom implementation pattern (common in LLMs):
    class CustomRMSNorm(nn.Module):
      dim: int
      eps: float = 1e-6

      @nn.compact
      def __call__(self, x):
        weight = self.param('weight', nn.initializers.ones, (self.dim,))
        variance = jnp.mean(x ** 2, axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(variance + self.eps)
        return weight * x

GroupNorm(num_groups=32, epsilon=1e-6, use_bias=True, use_scale=True)

    Group normalization. Statistics shared across equally-sized groups of channels.

Attention Modules
------------------

MultiHeadDotProductAttention(num_heads, dtype=None, qkv_features=None,
    out_features=None, dropout_rate=0.0, deterministic=None,
    kernel_init=variance_scaling, use_bias=True,
    attention_fn=dot_product_attention, decode=False, normalize_qk=False)

    Multi-head dot-product attention mechanism.

    layer = nn.MultiHeadDotProductAttention(num_heads=8, qkv_features=64)

    # Self-attention
    variables = layer.init(jax.random.key(0), x)
    out = layer.apply(variables, x)

    # Cross-attention
    out = layer.apply(variables, query, key, value)

    # With causal mask
    mask = nn.make_causal_mask(jnp.ones((batch, seq_len)))
    out = layer.apply(variables, x, mask=mask, deterministic=True)

    # Autoregressive decoding with KV cache
    layer = nn.MultiHeadDotProductAttention(num_heads=8, decode=True)
    variables = layer.init(jax.random.key(0), x)
    # variables['cache'] contains cached keys and values
    # Note: For PyTorch->JAX migrations, prefer pre-allocated NamedTuple buffers
    # over Flax's decode=True mutable cache (see targeted_kvcache_prefill_decode_jax.py)

    Key parameters:
    - decode=True: enables autoregressive KV caching
    - normalize_qk=True: applies QK normalization
    - deterministic=True: disables dropout

Mask Utilities
---------------

make_causal_mask(x, extra_batch_dims=0, dtype=bool)
    Creates a causal attention mask from input shape.

    mask = nn.make_causal_mask(jnp.ones((1, seq_len)))
    # Returns [1, 1, seq_len, seq_len] boolean mask

make_attention_mask(query_input, key_input, pairwise_fn=jnp.multiply,
                    extra_batch_dims=0, dtype=bool)
    Creates an attention mask from query and key padding masks.

    query_mask = jnp.array([1, 1, 1, 0])  # 1=valid, 0=padded
    key_mask = jnp.array([1, 1, 0, 0])
    mask = nn.make_attention_mask(query_mask, key_mask)

Activation Functions
---------------------
nn.relu, nn.gelu, nn.silu (swish), nn.softmax, nn.tanh, nn.sigmoid, nn.elu

    x = nn.silu(x)  # SiLU/Swish activation, common in modern LLMs
    x = nn.gelu(x, approximate=False)

Pooling Functions
------------------
nn.max_pool(inputs, window_shape, strides=None, padding='VALID')
nn.avg_pool(inputs, window_shape, strides=None, padding='VALID')
"""
