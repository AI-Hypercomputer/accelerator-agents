"""
TARGETED JAX PATTERN: Weight Initialization — PyTorch to Flax Mapping

CRITICAL: Weight initialization must match the PyTorch source EXACTLY. Wrong init
breaks routing, norms, and weight loading from PyTorch checkpoints. Each layer type
has a specific initializer -- do NOT use a single default for everything.

## PyTorch to Flax Initializer Mapping Table:

This table applies to models with `_init_weights` methods (e.g., HuggingFace-style).
When no `_init_weights` exists and the source uses bare `nn.Linear`, use the Flax
default (`lecun_normal`) as the closest match to PyTorch's default Kaiming uniform.

| PyTorch Layer / Init              | Flax Initializer                                         |
|-----------------------------------|----------------------------------------------------------|
| nn.Linear (general Dense)         | nn.initializers.normal(stddev=config.initializer_range)  |
| nn.Embedding                      | nn.initializers.normal(stddev=1.0)                       |
| MoE Router / Gate                 | nn.initializers.zeros_init() (when source explicitly zero-inits) |
| RMSNorm weight (1 + w formulation)| nn.initializers.zeros_init()                             |
| RMSNorm weight (w formulation)    | nn.initializers.ones_init()                              |
| LayerNorm weight                  | nn.initializers.ones_init()                              |
| LayerNorm bias                    | nn.initializers.zeros_init()                             |
| Log-decay / log-tau parameters    | Custom log_uniform_init or specific range                |
| Conv1d weight (depthwise)         | nn.initializers.normal(stddev=config.initializer_range)  |
| Bias (general)                    | nn.initializers.zeros_init()                             |

## WRONG: Using default or wrong init for router

    # WRONG! Normal init causes non-uniform routing from step 0
    class MoERouter(nn.Module):
        num_experts: int

        @nn.compact
        def __call__(self, x):
            return nn.Dense(self.num_experts)(x)  # Default normal init!

## CORRECT: Zero-init for router

    class MoERouter(nn.Module):
        num_experts: int

        @nn.compact
        def __call__(self, x):
            return nn.Dense(
                self.num_experts,
                kernel_init=nn.initializers.zeros_init(),
                use_bias=False,
            )(x)

## WRONG: Using ones_init for RMSNorm when source uses (1 + w) formulation

    # If PyTorch source initializes RMSNorm weight to zeros and computes:
    #   output = x * rsqrt(mean(x^2) + eps) * (1 + self.weight)
    # Then weight starts at 0, making the initial scale factor = 1.

    # WRONG! ones_init means initial scale = 1 + 1 = 2
    weight = self.param('scale', nn.initializers.ones_init(), (dim,))
    return normed * (1 + weight)

## CORRECT: Match the source formulation

    # If source uses (1 + w) with w initialized to zeros:
    weight = self.param('scale', nn.initializers.zeros_init(), (dim,))
    return normed * (1 + weight)

    # If source uses plain w with w initialized to ones:
    weight = self.param('scale', nn.initializers.ones_init(), (dim,))
    return normed * weight

## Dense layer initialization:

    # General Dense projection -- match config.initializer_range (typically 0.02)
    nn.Dense(
        features,
        kernel_init=nn.initializers.normal(stddev=config.initializer_range),
        use_bias=config.use_bias,
    )

## Embedding initialization:

    nn.Embed(
        num_embeddings=config.vocab_size,
        features=config.hidden_size,
        embedding_init=nn.initializers.normal(stddev=1.0),
    )

## Custom log-uniform initializer for decay/tau parameters:

    import jax
    import jax.numpy as jnp

    def log_uniform_init(min_val, max_val):
        '''Initialize in log-space uniformly between min_val and max_val.'''
        def init(key, shape, dtype=jnp.float32):
            log_min = jnp.log(jnp.array(min_val, dtype=dtype))
            log_max = jnp.log(jnp.array(max_val, dtype=dtype))
            return jax.random.uniform(key, shape, dtype=dtype,
                                      minval=log_min, maxval=log_max)
        return init

    # Usage for log-decay parameters:
    log_decay = self.param('log_decay', log_uniform_init(1.0, 16.0), (num_heads,))
    decay = jnp.exp(-jnp.exp(log_decay))

## Additional notes:

Note: RMSNorm epsilon defaults vary by model (1e-6 in Flax, 1e-5 in FLA/PyTorch).
Always match the source model's epsilon value.

Note: Flax names norm weights 'scale'; PyTorch uses 'weight'. Checkpoint loading
must handle this mapping (e.g., rename 'weight' -> 'scale' when loading PyTorch
weights into Flax).

## Why initialization matters:

1. **Router zeros**: Ensures uniform expert selection at initialization. Normal init
   creates random biases that can cause expert collapse (some experts never chosen).
2. **RMSNorm**: Wrong init changes the effective scale factor, which means loaded
   PyTorch weights will produce different outputs.
3. **Dense layers**: stddev=0.02 matches the default PyTorch nn.Linear init for
   transformer models (config.initializer_range).
4. **Weight loading**: When loading PyTorch checkpoints, the Flax model's init
   doesn't matter for loaded weights. But for any randomly-initialized weights
   (e.g., during pretraining), matching init is essential for convergence.
"""
