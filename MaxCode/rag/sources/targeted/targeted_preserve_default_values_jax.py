"""
TARGETED JAX PATTERN: Preserve Default Parameter Values Exactly

CRITICAL: When converting PyTorch to JAX, default parameter values must match
the source EXACTLY. Do not change defaults, even if you think a different value
is "better". Changed defaults silently alter model behavior and break
reproducibility between PyTorch and JAX versions.

## WRONG: Changing default values during conversion

    # PyTorch source:
    #   class Router(nn.Module):
    #       def __init__(self, input_dim, num_experts, k=1, capacity_factor=1.0):
    #           ...

    # WRONG! Changed capacity_factor from 1.0 to 1.25
    class Router(nn.Module):
        config: MoEConfig  # where MoEConfig has capacity_factor: float = 1.25

    # WRONG! Changed dropout from 0.1 to 0.0
    class FFNExpert(nn.Module):
        dropout_rate: float = 0.0  # Source default is 0.1!

    # WRONG! Changed noise_epsilon from 1e-2 to 1e-3
    class Router(nn.Module):
        noise_epsilon: float = 1e-3  # Source default is 1e-2!

## CORRECT: Match source defaults exactly

    # PyTorch source:
    #   class Router(nn.Module):
    #       def __init__(self, input_dim, num_experts, k=1, capacity_factor=1.0):

    # CORRECT: All defaults match source
    class Router(nn.Module):
        input_dim: int
        num_experts: int
        k: int = 1
        capacity_factor: float = 1.0  # Matches source exactly

    # CORRECT: If using a config dataclass, defaults must also match
    @dataclasses.dataclass
    class MoEConfig:
        input_dim: int
        output_dim: int
        num_experts: int
        k: int = 1
        capacity_factor: float = 1.0     # Must match source Router default
        noise_epsilon: float = 1e-2       # Must match source Router default
        dropout_rate: float = 0.1         # Must match source FFNExpert default
        num_layers: int = 2               # Must match source FFNExpert default

## WRONG: Changing weight initialization from PyTorch default

    # PyTorch nn.Linear uses Kaiming uniform by default (not zeros, not normal).
    # When the source uses bare nn.Linear(...) with no explicit init, use the
    # Flax default initializer (lecun_normal), NOT zeros_init.

    # WRONG! Source uses default init, but conversion uses zeros
    router_logits = nn.Dense(
        features=num_experts,
        use_bias=False,
        kernel_init=nn.initializers.zeros_init(),  # NOT what source does!
    )(x)

## CORRECT: Match PyTorch default initialization

    # When PyTorch source uses bare nn.Linear with no custom init:
    router_logits = nn.Dense(
        features=num_experts,
        use_bias=False,
        # Default Flax init (lecun_normal) is acceptable, or use:
        # kernel_init=nn.initializers.normal(stddev=config.initializer_range)
        # DO NOT use zeros_init unless the source explicitly does so.
    )(x)

    # ONLY use zeros_init when the source EXPLICITLY initializes to zeros:
    #   nn.init.zeros_(self.router.weight)  # PyTorch source has this line
    # Then and only then:
    router_logits = nn.Dense(
        features=num_experts,
        kernel_init=nn.initializers.zeros_init(),
    )(x)

## Note on _init_weights and constructor defaults:

When the source's `_init_weights` method explicitly zero-initializes a layer
(e.g., router weights via `nn.init.zeros_`), use `zeros_init()` in the Flax
conversion. This IS matching the source, since `_init_weights` overrides the
constructor default. The rule "match the source default" means match the
EFFECTIVE default after all initialization code runs, not just the bare
constructor signature.

## Why preserving defaults matters:

1. **Reproducibility**: Changed defaults mean the JAX model behaves differently
   from PyTorch even with identical weights and inputs.
2. **Capacity factor**: Changing capacity_factor from 1.0 to 1.25 changes how many
   tokens each expert receives, altering load balancing dynamics.
3. **Dropout rate**: A different default dropout rate changes regularization strength,
   leading to different training outcomes.
4. **Router init**: Zero-initialized router weights produce uniform routing at step 0,
   while Kaiming/lecun_normal produces non-uniform routing. This affects early
   training dynamics and can lead to expert collapse or slower convergence.
5. **Trust the source**: The original author chose specific defaults for a reason.
   The conversion should preserve their intent exactly.
"""
