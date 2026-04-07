"""
TARGETED JAX PATTERN: Source Faithfulness — Do Not "Improve" the Source

CRITICAL: The goal of PyTorch-to-JAX conversion is a FAITHFUL TRANSLATION, not
a redesign or optimization. The converted code must produce identical behavior to
the source for the same inputs and weights. Never change defaults, initializers,
reduction operations, or function semantics — even if you believe a different
choice is "better", "more stable", or "more efficient".

## Principle 1: Preserve Exact Initializer Semantics

WRONG: Adding an explicit initializer when the source uses the framework default.

    # PyTorch source (uses default Kaiming uniform init):
    #   self.router = nn.Linear(input_dim, num_experts, bias=False)

    # WRONG! Source does NOT explicitly initialize to zeros.
    # Adding zeros_init changes the model's behavior at initialization.
    router_logits = nn.Dense(
        features=num_experts,
        use_bias=False,
        kernel_init=nn.initializers.zeros_init(),  # NOT in source!
    )(x)

CORRECT: Use the Flax default init (lecun_normal) to match "bare nn.Linear".

    # CORRECT: No explicit kernel_init => Flax default (lecun_normal),
    # which is the closest match to PyTorch's default Kaiming uniform.
    router_logits = nn.Dense(
        features=num_experts,
        use_bias=False,
    )(x)

    # ONLY use a custom initializer when the PyTorch source EXPLICITLY sets one:
    #   nn.init.zeros_(self.router.weight)      => kernel_init=nn.initializers.zeros_init()
    #   nn.init.normal_(self.fc.weight, std=0.02) => kernel_init=nn.initializers.normal(stddev=0.02)
    #   nn.init.xavier_uniform_(self.fc.weight)   => kernel_init=nn.initializers.xavier_uniform()


## Principle 2: Preserve Exact Default Parameter Values

WRONG: Changing numeric defaults because you think a different value is better.

    # PyTorch source:
    #   def __init__(self, ..., capacity_factor=1.0, noise_epsilon=1e-2):

    # WRONG! Changed capacity_factor. The comment does NOT justify this.
    @dataclass
    class Config:
        capacity_factor: float = 1.25  # "Increased for stability"
        # This silently changes model behavior!

CORRECT: Copy every default value exactly from the source.

    # CORRECT: All defaults match source constructor signatures exactly.
    @dataclass
    class Config:
        capacity_factor: float = 1.0   # Matches source
        noise_epsilon: float = 1e-2    # Matches source

    # This applies to ALL numeric values: learning rates, epsilon values,
    # dropout rates, capacity factors, number of layers, hidden dimensions, etc.
    # If the source says 1.0, write 1.0. If the source says 0.1, write 0.1.
    # NEVER round, adjust, or "improve" any default.


## Principle 3: Preserve Exact Reduction Operations

WRONG: Substituting one reduction for another.

    # PyTorch source:
    #   return routing_weights.mean(dim=0)

    # WRONG! .sum() != .mean() -- different semantics!
    def expert_utilization(routing_weights):
        return routing_weights.sum(axis=0)  # Should be .mean()!

    # PyTorch source:
    #   expert_counts = routing_weights.sum(dim=0)

    # WRONG! .mean() != .sum()
    def expert_counts(routing_weights):
        return routing_weights.mean(axis=0)  # Should be .sum()!

CORRECT: Use the exact same reduction as the source.

    # If source uses .mean(dim=0), use .mean(axis=0)
    def expert_utilization(routing_weights):
        return jnp.mean(routing_weights, axis=0)

    # If source uses .sum(dim=0), use .sum(axis=0)
    def expert_counts(routing_weights):
        return jnp.sum(routing_weights, axis=0)

    # PyTorch dim= maps to JAX axis= with the same integer value.
    # torch.mean(x, dim=0)  =>  jnp.mean(x, axis=0)
    # torch.sum(x, dim=-1)  =>  jnp.sum(x, axis=-1)
    # torch.max(x, dim=1)   =>  jnp.max(x, axis=1)
    # NEVER swap .mean() for .sum() or vice versa.


## Principle 4: Preserve Function Placement and Structure

WRONG: Relocating a method from one class to another.

    # PyTorch source:
    #   class Router(nn.Module):
    #       def __init__(self, ...):
    #           self.capacity = lambda batch_size: int(batch_size * cf * k / E)

    # WRONG! Moving capacity computation to a different class
    class MixtureOfExperts(nn.Module):
        def __call__(self, x):
            capacity = int(...)  # Relocated from Router

CORRECT: Keep methods and attributes on the same class as the source.

    # CORRECT: capacity stays on Router where the source defines it
    class Router(nn.Module):
        ...
        def capacity(self, batch_size: int) -> int:
            return int(batch_size * self.capacity_factor * self.k / self.num_experts)


## Principle 5: Preserve All Utility Components

WRONG: Dropping "non-essential" components like logging, metrics, or I/O.

    # PyTorch source has TensorBoard logging in the trainer.
    # WRONG! Dropping it because "it's not core model logic"
    class Trainer:
        def __init__(self, ...):
            # No tensorboard setup  <-- MISSING from source

CORRECT: Convert ALL components, including logging and metrics.

    # CORRECT: Preserve TensorBoard logging using JAX-ecosystem equivalent
    class Trainer:
        def __init__(self, ..., tensorboard_dir=None):
            self.writer = None
            if tensorboard_dir:
                os.makedirs(tensorboard_dir, exist_ok=True)
                from tensorboardX import SummaryWriter
                self.writer = SummaryWriter(tensorboard_dir)


## Why faithfulness matters:

1. **Reproducibility**: Users expect identical outputs from the JAX version when
   loaded with the same weights. Changed defaults or reductions break this.
2. **Weight loading**: Different initializers mean the JAX model cannot use
   PyTorch pretrained weights correctly for fine-tuning or inference.
3. **Testing**: Equivalence tests compare source and converted outputs. Semantic
   changes cause test failures that are hard to debug.
4. **Trust**: If users find the conversion changed their defaults, they lose
   confidence in the entire output and must audit every line.
5. **Downstream code**: Other code may depend on specific method placements,
   return value semantics, or default behaviors.
"""
