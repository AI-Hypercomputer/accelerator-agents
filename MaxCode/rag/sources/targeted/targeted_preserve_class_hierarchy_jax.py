"""
TARGETED JAX PATTERN: Preserve Class Hierarchy and All Source Components

CRITICAL: When converting PyTorch to JAX/Flax, preserve EVERY class, function,
and method from the source. Do not merge classes, drop base classes, or omit
utility functions/classes — even if they seem redundant. The goal is a faithful
1:1 conversion, not a redesign.

## WRONG: Merging base class into subclass

    # Source has:
    #   class ExpertBase(nn.Module): ...    # base with 2-layer network
    #   class FFNExpert(ExpertBase): ...    # subclass with configurable layers

    # WRONG! Merging them loses the base class and breaks code that
    # instantiates ExpertBase directly.
    class FFNExpert(nn.Module):
        config: MoEConfig
        # ... only the subclass, base class gone

## CORRECT: Preserve both classes

    class ExpertBase(nn.Module):
        input_dim: int
        output_dim: int
        hidden_dim: int = None

        def setup(self):
            hdim = self.hidden_dim if self.hidden_dim is not None else 4 * self.input_dim
            self.dense1 = nn.Dense(hdim)
            self.dense2 = nn.Dense(self.output_dim)

        def __call__(self, x):
            x = self.dense1(x)
            x = nn.relu(x)
            x = self.dense2(x)
            return x

    class FFNExpert(nn.Module):
        input_dim: int
        output_dim: int
        hidden_dim: int = None
        num_layers: int = 2
        dropout_rate: float = 0.1

        @nn.compact
        def __call__(self, x, deterministic=True):
            hdim = self.hidden_dim if self.hidden_dim is not None else 4 * self.input_dim
            for i in range(self.num_layers - 1):
                x = nn.Dense(hdim, name=f'dense_{i}')(x)
                x = nn.relu(x)
                x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
            x = nn.Dense(self.output_dim, name=f'dense_{self.num_layers - 1}')(x)
            return x

## WRONG: Dropping get_config / serialization methods

    # Source has get_config() on multiple classes for checkpoint serialization.
    # WRONG! Omitting these breaks save/load workflows.
    class MixtureOfExperts(nn.Module):
        # ... no get_config method

## CORRECT: Preserve get_config methods

    class MixtureOfExperts(nn.Module):
        input_dim: int
        output_dim: int
        num_experts: int
        k: int = 1

        # ... other methods ...

        def get_config(self):
            return {
                'input_dim': self.input_dim,
                'output_dim': self.output_dim,
                'num_experts': self.num_experts,
                'k': self.k,
            }

## WRONG: Omitting utility classes and functions

    # Source has:
    #   def expert_utilization(routing_weights): ...
    #   def expert_capacity_utilization(routing_weights, capacity): ...
    #   def routing_entropy(routing_weights): ...
    #   def expert_correlation(expert_outputs): ...
    #   class MoEMetrics: ...

    # WRONG! Only converting some functions and dropping the class.
    def expert_utilization(routing_weights):
        return routing_weights.mean(axis=0)
    def routing_entropy(routing_weights):
        ...
    # expert_capacity_utilization -- MISSING
    # expert_correlation -- MISSING
    # MoEMetrics class -- MISSING

## CORRECT: Convert ALL functions and classes

    def expert_utilization(routing_weights):
        return jnp.mean(routing_weights, axis=0)

    def expert_capacity_utilization(routing_weights, capacity):
        expert_counts = jnp.sum(routing_weights, axis=0)
        return expert_counts / capacity

    def routing_entropy(routing_weights):
        eps = 1e-10
        probs = routing_weights + eps
        return -(probs * jnp.log(probs)).sum(axis=-1).mean()

    def expert_correlation(expert_outputs):
        num_experts = len(expert_outputs)
        correlations = jnp.zeros((num_experts, num_experts))
        for i in range(num_experts):
            for j in range(i + 1, num_experts):
                xi = expert_outputs[i].flatten()
                xj = expert_outputs[j].flatten()
                corr = jnp.dot(xi, xj) / (jnp.linalg.norm(xi) * jnp.linalg.norm(xj))
                correlations = correlations.at[i, j].set(corr)
                correlations = correlations.at[j, i].set(corr)
        return correlations

    class MoEMetrics:
        def __init__(self, num_experts, expert_capacity=None):
            self.num_experts = num_experts
            self.expert_capacity = expert_capacity

        def compute_metrics(self, routing_weights, expert_outputs=None):
            metrics = {
                'expert_utilization': expert_utilization(routing_weights),
                'routing_entropy': routing_entropy(routing_weights),
            }
            if self.expert_capacity is not None:
                metrics['capacity_utilization'] = expert_capacity_utilization(
                    routing_weights, self.expert_capacity
                )
            if expert_outputs is not None:
                metrics['expert_correlation'] = expert_correlation(expert_outputs)
            return metrics

## Why preserving everything matters:

1. **API compatibility**: Downstream code may instantiate ExpertBase, call get_config(),
   or use MoEMetrics. Dropping them breaks the public interface.
2. **Testing**: Equivalence tests compare source and converted outputs class-by-class.
   Missing classes cause test failures.
3. **Faithfulness**: The conversion should be a translation, not a redesign. Users
   expect to find every source component in the output.
4. **Weight loading**: get_config() is used during checkpoint serialization/deserialization.
   Without it, weights cannot be saved or loaded correctly.
"""
