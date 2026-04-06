"""
TARGETED JAX PATTERN: MoE Expert Dispatch with Capacity-Based Routing

CRITICAL: When converting Mixture-of-Experts layers, the Experts class MUST use
capacity-based dispatch with einsum dispatch/combine tensors. Do NOT use per-token
weight gathering or dense all-experts einsum.

## WRONG approach 1 (per-token gather -- DO NOT DO THIS):

    # WRONG! Gathers individual expert weights per token
    flat_indices = top_k_index.reshape(-1)
    gate_up_w = gate_up_proj[flat_indices]       # [T*K, 2I, H]
    hidden_repeated = jnp.repeat(x, top_k, axis=0)
    out = jnp.sum(hidden_repeated[:, None, :] * gate_up_w, axis=-1)  # unbatched!
    # This does T*K individual matmuls -- not batched, XLA-unfriendly

## WRONG approach 2 (dense einsum -- DO NOT DO THIS):

    # WRONG! Computes ALL experts for ALL tokens
    expert_outputs = jnp.einsum('th,ehi->tei', x, expert_w1)  # O(T*E*H*I)
    # For E=64: wastes 93% of compute (each token only uses K=4 experts)

## CORRECT approach (capacity-based dispatch with einsum):

    import jax
    import jax.numpy as jnp
    from flax import linen as nn

    class Experts(nn.Module):
        config: Qwen3NextConfig
        capacity_factor: float = 1.5

        @nn.compact
        def __call__(self, hidden_states, top_k_indices, top_k_weights):
            config = self.config
            num_experts = config.num_experts
            hidden_dim = config.hidden_size
            intermediate_dim = config.moe_intermediate_size
            top_k = config.num_experts_per_tok

            # Expert weight parameters: [E, 2*I, H] and [E, H, I]
            gate_up_proj = self.param('gate_up_proj',
                nn.initializers.normal(config.initializer_range),
                (num_experts, 2 * intermediate_dim, hidden_dim))
            down_proj = self.param('down_proj',
                nn.initializers.normal(config.initializer_range),
                (num_experts, hidden_dim, intermediate_dim))

            num_tokens = hidden_states.shape[0]

            # ---- Step 1: Compute per-expert capacity ----
            raw_capacity = max((num_tokens * top_k + num_experts - 1) // num_experts, 1)
            capacity = int(raw_capacity * self.capacity_factor)

            # ---- Step 2: Build dispatch and combine tensors ----
            # expert_one_hot: [T, K, E]
            expert_one_hot = jax.nn.one_hot(top_k_indices, num_experts)

            # Flatten T*K for per-expert position counting
            flat_mask = expert_one_hot.reshape(-1, num_experts)  # [T*K, E]

            # Position within each expert's buffer (0-indexed via cumsum)
            positions = (jnp.cumsum(flat_mask, axis=0) - 1) * flat_mask  # [T*K, E]

            # Drop tokens exceeding capacity
            within_cap = (positions < capacity) & (flat_mask > 0)
            safe_positions = jnp.where(within_cap, positions, 0).astype(jnp.int32)

            # Dispatch tensor: [T*K, E, C] via one-hot on position
            pos_one_hot = jax.nn.one_hot(safe_positions, capacity)  # [T*K, E, C]
            dispatch_flat = pos_one_hot * within_cap[..., None]

            # Combine tensor: dispatch weighted by routing weights
            flat_weights = top_k_weights.reshape(-1)  # [T*K]
            combine_flat = dispatch_flat * flat_weights[:, None, None]

            # Aggregate over K dimension: [T, E, C]
            dispatch = dispatch_flat.reshape(num_tokens, top_k, num_experts, capacity).sum(axis=1)
            combine = combine_flat.reshape(num_tokens, top_k, num_experts, capacity).sum(axis=1)

            # ---- Step 3: Dispatch tokens to expert buffers ----
            # [E, C, H] = einsum([T, E, C], [T, H])
            expert_inputs = jnp.einsum('tec,th->ech', dispatch, hidden_states)

            # ---- Step 4: Batched expert computation ----
            gate_up_out = jnp.einsum('ech,eih->eci', expert_inputs, gate_up_proj)  # [E, C, 2I]
            gate_part, up_part = jnp.split(gate_up_out, 2, axis=-1)
            expert_out = jnp.einsum(
                'eci,ehi->ech', jax.nn.silu(gate_part) * up_part, down_proj
            )  # [E, C, H]

            # ---- Step 5: Combine -- scatter results back ----
            # [T, H] = einsum([T, E, C], [E, C, H])
            output = jnp.einsum('tec,ech->th', combine, expert_out)

            return output

## WHY this pattern is correct:

1. **Batched einsums**: All expert computation is batched via einsum. No Python loops,
   no per-token gathers, no `.at[].add()`. XLA compiles this into efficient matmuls.
2. **O(E*C*H*I)** compute where C = ceil(T*K/E)*1.5, typically C << T.
   For E=64, K=4, T=1024: C ~= 96 vs T=1024. Each expert only processes its share.
3. **Capacity overflow**: Tokens exceeding an expert's capacity are dropped via the
   `within_cap` mask. With 1.5x capacity factor, drops are rare for trained routers.
4. **dispatch/combine tensors**: The dispatch tensor routes tokens TO expert buffers,
   the combine tensor routes results BACK with routing weights. Both are [T, E, C].
5. **Matches PyTorch**: The PyTorch Qwen3NextExperts uses this capacity-based pattern
   internally (via scatter/gather ops). The einsum formulation is the JAX equivalent.

## Router weight initialization:

CRITICAL: The router (gate) weight MUST be initialized with zeros:
    weight = self.param('weight', nn.initializers.zeros_init(), (num_experts, hidden_dim))

NOT with normal initialization. Zero-init ensures uniform routing at start of training.
"""
