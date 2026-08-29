"""
TARGETED JAX PATTERN: Load Balancing Loss with Attention Mask

This function computes the auxiliary load-balancing loss from Switch Transformer
(equations 4-6). It MUST support an optional attention_mask parameter to exclude
padding tokens from the loss computation. Without the mask, padding tokens
pollute the routing statistics and destabilize MoE training.

## WRONG: No attention_mask support

    def load_balancing_loss(gate_logits, num_experts, top_k):
        concatenated = jnp.concatenate(gate_logits, axis=0)
        routing_weights = jax.nn.softmax(concatenated, axis=-1)
        _, selected_experts = jax.lax.top_k(routing_weights, top_k)
        expert_mask = jax.nn.one_hot(selected_experts, num_experts)
        tokens_per_expert = jnp.mean(expert_mask, axis=0)
        router_prob_per_expert = jnp.mean(routing_weights, axis=0)
        return jnp.sum(tokens_per_expert * router_prob_per_expert[None, :]) * num_experts

    # WHY THIS IS WRONG: Without attention_mask, padding tokens are counted in
    # the mean, which dilutes the expert frequency statistics. In batched
    # inference with variable-length sequences, this makes the loss meaningless.

## WRONG: Collapsing the top_k dimension with axis=(0, 1)

    # expert_mask shape: [num_tokens, top_k, num_experts]
    # PyTorch source: torch.mean(expert_mask.float(), dim=0)
    #   -> result shape: [top_k, num_experts]

    # WRONG! axis=(0, 1) reduces BOTH token AND top_k dimensions.
    # Result shape becomes [num_experts] instead of [top_k, num_experts].
    tokens_per_expert = jnp.mean(expert_mask, axis=(0, 1))  # WRONG SHAPE!

    # WRONG! Flattening before reducing also collapses top_k.
    expert_mask_flat = expert_mask.reshape(-1, num_experts)
    tokens_per_expert = jnp.mean(expert_mask_flat, axis=0)  # WRONG SHAPE!

    # WHY THIS IS WRONG: PyTorch dim=0 reduces ONLY the first dimension.
    # The top_k dimension must be preserved. Collapsing it changes the loss
    # value and breaks expert routing during training.

## CORRECT: With attention_mask support

    def load_balancing_loss(
        gate_logits: list[jnp.ndarray],
        num_experts: int,
        top_k: int,
        attention_mask: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        if not gate_logits:
            return jnp.array(0.0)

        # Concatenate all MoE layers: [num_layers * B * T, num_experts]
        concatenated = jnp.concatenate(gate_logits, axis=0)

        routing_weights = jax.nn.softmax(concatenated, axis=-1)
        _, selected_experts = jax.lax.top_k(routing_weights, top_k)
        expert_mask = jax.nn.one_hot(selected_experts, num_experts)
        # expert_mask: [num_layers * B * T, top_k, num_experts]

        if attention_mask is None:
            # No padding: simple mean over all tokens
            tokens_per_expert = jnp.mean(expert_mask.astype(jnp.float32), axis=0)
            router_prob_per_expert = jnp.mean(routing_weights, axis=0)
        else:
            # With padding: mask out padding tokens before computing statistics
            batch_size, seq_len = attention_mask.shape
            num_layers = concatenated.shape[0] // (batch_size * seq_len)

            # Expand mask to [num_layers * B * T, top_k, num_experts]
            expert_attn_mask = jnp.broadcast_to(
                attention_mask[None, :, :, None, None],
                (num_layers, batch_size, seq_len, top_k, num_experts),
            ).reshape(-1, top_k, num_experts)

            tokens_per_expert = (
                jnp.sum(expert_mask.astype(jnp.float32) * expert_attn_mask, axis=0)
                / jnp.maximum(jnp.sum(expert_attn_mask, axis=0), 1.0)
            )

            # Expand mask to [num_layers * B * T, num_experts]
            router_attn_mask = jnp.broadcast_to(
                attention_mask[None, :, :, None],
                (num_layers, batch_size, seq_len, num_experts),
            ).reshape(-1, num_experts)

            router_prob_per_expert = (
                jnp.sum(routing_weights * router_attn_mask, axis=0)
                / jnp.maximum(jnp.sum(router_attn_mask, axis=0), 1.0)
            )

        overall_loss = jnp.sum(tokens_per_expert * router_prob_per_expert[None, :])
        return overall_loss * num_experts

## KEY POINTS:
## - The attention_mask parameter is REQUIRED (even if optional=None)
## - Use jnp.maximum(..., 1.0) to avoid division by zero
## - Broadcast the mask to match [num_layers * B * T, ...] shape
## - The ForCausalLM forward method should pass attention_mask through:
##     aux_loss = load_balancing_loss(router_logits, num_experts, top_k, attention_mask)
"""
