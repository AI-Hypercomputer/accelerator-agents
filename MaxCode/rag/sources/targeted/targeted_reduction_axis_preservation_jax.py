"""
TARGETED JAX PATTERN: Preserve Exact Reduction Axes — Never Flatten or Combine

CRITICAL: When PyTorch uses `dim=N` in a reduction (mean, sum, max, etc.), the
JAX conversion MUST use `axis=N` with the SAME single integer. Never combine
multiple axes like `axis=(0, 1)`, and never reshape/flatten the tensor before
reducing. These change the output shape and numerical result.

This mistake is especially common in MoE load-balancing loss functions where
`expert_mask` has shape [tokens, top_k, num_experts]. The LLM "helpfully"
collapses the top_k dimension, but PyTorch's `dim=0` preserves it.

## WRONG: Combining axes when source uses a single dim

    # PyTorch source:
    #   expert_mask = one_hot(selected_experts, num_experts)
    #   # expert_mask shape: [num_tokens, top_k, num_experts]
    #   tokens_per_expert = torch.mean(expert_mask.float(), dim=0)
    #   # result shape: [top_k, num_experts]

    # WRONG! axis=(0, 1) reduces BOTH token and top_k dims.
    # Result shape becomes [num_experts] instead of [top_k, num_experts].
    tokens_per_expert = jnp.mean(expert_mask, axis=(0, 1))

    # WRONG! Flattening first, then reducing, also collapses the top_k dim.
    expert_mask_flat = expert_mask.reshape(-1, num_experts)
    tokens_per_expert = jnp.mean(expert_mask_flat, axis=0)

## WRONG: Flattening before sum changes the semantics

    # PyTorch source:
    #   tokens_per_expert = torch.sum(
    #       expert_mask.float() * expert_attention_mask, dim=0
    #   ) / torch.sum(expert_attention_mask, dim=0)
    #   # Both sums reduce dim=0 only, preserving [top_k, num_experts]

    # WRONG! Flattening expert_mask before summing collapses top_k.
    expert_mask_flattened = expert_mask.reshape(-1, num_experts)
    attn_mask_flattened = expert_attention_mask.reshape(-1, num_experts)
    tokens_per_expert = jnp.sum(expert_mask_flattened * attn_mask_flattened, axis=0)

## CORRECT: dim=0 becomes axis=0, nothing else changes

    # PyTorch source:
    #   tokens_per_expert = torch.mean(expert_mask.float(), dim=0)
    #   # shape: [num_tokens, top_k, num_experts] -> [top_k, num_experts]

    # CORRECT: axis=0 reduces only the first dimension, preserving top_k.
    tokens_per_expert = jnp.mean(expert_mask.astype(jnp.float32), axis=0)
    # result shape: [top_k, num_experts] -- matches PyTorch exactly

## CORRECT: Masked sum with axis=0 only

    # PyTorch source:
    #   tokens_per_expert = torch.sum(
    #       expert_mask.float() * expert_attention_mask, dim=0
    #   ) / torch.sum(expert_attention_mask, dim=0)

    # CORRECT: reduce axis=0 without any reshaping or flattening.
    tokens_per_expert = (
        jnp.sum(expert_mask.astype(jnp.float32) * expert_attention_mask, axis=0)
        / jnp.maximum(jnp.sum(expert_attention_mask, axis=0), 1e-9)
    )
    # result shape: [top_k, num_experts] -- matches PyTorch exactly

## CORRECT: Subsequent operations use the preserved shape

    # PyTorch source:
    #   router_prob_per_expert = torch.mean(routing_weights, dim=0)
    #   # routing_weights shape: [num_tokens, num_experts]
    #   # result shape: [num_experts]
    #   overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert)

    # CORRECT: router_prob_per_expert is [num_experts], tokens_per_expert is
    # [top_k, num_experts]. Broadcasting handles the shape difference.
    router_prob_per_expert = jnp.mean(routing_weights, axis=0)
    overall_loss = jnp.sum(tokens_per_expert * router_prob_per_expert[None, :])

## The general rule:

    # torch.mean(x, dim=N)   =>  jnp.mean(x, axis=N)
    # torch.sum(x, dim=N)    =>  jnp.sum(x, axis=N)
    # torch.max(x, dim=N)    =>  jnp.max(x, axis=N)
    # torch.min(x, dim=N)    =>  jnp.min(x, axis=N)
    #
    # The axis integer is ALWAYS the same as the dim integer.
    # NEVER combine axes: dim=0 does NOT become axis=(0, 1).
    # NEVER flatten before reducing: reshape(-1, K) + axis=0 != axis=0 on original.
    # NEVER add axes that are not in the source.

## Why this matters:

1. **Shape change**: `axis=(0, 1)` produces a different output shape than
   `axis=0`. Downstream code expecting [top_k, num_experts] will break or
   silently compute wrong results with [num_experts].

2. **Numerical change**: Reducing over more elements changes the mean/sum
   value. `mean(x, axis=0)` divides by `x.shape[0]`, while
   `mean(x, axis=(0,1))` divides by `x.shape[0] * x.shape[1]`.

3. **Load-balancing loss**: In MoE models, this bug makes the auxiliary loss
   numerically wrong, which destabilizes expert routing during training.
   Experts may collapse to a single active expert or oscillate wildly.

4. **Flattening is not neutral**: `x.reshape(-1, K)` followed by `sum(axis=0)`
   is mathematically equivalent to `sum(axis=tuple(range(x.ndim-1)))` — it
   reduces ALL leading dimensions, not just the first one.

5. **Rule of thumb**: If the source says `dim=0`, write `axis=0` and touch
   nothing else. Do not reshape, flatten, squeeze, or combine axes. The
   tensor shape flowing through JAX should match PyTorch at every step.
"""
