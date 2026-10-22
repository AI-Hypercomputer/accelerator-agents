"""
TARGETED RAG: Preserve Helper Function Call Sites — No Dead Code
=================================================================

When converting PyTorch to JAX, if the source defines a helper function and
calls it from another function, the JAX version MUST also call the helper.
Do not inline the helper's logic and leave the helper as dead code.

WRONG -- Inlining logic and leaving helper as dead code:
----------------------------------------------------------
    # PyTorch source:
    #   def fill_with_neg_inf(t):
    #       return t.float().fill_(float('-inf')).type_as(t)
    #
    #   def buffered_future_mask(tensor, tensor2=None):
    #       dim1 = dim2 = tensor.size(0)
    #       if tensor2 is not None:
    #           dim2 = tensor2.size(0)
    #       future_mask = torch.triu(fill_with_neg_inf(torch.ones(dim1, dim2)), ...)
    #       return future_mask[:dim1, :dim2]

    # WRONG! fill_with_neg_inf is defined but never called -- dead code
    def fill_with_neg_inf(t):
        return jnp.full_like(t, float('-inf'), dtype=t.dtype)

    def buffered_future_mask(tensor, tensor2=None):
        dim1 = tensor.shape[0]
        dim2 = dim1 if tensor2 is None else tensor2.shape[0]
        # WRONG: inlined the logic instead of calling fill_with_neg_inf
        inf_matrix = jnp.full((dim1, dim2), float('-inf'), dtype=jnp.float32)
        future_mask = jnp.triu(inf_matrix, 1 + abs(dim2 - dim1))
        return future_mask[:dim1, :dim2]

WHY THIS IS WRONG:
- fill_with_neg_inf preserves dtype via .type_as(t) -- important for FP16/BF16
- The inlined version hardcodes jnp.float32, losing mixed-precision support
- Dead code confuses maintenance -- readers expect the helper to be used
- The source author created the helper for a reason (dtype safety)

CORRECT -- Call the helper function just as the source does:
-------------------------------------------------------------
    def fill_with_neg_inf(t):
        \"\"\"FP16-compatible function that fills a tensor with -inf.\"\"\"
        return jnp.full_like(t, float('-inf'))

    def buffered_future_mask(tensor, tensor2=None):
        dim1 = tensor.shape[0]
        dim2 = dim1 if tensor2 is None else tensor2.shape[0]
        # CORRECT: calls fill_with_neg_inf just like the source
        future_mask = jnp.triu(
            fill_with_neg_inf(jnp.ones((dim1, dim2))),
            1 + abs(dim2 - dim1)
        )
        return future_mask[:dim1, :dim2]

GENERAL RULE:
- If the source defines function A and calls it from function B,
  the JAX version must also call A from B.
- Never inline A's logic into B and leave A as dead code.
- This preserves dtype handling, code structure, and maintainability.
"""
