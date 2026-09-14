"""
TARGETED RAG: Do Not Invent Attributes or Fix Bugs in Source Code
===================================================================

When converting PyTorch to JAX, faithfully translate what the source code
ACTUALLY DOES, not what it SHOULD do. If the source has a bug (e.g.,
referencing an undefined attribute), the JAX version should reproduce
that same behavior, not silently fix it by adding the missing attribute.

WRONG -- Adding attributes that don't exist in the PyTorch source:
-------------------------------------------------------------------
    # PyTorch source:
    #   class TransformerEncoder(nn.Module):
    #       def __init__(self, embed_dim, num_heads, layers, ...):
    #           self.embed_dim = embed_dim
    #           self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)
    #           # NOTE: self.max_source_positions is NEVER defined here
    #
    #       def max_positions(self):
    #           if self.embed_positions is None:
    #               return self.max_source_positions  # Would crash: AttributeError
    #           return min(self.max_source_positions, self.embed_positions.max_positions())
    #           # Also uses self.max_source_positions -- would crash

    # WRONG! Invented max_source_positions with a made-up default value
    class TransformerEncoder(nn.Module):
        embed_dim: int
        num_heads: int
        layers: int
        max_source_positions: int = 100000  # NOT IN SOURCE! Invented attribute!

        def max_positions(self):
            return min(self.max_source_positions, self.embed_positions.max_positions())

WHY THIS IS WRONG:
- The PyTorch source never defines max_source_positions in __init__
- Adding it with a default value of 100000 introduces behavior that doesn't
  exist in the original model
- The original max_positions() method would crash if called -- the JAX version
  silently "fixes" this by inventing an attribute
- Users loading PyTorch weights into the JAX model will have an unexpected
  extra parameter that doesn't correspond to any PyTorch state
- The invented default (100000) is arbitrary and may not match user expectations

CORRECT -- Faithfully reproduce the source's behavior:
--------------------------------------------------------
    # Option A: Reproduce the bug faithfully
    class TransformerEncoder(nn.Module):
        embed_dim: int
        num_heads: int
        layers: int
        # Do NOT add max_source_positions -- it's not in the source

        def max_positions(self):
            # Faithfully translated: embed_positions is always non-None,
            # so we only need the path that actually executes
            return self.embed_positions.max_positions()

    # Option B: If max_positions() is never called in the model's forward pass,
    # translate only the code paths that are actually reachable
    class TransformerEncoder(nn.Module):
        embed_dim: int
        num_heads: int
        layers: int
        # max_positions() method omitted since it references undefined attributes
        # and is never called during forward()

RULE: Never add attributes, parameters, or default values that don't exist in
the PyTorch source. If the source has unreachable or buggy code paths,
either faithfully reproduce them or omit them -- but never "fix" them
by inventing new state.
"""
