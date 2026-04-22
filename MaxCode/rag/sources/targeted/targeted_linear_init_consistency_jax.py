"""
TARGETED RAG: Use Consistent Initialization for All Linear Layers
==================================================================

When converting PyTorch models that define a custom Linear() helper function
with explicit initialization (e.g., xavier_uniform), ALL nn.Linear layers
in the model must use that same helper in JAX. Do not use bare nn.Dense for
some layers while using the custom helper for others.

WRONG -- Inconsistent initialization across layers:
-----------------------------------------------------
    # PyTorch source defines a custom Linear helper:
    #   def Linear(in_features, out_features, bias=True):
    #       m = nn.Linear(in_features, out_features, bias)
    #       nn.init.xavier_uniform_(m.weight)
    #       if bias: nn.init.constant_(m.bias, 0.)
    #       return m
    #
    # Some layers use it: self.fc1 = Linear(dim, 4*dim)
    # Other layers use bare nn.Linear: self.proj1 = nn.Linear(dim, dim)

    # JAX helper correctly uses xavier_uniform:
    def Linear(in_features, out_features, bias=True, name=None):
        return nn.Dense(out_features, use_bias=bias,
                        kernel_init=nn.initializers.xavier_uniform(),
                        bias_init=nn.initializers.zeros_init(),
                        name=name)

    # WRONG! fc1 uses the helper but proj1 uses bare nn.Dense
    fc1 = Linear(dim, 4 * dim, name='fc1')       # xavier_uniform -- correct
    proj1 = nn.Dense(dim, name='proj1')           # lecun_normal -- WRONG!

WHY THIS IS WRONG:
- In PyTorch, both bare nn.Linear layers use kaiming_uniform by default
- The JAX helper uses xavier_uniform (matching the PyTorch helper)
- But bare nn.Dense uses lecun_normal (different from PyTorch's kaiming_uniform)
- This creates INCONSISTENT initialization between layers in the same model
- Layers initialized with different distributions train differently
- Weight transfer from PyTorch checkpoints will have mismatched assumptions

CORRECT -- Use the same Linear helper for ALL linear layers:
--------------------------------------------------------------
    # CORRECT: All linear layers use the same helper, matching PyTorch behavior
    fc1 = Linear(dim, 4 * dim, name='fc1')
    proj1 = Linear(dim, dim, name='proj1')        # Use helper, not bare nn.Dense
    proj2 = Linear(dim, dim, name='proj2')         # Use helper, not bare nn.Dense
    out_layer = Linear(dim, output_dim, name='out_layer')  # Use helper here too

    # If the PyTorch source uses bare nn.Linear (no custom init), use bare nn.Dense:
    #   self.proj = nn.Linear(dim, dim)  ->  proj = nn.Dense(dim, name='proj')
    #
    # If the PyTorch source uses a custom init helper, use the JAX equivalent for ALL:
    #   self.fc1 = Linear(dim, 4*dim)    ->  fc1 = Linear(dim, 4*dim, name='fc1')
    #   self.proj = nn.Linear(dim, dim)  ->  proj = Linear(dim, dim, name='proj')
    #
    # The key insight: in PyTorch, nn.Linear always uses kaiming_uniform.
    # When some layers get xavier_uniform via a helper, the REST still have
    # kaiming_uniform. In JAX, bare nn.Dense uses lecun_normal (different!).
    # So for layers without explicit init in PyTorch, using bare nn.Dense in JAX
    # is acceptable. But when the SAME CLASS mixes helper and bare, be consistent.

RULE: When a model defines a custom Linear() helper, use it for ALL linear
layers in that model to ensure consistent initialization behavior.
"""
