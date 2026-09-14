# Flax Linen Documentation: setup vs nn.compact
# Source: https://flax-linen.readthedocs.io/en/latest/guides/flax_fundamentals/setup_or_nncompact.html
"""
Flax Linen: setup vs compact Documentation

Overview
--------
Flax's module system provides two distinct approaches for defining submodules and variables:

Explicit Definition (setup): Variables and submodules are assigned to self.<attr> within a
setup() method, mirroring PyTorch's conventional pattern. Forward pass logic is then
implemented in separate methods.

Inline Definition (nn.compact): Network architecture is written directly within a single
method marked with the @nn.compact decorator, collocating component definitions with
their usage points.

Both methods are functionally equivalent and fully interoperable throughout Flax.

Code Examples
-------------

Setup Approach::

    class MLP(nn.Module):
      def setup(self):
        self.dense1 = nn.Dense(32)
        self.dense2 = nn.Dense(32)

      def __call__(self, x):
        x = self.dense1(x)
        x = nn.relu(x)
        x = self.dense2(x)
        return x

Compact Approach::

    class MLP(nn.Module):
      @nn.compact
      def __call__(self, x):
        x = nn.Dense(32, name="dense1")(x)
        x = nn.relu(x)
        x = nn.Dense(32, name="dense2")(x)
        return x

When to Choose Each Approach
----------------------------

Prefer nn.compact when:
- Reducing navigation between variable definitions and usage sites
- Handling conditional logic or loops that affect variable creation
- Aligning code structure with mathematical notation
- Implementing shape inference dependent on input dimensions

Prefer setup when:
- Maintaining PyTorch compatibility conventions
- Preferring explicit separation between definitions and application
- Requiring multiple distinct forward pass methods

Key patterns for nn.compact:
- Submodules are instantiated inline: nn.Dense(features, name="layer_name")(x)
- Parameters declared via self.param('name', init_fn, shape)
- Variables declared via self.variable('collection', 'name', init_fn)
- Only one method per module can use @nn.compact
- Auto-naming: if no name= is provided, Flax assigns Dense_0, Dense_1, etc.
"""
