# Flax Linen Documentation: Flax Basics
# Source: https://flax-linen.readthedocs.io/en/latest/guides/flax_fundamentals/flax_basics.html
"""
Flax Basics: Complete Reference Documentation

Core Workflow Components
========================

1. Model Instantiation and Parameter Initialization
----------------------------------------------------
Flax uses nn.Module base class for all models. Parameters are NOT stored with models
themselves but rather initialized separately through the init() method using a PRNG key
and dummy input data.

Key concept: The dummy input data triggers shape inference - you only declare the number
of features wanted in the output, and Flax automatically determines kernel dimensions
from input specifications alone.

Parameters are returned as a pytree structure matching the model's architecture.

    import flax.linen as nn
    import jax
    import jax.numpy as jnp

    model = nn.Dense(features=5)
    key = jax.random.PRNGKey(0)
    params = model.init(key, jnp.ones((1, 3)))  # shape inference from dummy input

2. Forward Passes with apply()
------------------------------
Models cannot be called directly. Use apply() with parameters:

    output = model.apply(params, x)

3. Training with Gradient Descent
---------------------------------
- Define loss function with jax.vmap() for vectorization
- Compute gradients using jax.value_and_grad()
- Update parameters iteratively with learning rate scaling

4. Optimization with Optax
--------------------------
    import optax
    tx = optax.adam(learning_rate=1e-3)
    opt_state = tx.init(params)
    grads = jax.grad(loss_fn)(params, x, y)
    updates, opt_state = tx.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

Defining Custom Models
======================

Module Basics
-------------
Custom models extend nn.Module (a Python dataclass) with:
- Data fields for configuration
- setup() method for submodule registration
- __call__() method for forward computation

Explicit approach (using setup):

    class ExplicitMLP(nn.Module):
      features: Sequence[int]

      def setup(self):
        self.layers = [nn.Dense(feat) for feat in self.features]

      def __call__(self, inputs):
        x = inputs
        for i, layer in enumerate(self.layers[:-1]):
          x = nn.relu(layer(x))
        x = self.layers[-1](x)
        return x

Compact approach (using @nn.compact):

    class SimpleMLP(nn.Module):
      features: Sequence[int]

      @nn.compact
      def __call__(self, inputs):
        x = inputs
        for i, feat in enumerate(self.features[:-1]):
          x = nn.relu(nn.Dense(feat, name=f'layers_{i}')(x))
        x = nn.Dense(self.features[-1], name=f'layers_{len(self.features)-1}')(x)
        return x

Parameter Declaration
---------------------
Custom parameters use self.param() within modules:

    kernel = self.param('kernel',
                        self.kernel_init,
                        (inputs.shape[-1], self.features))

Arguments:
- Name for parameter identification in pytree
- Initialization function with signature (PRNGKey, *args, **kwargs)
- Shape and dtype arguments passed to init function

Variables and State Management
------------------------------
Beyond parameters, modules can maintain mutable state through variables:

Pattern: self.variable(collection_name, variable_name, init_fn, *args)

Usage example - batch normalization with running mean:
- Detect initialization via self.has_variable()
- Create tracked variables with self.variable()
- Update during apply() with mutable=['collection_name']
- Extract and update state between training steps

State update pattern:

    y, updated_state = model.apply(variables, x, mutable=['batch_stats'])
    variables = flax.core.freeze({'params': params, **updated_state})

This separates mutable state from frozen parameters for explicit control during training.

Serialization
-------------
- serialization.to_bytes() - convert parameters to byte representation
- serialization.to_state_dict() - convert to dictionary format
- serialization.from_bytes() - restore from bytes using a template structure
"""
