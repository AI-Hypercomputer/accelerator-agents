# Flax Linen Module API Reference
# Source: https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/module.html
"""
Complete Flax Linen Module API Reference
=========================================

flax.linen.Module is the foundational base class for all neural network modules in Flax.
All Flax Modules are Python 3.7 dataclasses and should override setup() rather than __init__.

Setup vs Compact Patterns
--------------------------

Setup Pattern::

    class MyModule(nn.Module):
      features: Tuple[int, ...] = (16, 4)

      def setup(self):
        self.dense1 = nn.Dense(self.features[0])
        self.dense2 = nn.Dense(self.features[1])

      def __call__(self, x):
        return self.dense2(nn.relu(self.dense1(x)))

Compact Pattern::

    class MyModule(nn.Module):
      features: int = 16

      @nn.compact
      def __call__(self, x):
        x = nn.Dense(self.features)(x)
        x = nn.relu(x)
        return nn.Dense(4)(x)

Initialization Methods
-----------------------

init(rngs, *args, method=None, mutable=DenyList(deny='intermediates'), **kwargs)
    Initializes module variables. A single PRNGKey is treated as {'params': key}.
    For multiple RNG streams, pass a dict: {'params': key1, 'dropout': key2}.

    model = MyModule()
    variables = model.init(jax.random.key(0), dummy_input)

init_with_output(rngs, *args, ...)
    Returns both the output and variables as a tuple: (output, vars).

lazy_init(rngs, *args, ...)
    Initializes variables without computing on actual data.
    Accepts jax.ShapeDtypeStruct for memory-efficient initialization.

Execution Methods
------------------

apply(variables, *args, rngs=None, method=None, mutable=False, **kwargs)
    Applies a module method to variables and returns output.
    If mutable collections specified, returns (output, updated_state).

    output = model.apply(variables, x)
    output, state = model.apply(variables, x, mutable=['batch_stats'])

bind(variables, *args, rngs=None, mutable=False)
    Creates an interactive Module instance. Useful for debugging.

Variable Management
--------------------

param(name, init_fn, *init_args, unbox=True, **init_kwargs)
    Declares read-only parameters in the "params" collection.
    init_fn receives PRNG key automatically as first argument.

    # Inside @nn.compact or setup():
    kernel = self.param('kernel', nn.initializers.lecun_normal(), (in_feat, out_feat))
    bias = self.param('bias', nn.initializers.zeros, (out_feat,))

variable(col, name, init_fn=None, *init_args, unbox=True, **init_kwargs)
    Declares mutable or immutable variables in named collections.
    Unlike param(), PRNG keys must be passed explicitly.

    # For KV cache or running statistics:
    cache_key = self.variable('cache', 'cached_key', jnp.zeros, (max_len, head_dim))
    cache_key.value = updated_value  # update during forward pass

get_variable(col, name, default=None)
    Retrieves variable values from specified collections.

put_variable(col, name, value)
    Updates mutable variable values.

has_variable(col, name)
    Checks variable existence. Useful for conditional initialization.

    is_initialized = self.has_variable('cache', 'cached_key')

RNG Management
---------------

make_rng(name='params')
    Returns a new PRNG key from a named RNG sequence.
    Each call splits the previous key for new values.

    dropout_key = self.make_rng('dropout')

Inspection Methods
-------------------

is_initializing()
    Returns True when running under module.init() or nn.init()().

    if self.is_initializing():
        # Do initialization-specific logic
        cache = jnp.zeros((max_len, features))

is_mutable_collection(col)
    Checks if a variable collection is mutable during current execution.

path (property)
    Returns the module's path as a tuple.

Intermediate Value Capture
---------------------------

sow(col, name, value, reduce_fn=<append>, init_fn=<empty_tuple>)
    Stores intermediate values without explicit container passing.

    self.sow('intermediates', 'attention_weights', attn_weights)
    # Later: y, state = model.apply(variables, x, mutable=['intermediates'])

Complete Training Pattern
--------------------------

::

    class Transformer(nn.Module):
      config: TransformerConfig

      @nn.compact
      def __call__(self, x, train=False):
        x = nn.Dense(self.config.hidden_size)(x)
        x = nn.Dropout(rate=0.1, deterministic=not train)(x)
        x = nn.LayerNorm()(x)
        return nn.Dense(self.config.vocab_size)(x)

    model = Transformer(config=config)
    variables = model.init({'params': key1, 'dropout': key2}, dummy_input)

    # Training step
    def train_step(variables, batch, dropout_rng):
        def loss_fn(params):
            logits = model.apply(
                {'params': params},
                batch['input'],
                train=True,
                rngs={'dropout': dropout_rng}
            )
            return cross_entropy_loss(logits, batch['labels'])

        grads = jax.grad(loss_fn)(variables['params'])
        return grads

Multiple RNG Streams
---------------------

::

    class NoisyModel(nn.Module):
      @nn.compact
      def __call__(self, x, add_noise=False):
        x = nn.Dense(16)(x)
        if add_noise:
          noise_key = self.make_rng('noise')
          x = x + jax.random.normal(noise_key, x.shape)
        return nn.Dense(1)(x)

    model = NoisyModel()
    rngs = {'params': jax.random.key(0), 'noise': jax.random.key(1)}
    variables = model.init(rngs, x)
    out = model.apply(variables, x, add_noise=True, rngs=rngs)
"""
