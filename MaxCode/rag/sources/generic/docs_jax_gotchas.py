# JAX Common Gotchas and Patterns
# Source: https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html
"""
JAX Sharp Bits: Common Gotchas and Patterns
=============================================

Pure Functions
--------------
JAX transforms and compilation work exclusively on functionally pure Python functions.
A pure function must satisfy:
- All input data enters through function parameters
- All results exit through function returns
- Invoking with identical inputs always produces identical outputs

Side effects (print, global state, iterators) only execute on first JIT call:

    # BAD: print only runs on first call
    @jit
    def f(x):
        print("called")  # only prints once!
        return x + 1

    # BAD: global variable captured at trace time
    g = 0.
    @jit
    def f(x):
        return x + g  # uses g=0 forever, even if g changes later

    # BAD: iterators have state
    iterator = iter(range(10))
    jax.lax.fori_loop(0, 10, lambda i, x: x + next(iterator), 0)  # WRONG

Immutable Arrays and .at[] Updates
------------------------------------
JAX arrays are immutable. Direct index assignment fails:

    jax_array[1, :] = 1.0  # TypeError!

Use functional .at API instead:

    updated = jax_array.at[1, :].set(1.0)      # set values
    updated = jax_array.at[1, :].add(1.0)      # add to values
    updated = jax_array.at[1, :].mul(2.0)      # multiply values
    updated = jax_array.at[::2, 3:].add(7.)    # slice indexing

IMPORTANT: Inside JIT, the compiler optimizes .at[] to in-place when input isn't reused.
IMPORTANT: Slice sizes in JIT must be static (can't depend on array values).

Random Numbers
--------------
JAX uses explicit key-based state management (no global RNG state):

    key = jax.random.key(0)
    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, (5, 5))

    # Split for multiple independent uses
    key, *subkeys = jax.random.split(key, num=4)

Never reuse the same key for different random operations.

Control Flow in JIT
--------------------
Python if/else and for loops are traced once. Use JAX primitives for dynamic control:

    # Instead of: if x > 0: ...
    result = jax.lax.cond(x > 0, true_fn, false_fn, x)

    # Instead of: for i in range(n): ...
    result = jax.lax.fori_loop(0, n, body_fn, init_val)

    # For sequential state + accumulation:
    final_carry, outputs = jax.lax.scan(step_fn, init_carry, xs)

    # For parallel prefix operations:
    result = jax.lax.associative_scan(binary_fn, elems)

    # Dynamic while loop:
    result = jax.lax.while_loop(cond_fn, body_fn, init_val)

Static vs Dynamic Shapes
--------------------------
All output and intermediate arrays must have static shape in JIT:

    # BAD: shape depends on values
    x_filtered = x[~jnp.isnan(x)]  # dynamic shape!

    # GOOD: use where to maintain static shape
    x_clean = jnp.where(~jnp.isnan(x), x, 0)

Out-of-Bounds Indexing
-----------------------
JAX can't raise errors from accelerators. Instead:
- Retrieval: indices clamped to bounds (returns last element)
- Updates: out-of-bounds ops silently skipped

    jnp.arange(10)[11]  # Returns 9, not error
    jnp.arange(10.0).at[11].get(mode='fill', fill_value=jnp.nan)  # Returns nan

Double Precision (64-bit)
--------------------------
JAX defaults to float32. Enable float64 explicitly:

    jax.config.update("jax_enable_x64", True)  # must run at startup
    # Or: JAX_ENABLE_X64=True python script.py

PyTree Patterns
----------------
JAX operates on pytrees - nested structures of arrays. Common patterns:

    # Pytrees can be dicts, lists, tuples, NamedTuples, dataclasses
    params = {'dense': {'kernel': w, 'bias': b}}

    # tree_map applies a function to all leaves
    doubled = jax.tree_util.tree_map(lambda x: 2 * x, params)

    # Custom pytrees via register_pytree_node
    from jax import tree_util
    tree_util.register_pytree_node(
        MyClass,
        lambda obj: ((obj.dynamic_field,), {'static': obj.static_field}),
        lambda aux, children: MyClass(*children, **aux)
    )

Key Differences from NumPy
----------------------------
- Arrays are immutable (use .at[] for updates)
- No in-place operations (+=, *= create new arrays)
- Explicit PRNG key management (no global state)
- Type promotion rules differ
- No dynamic shapes in JIT
- Out-of-bounds indexing clamps instead of raising
"""
