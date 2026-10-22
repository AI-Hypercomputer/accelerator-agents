# JAX LAX Primitive Functions Documentation
# Source: https://docs.jax.dev/en/latest/jax.lax.html
"""
JAX LAX Primitive Functions
===========================

jax.lax.scan
-------------
Signature: scan(f, init, xs=None, length=None, reverse=False, unroll=1)

Scan a function over leading array axes while carrying along state.
This enables sequential operations with accumulated results, similar to
a fold operation in functional programming.

Parameters:
- f: Function taking (carry, x) and returning (new_carry, y)
- init: Initial carry value
- xs: Input sequence (optional, stacked along axis 0)
- length: Iteration count (optional, inferred from xs)
- reverse: Process in reverse order
- unroll: Loop unrolling factor

Returns: (final_carry, stacked_ys)

Example::

    def cumsum(carry, x):
        new_carry = carry + x
        return new_carry, new_carry

    final, history = jax.lax.scan(cumsum, 0, jnp.array([1, 2, 3, 4]))
    # final = 10, history = [1, 3, 6, 10]

Use for recurrent computations, RNN cells, sequential state updates.
Inside nn.compact, use nn.scan to lift scan over Flax modules.

jax.lax.associative_scan
--------------------------
Signature: associative_scan(fn, elems, reverse=False, axis=0)

Performs a scan with an associative binary operation, in parallel.
Unlike sequential scan, this exploits associativity for O(log n) depth.

Parameters:
- fn: Binary associative function f(a, b) where f(f(a,b), c) == f(a, f(b,c))
- elems: Array elements to process
- reverse: Reverse processing direction
- axis: Dimension along which to scan

Example::

    # Parallel prefix sum
    result = jax.lax.associative_scan(jnp.add, jnp.array([1, 2, 3, 4]))
    # result = [1, 3, 6, 10]

jax.lax.dynamic_update_slice
------------------------------
Signature: dynamic_update_slice(operand, update, start_indices)

Wraps XLA's DynamicUpdateSlice operator. Updates a slice at dynamically
determined indices within a larger array. Useful for KV-cache updates.

Example::

    arr = jnp.zeros((5, 3))
    update = jnp.ones((2, 3))
    result = jax.lax.dynamic_update_slice(arr, update, (1, 0))
    # Updates rows 1-2 with ones

Common pattern for KV cache::

    cache = jax.lax.dynamic_update_slice(
        cache,                    # existing cache [max_len, features]
        new_kv[None],             # new entry [1, features]
        (cache_index, 0)          # write position
    )

jax.lax.dynamic_slice
-----------------------
Signature: dynamic_slice(operand, start_indices, slice_sizes)

Wraps XLA's DynamicSlice operator. Extracts array slices using
runtime-determined start positions.

Parameters:
- operand: Source array
- start_indices: Runtime start positions (one per dimension)
- slice_sizes: Static slice sizes (must be constants)

Example::

    arr = jnp.arange(10)
    result = jax.lax.dynamic_slice(arr, (3,), (4,))
    # result = [3, 4, 5, 6]

jax.lax.conv_general_dilated
------------------------------
Signature: conv_general_dilated(lhs, rhs, window_strides, padding,
                                 lhs_dilation=None, rhs_dilation=None,
                                 dimension_numbers=None, precision=None)

General n-dimensional convolution operator with optional dilation.

Parameters:
- lhs: Input array
- rhs: Kernel weights
- window_strides: Stride configuration
- padding: 'SAME', 'VALID', or explicit padding pairs
- dimension_numbers: Tuple of (lhs_spec, rhs_spec, out_spec) strings

Example for 1D causal convolution::

    # Input: [batch, length, channels] -> need ('NHC', 'HIO', 'NHC')
    out = jax.lax.conv_general_dilated(
        x, kernel,
        window_strides=(1,),
        padding=((kernel_size - 1, 0),),  # causal: pad left only
        dimension_numbers=('NHC', 'HIO', 'NHC')
    )

jax.lax.cond
--------------
Signature: cond(pred, true_fun, false_fun, *operands)

Conditionally apply true_fun or false_fun based on a boolean predicate.
Both branches are traced; use instead of Python if/else in JIT code.

Example::

    result = jax.lax.cond(
        x > 0,
        lambda x: x + 1,    # true branch
        lambda x: x - 1,    # false branch
        x
    )

jax.lax.fori_loop
-------------------
Signature: fori_loop(lower, upper, body_fun, init_val)

Loop from lower to upper by reduction to jax.lax.while_loop().
Implements bounded iteration with state accumulation.

Parameters:
- lower: Loop start index
- upper: Loop end index (exclusive)
- body_fun: Function(i, carry) -> new_carry
- init_val: Initial carry state

Example::

    def body(i, carry):
        return carry + i
    result = jax.lax.fori_loop(0, 10, body, 0)  # 45
"""
