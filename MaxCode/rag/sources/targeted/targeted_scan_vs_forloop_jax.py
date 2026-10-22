"""
TARGETED JAX PATTERN: scan vs fori_loop vs Python for-loop

When converting sequential loops from PyTorch to JAX, choose the right primitive.
NEVER use a plain Python for-loop over a dynamic range for sequential computation --
it unrolls at trace time, causing slow compilation and large XLA graphs.

## Decision Table:

| Pattern                          | JAX Primitive        | When to Use                          |
|----------------------------------|----------------------|--------------------------------------|
| Sequential state + collect outputs| `jax.lax.scan`      | RNN steps, chunk scans, time series  |
| Sequential state, no outputs     | `jax.lax.fori_loop` | Iterative refinement, power iteration|
| Fixed small N (< ~8)            | Python for-loop      | Unrolling is acceptable              |
| Independent iterations           | `jax.vmap`          | Batched computation, no dependencies |

## WRONG: Python for-loop for sequential scan (DO NOT DO THIS):

    # WRONG! Unrolls N iterations at trace time -> huge XLA graph, slow compile
    state = init_state
    outputs = []
    for i in range(num_chunks):
        state, out = step_fn(state, inputs[i])
        outputs.append(out)
    outputs = jnp.stack(outputs)

## CORRECT: jax.lax.scan for sequential state + outputs:

    import jax
    import jax.numpy as jnp

    def scan_chunks(init_state, inputs):
        '''
        Process chunks sequentially, accumulating state and collecting outputs.

        Args:
            init_state: [batch, heads, k_dim, v_dim] initial recurrent state
            inputs: tuple of arrays, each with leading dim = num_chunks
                    (arrays are sliced along axis 0 for each step)

        Returns:
            final_state: [batch, heads, k_dim, v_dim]
            all_outputs: [num_chunks, batch, heads, chunk_size, v_dim]
        '''
        def step_fn(carry, chunk_input):
            state = carry
            q_c, k_c, v_c, decay_c = chunk_input

            # Inter-chunk: query the accumulated state
            inter_out = jnp.einsum('bhkd,bhkv->bhdv', q_c, state)

            # Intra-chunk: local attention within the chunk
            intra_out = local_attention(q_c, k_c, v_c, decay_c)

            out = inter_out + intra_out

            # Update state for next chunk
            new_state = state * decay_c[..., -1:, None] + jnp.einsum(
                'bhck,bhcv->bhkv', k_c, v_c
            )

            return new_state, out

        final_state, all_outputs = jax.lax.scan(step_fn, init_state, inputs)
        return final_state, all_outputs

## CORRECT: Reshaping inputs for scan

    # Inputs are [batch, heads, seq_len, dim]
    # Need to reshape to [num_chunks, batch, heads, chunk_size, dim] for scan

    batch, heads, seq_len, dim = x.shape
    chunk_size = 64
    num_chunks = seq_len // chunk_size

    # Reshape: split seq_len into (num_chunks, chunk_size)
    x_chunked = x.reshape(batch, heads, num_chunks, chunk_size, dim)

    # Transpose time axis to LEADING position for scan
    # scan slices along axis 0, so num_chunks must be first
    x_chunked = jnp.transpose(x_chunked, (2, 0, 1, 3, 4))
    # Now: [num_chunks, batch, heads, chunk_size, dim]

    # Pack multiple arrays into a tuple for scan
    scan_inputs = (q_chunked, k_chunked, v_chunked, decay_chunked)

## CORRECT: jax.lax.fori_loop for state-only iteration:

    def iterative_refinement(init_x, num_iters):
        '''State-only loop -- no outputs collected per step.'''
        def body_fn(i, state):
            x = state
            x = x - learning_rate * gradient(x)
            return x

        final_x = jax.lax.fori_loop(0, num_iters, body_fn, init_x)
        return final_x

## scan with auxiliary state (carry multiple values):

    def step_fn(carry, inputs):
        state, running_sum = carry  # Unpack multiple carry values
        x = inputs

        out = state @ x
        new_state = update(state, x)
        new_sum = running_sum + jnp.sum(out)

        return (new_state, new_sum), out  # Pack carry back as tuple

    (final_state, total_sum), outputs = jax.lax.scan(
        step_fn, (init_state, jnp.zeros(())), inputs
    )

## Key gotchas:

1. **scan slices axis 0**: The scanned array's leading dimension is the loop length.
   Transpose your data so the time/chunk axis is first.
2. **Carry must be a pytree**: Use tuples or NamedTuples for multiple carry values.
3. **Static shapes**: All arrays in the scan body must have shapes determinable at
   trace time. No data-dependent shapes inside the body.
4. **scan unroll parameter**: `jax.lax.scan(..., unroll=k)` unrolls k iterations for
   better optimization at the cost of compile time. Default unroll=1.
"""
