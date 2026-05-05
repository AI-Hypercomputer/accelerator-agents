"""
TARGETED JAX PATTERN: Causal Conv1d — Separate Prefill and Decode Functions

APPLICABILITY: This pattern applies ONLY to **causal** convolutions — those used
in autoregressive models, SSMs, and linear attention layers. Identify causal
conv1d by looking for: `conv_state` / rolling state management, output slicing
like `[:, :, :seq_len]` after the conv, or functions named `causal_conv1d`.

DO NOT apply this pattern to standard (non-causal) conv1d layers found in
encoders, classifiers, or non-autoregressive models. For those, translate the
padding directly (e.g., PyTorch `padding="same"` -> JAX `padding="SAME"`,
PyTorch `padding=P` -> JAX `padding=((P, P),)`).

CRITICAL: When this pattern DOES apply, implement causal conv1d as TWO separate
functions, not a single unified function with conditional branching. This gives
clearer semantics, better XLA optimization, and matches the PyTorch source's
separate causal_conv1d_fn and causal_conv1d_update functions.

## WRONG approach (single unified function -- DO NOT DO THIS):

    # WRONG! Single function with conditional branching
    def causal_conv1d(x, weight, bias=None, conv_state=None):
        if conv_state is not None:
            # decode path
            conv_state = jnp.roll(conv_state, -1, axis=-1)
            conv_state = conv_state.at[:, :, -1].set(x[:, :, 0])
            y = jnp.sum(conv_state * weight, axis=-1) + bias
            return jax.nn.silu(y), conv_state
        else:
            # prefill path
            x_padded = jnp.pad(x, ((0,0), (0,0), (weight.shape[-1]-1, 0)))
            y = jax.lax.conv_general_dilated(...)
            return jax.nn.silu(y), None

## CORRECT approach (two separate functions):

    import jax
    import jax.numpy as jnp

    def causal_conv1d(x, weight, bias=None, activation='silu'):
        '''
        Causal conv1d for PREFILL: processes full sequence.

        Args:
            x: [batch, channels, seq_len] input (channels-first)
            weight: [channels, 1, kernel_size] depthwise conv kernel
            bias: [channels] optional bias
            activation: activation function name ('silu' or None)

        Returns:
            y: [batch, channels, seq_len] output
            conv_state: [batch, channels, kernel_size-1] state for subsequent decode
        '''
        batch, channels, seq_len = x.shape
        kernel_size = weight.shape[-1]

        # Depthwise 1D causal convolution: left-only padding prevents
        # future information leakage. Passing the padding tuple directly
        # to conv_general_dilated is cleaner than a separate jnp.pad call.
        y = jax.lax.conv_general_dilated(
            lhs=x,                                  # [B, C, T]
            rhs=weight,                             # [C, 1, K]
            window_strides=(1,),
            padding=((kernel_size - 1, 0),),        # left-only pad
            feature_group_count=channels,
            dimension_numbers=('NCH', 'IOH', 'NCH'),
        )

        if bias is not None:
            y = y + bias[None, :, None]

        if activation == 'silu':
            y = jax.nn.silu(y)

        # Save the last (kernel_size - 1) timesteps as conv state for decode
        conv_state = x[:, :, -(kernel_size - 1):]   # [B, C, K-1]

        return y, conv_state

    def causal_conv1d_update(x_t, conv_state, weight, bias=None, activation='silu'):
        '''
        Causal conv1d for DECODE: processes single timestep.

        Args:
            x_t: [batch, channels] or [batch, channels, 1] single token input
            conv_state: [batch, channels, kernel_size-1] rolling state
            weight: [channels, 1, kernel_size] depthwise conv kernel
            bias: [channels] optional bias
            activation: activation function name ('silu' or None)

        Returns:
            y_t: [batch, channels] output for this timestep
            new_conv_state: [batch, channels, kernel_size-1] updated state
        '''
        if x_t.ndim == 3:
            x_t = x_t.squeeze(-1)  # [B, C]

        # Roll state left: drop oldest, append new input
        new_conv_state = jnp.concatenate(
            [conv_state[:, :, 1:], x_t[:, :, None]], axis=-1
        )  # [B, C, K-1]

        # Full window = [state..., x_t] = new_conv_state padded? No:
        # weight is [C, 1, K], state is [B, C, K-1], we need K values
        full_window = jnp.concatenate(
            [conv_state, x_t[:, :, None]], axis=-1
        )  # [B, C, K]

        # Depthwise multiply-sum (equivalent to conv with kernel_size window)
        weight_squeezed = weight.squeeze(1)  # [C, K]
        y_t = jnp.sum(full_window * weight_squeezed[None, :, :], axis=-1)  # [B, C]

        if bias is not None:
            y_t = y_t + bias

        if activation == 'silu':
            y_t = jax.nn.silu(y_t)

        return y_t, new_conv_state

## Usage in a GatedDeltaNet layer:

    class GatedDeltaNetLayer(nn.Module):
        @nn.compact
        def __call__(self, x, cache=None, decode=False):
            # ... projection ...

            if not decode:
                # Prefill: full sequence convolution
                conv_out, conv_state = causal_conv1d(
                    q_conv_input, self.conv_weight, self.conv_bias
                )
                # ... chunk-parallel delta rule ...
            else:
                # Decode: single-step update
                conv_out, new_conv_state = causal_conv1d_update(
                    q_conv_input, cache.conv_state, self.conv_weight, self.conv_bias
                )
                # ... recurrent delta rule ...

## Why two functions:

1. **XLA optimization**: Two simple functions compile to tighter kernels than one
   function with dynamic branching.
2. **Clarity**: Prefill processes [B, C, T], decode processes [B, C, 1]. Different
   shapes, different algorithms, different code.
3. **Matches PyTorch**: The source has separate `causal_conv1d_fn` and
   `causal_conv1d_update` functions.
4. **Cache management**: Prefill returns initial conv_state. Decode takes and
   returns updated conv_state. Clean separation of concerns.
"""
