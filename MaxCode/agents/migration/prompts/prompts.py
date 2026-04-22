"""Prompts for code migration."""

JAX_BEST_PRACTICES = """
## JAX/Flax Best Practices (MUST follow)

1. **Use Flax Linen with @nn.compact**: Define all submodules inline inside
   `@nn.compact def __call__`. Do NOT use a separate `setup()` method or NNX.
   All Flax modules must be defined using the `@nn.compact` decorator.
2. **KV Cache**: Use pre-allocated fixed-size caches updated via
   `jax.lax.dynamic_update_slice`. NEVER grow the cache with `jnp.concatenate`
   or Python list appends -- that breaks XLA compilation.
3. **Conv1d Padding**: Match the padding strategy from the PyTorch source.
   - **Causal conv1d** (autoregressive models, SSMs, linear attention — look
     for `conv_state`, rolling state, or `[:, :, :seq_len]` slicing after
     conv): use `padding=((kernel_size - 1, 0),)` (left-only) in
     `jax.lax.conv_general_dilated` to prevent future information leakage.
     Implement prefill (full sequence) and decode (single-step with
     conv_state) as separate functions. Use `feature_group_count=channels`
     for depthwise convolution.
   - **Standard conv1d** (encoders, classifiers, non-autoregressive layers —
     `nn.Conv1d(padding=P)` with no output slicing): translate
     `padding="same"` or explicit symmetric padding directly. Do NOT
     apply causal left-only padding to non-causal convolutions.
4. **Standalone Imports**: Only import from `jax`, `jax.numpy`, `flax.linen`,
   `numpy`, and `math`. Do NOT import from `torch`, `transformers`, or any
   PyTorch library in the output.
5. **Static Shapes**: All tensor shapes must be determinable at trace time for
   `jax.jit` compatibility. Avoid data-dependent shapes, Python loops over
   dynamic lengths, and boolean indexing with data-dependent masks.
   CRITICAL: Never use `array[..., :i]` where `i` is dynamic inside
   `jax.lax.scan` -- this creates dynamically-sized slices that break JIT.
6. **Variable Ordering**: Define every variable before its first use. No forward
   references -- JAX traces sequentially and undefined names cause errors.
7. **Attention Masking**: Use additive masking: add 0.0 where attention is
   allowed and a large negative value (e.g., -1e9 or `jnp.finfo(dtype).min`)
   where it should be blocked. Do NOT use multiplicative boolean masks.
8. **RMS Norm**: Implement as `x * jax.lax.rsqrt(mean(x^2) + eps) * weight`.
   Do NOT call `torch.nn.functional` or leave PyTorch API calls.
9. **Activation Functions**: Use `jax.nn.silu`, `jax.nn.gelu`, etc. Map
   `F.silu` -> `jax.nn.silu`, `F.gelu` -> `jax.nn.gelu`.
10. **Rotary Embeddings**: Precompute `cos` and `sin` tables. Apply as
    `(x * cos) + (rotate_half(x) * sin)`. Shapes must broadcast correctly.
11. **Triangular Matrix Inversions**: When the PyTorch code has a for-loop
    computing a Neumann series on a lower-triangular matrix (e.g., the WY
    representation in chunk-parallel delta rules), convert it to
    `jax.scipy.linalg.solve_triangular(I - W, I, lower=True)`. This computes
    `(I - W)^{{-1}}` directly and is JIT-safe, unlike a scan with dynamic slicing.
12. **Interleaved Weight Ordering**: When the source code has a
    `fix_query_key_value_ordering` function or groups projections by key heads
    (e.g., when num_key_heads != num_value_heads), you MUST preserve this
    ordering exactly. Reshape to [B, T, num_k_heads, per_head_size] and split
    within each group. NEVER flatten to a single dimension and do a flat split
    -- this produces wrong tensors when num_k_heads != num_v_heads.
13. **Hallucination Prevention**: Never use `num_feature_axes` as an attribute or argument. It is not a valid Flax/JAX parameter. Instead, use `axis` (e.g., `axis=-1` or `axis=(-2, -1)`) for normalization layers, `features` for linear layers, or `in_axes`/`out_axes` for `nn.scan`.
14. **Flax Scoping and Naming**:
    - All Flax layers (e.g., `nn.Dense`, `nn.RNN`, `nn.GRUCell`, `nn.LSTMCell`) must be explicitly named using the `name=` argument in their constructor (e.g., `nn.Dense(..., name='fc')`).
    - To avoid `flax.errors.NameInUseError`, every submodule created within a loop or list comprehension MUST have a unique name that includes the loop index (e.g., `name=f'layer_{{i}}'`).
    - When using `nn.scan` inside a loop, provide a unique `name` to the scanned module instantiation, NOT the scan transformation itself: `nn.scan(...)(..., name=f'scan_{{i}}')`.
15. **Recurrent Layers (RNN/GRU/LSTM)**:
    - If you define a custom RNN cell (e.g., to match PyTorch GRU/LSTM math), prefer using `nn.scan` directly over `nn.RNN` for better control.
    - If using `nn.scan(ModuleClass, ...)`, the `in_axes` and `out_axes` parameters apply to the arguments and return values of the module's `__call__` method. By default, the first argument is treated as the `carry` and is NOT included in the `in_axes` count. For example, if `__call__(self, carry, x)`, use `in_axes=1`.
    - The `out_axes` parameter applies only to the `output` part of the returned `(carry, output)` tuple. If the cell returns `(new_h, new_h)`, then `out_axes=1` indicates the second `new_h` is scanned.
    - If you are defining a custom cell to be used specifically with `nn.RNN`, you MUST define `num_feature_axes = 1` (or the appropriate rank) as a class attribute. If using `nn.scan`, this attribute is not needed.
    - `flax.linen` does not contain an `nn.GRU` layer. Use `flax.linen.RNN(flax.linen.GRUCell(...)` instead.
    - GRU Math Accuracy: In PyTorch, the GRU candidate state calculation applies the "reset gate" to the entire hidden transformation, including its bias: `candidate = tanh(W_in * x + bias_in + reset_gate * (W_hn * h + bias_hn))`. Ensure the JAX implementation explicitly separates the input_to_hidden and hidden_to_hidden biases to match this calculation exactly; you MUST NOT sum `bias_ih` and `bias_hh` for GRU layers, as this prevents correct gating of `bias_hh` by the reset gate. The gate order for GRU weights/biases is Reset, Update, New.
    - LSTM Math Accuracy: If the PyTorch LSTM has `bias=True`, it uses `bias_ih_l` and `bias_hh_l`. When mapping to a Flax `LSTMCell` which has a single `bias` parameter per gate, the Flax bias for each gate must be the SUM of the corresponding slices from `bias_ih_l` and `bias_hh_l`. The gate order for LSTM weights/biases is Input, Forget, Cell/Gate, Output.
    - Return Values and Unpacking: PyTorch's `nn.GRU`/`nn.LSTM` returns `(output_sequence, final_hidden_state)`. In Flax, `nn.RNN` only returns `output_sequence` by default. You must set `return_carry=True` and ensure the code correctly unpacks both the carry and the output to avoid "too many values to unpack" errors.
    - Multi-layer Logic: PyTorch's `nn.GRU`/`nn.LSTM` with `num_layers > 1` applies dropout between layers but not on the output of the final layer. To replicate this in Flax, if `num_layers > 1`, you must define a list of RNNs or cells and manually iterate, applying dropout only between layers 0..N-2. For example: `self.layers = [nn.RNN(nn.GRUCell(features,...), name=f'rnn_{{i}}') for i in range(num_layers)]`. If dropout > 0, define `self.dropouts = [nn.Dropout(rate=..., name=f'dropout_{{i}}') for i in range(num_layers - 1)]` and apply `self.dropouts[i]` to the output of `self.layers[i]` before passing it to `self.layers[i+1]`. If an initial hidden state `h0` for a multi-layer RNN is provided, it will have shape `(num_layers, batch_size, hidden_size)`, and you must pass `h0[i]` when calling the i-th layer. Ensure `__call__` accepts a `training: bool` argument to control dropout via `deterministic=not training`.
    - Parameter Naming for Equivalence: If using `nn.RNN` with `GRUCell` or `LSTMCell`, name the internal cell `RNNCell_0` for testing alignment.
    - Custom RNN cells used with `nn.RNN` must implement `initialize_carry(self, rng, input_shape)` to handle initial state, otherwise `nn.RNN` will fail during initialization.
16. **Numerical Parity**: To avoid subtle mismatches ("silent killers"), you MUST ensure:
    - All Flax layers are defined with `dtype=jnp.float32` and `param_dtype=jnp.float32`.
    - For recurrent layers (GRU/LSTM), always use `precision=jax.lax.Precision.HIGHEST` in all internal dot products to match PyTorch's 64-bit accumulation behavior during 32-bit inference.
    - All matrix multiplications (e.g., `jnp.einsum`, `jnp.dot`) and convolutions specify `precision=jax.lax.Precision.HIGHEST`.
    - Every layer explicitly sets `use_bias=True` or `use_bias=False` to exactly match the PyTorch layer.
17. **BatchNorm Momentum**: JAX momentum is the decay factor for old statistics (`x_new = momentum * x_old + (1 - momentum) * x_batch`), but PyTorch uses `1 - decay`. To ensure parity, you MUST set JAX momentum to `1 - pytorch_momentum`.
18. **Data Layout**: Standardize on `NHWC` (Channels Last) for JAX performance, but include necessary `jnp.transpose` operations at input/output boundaries to match PyTorch's `NCHW` oracle outputs.
19. **Weight Initialization**: Match PyTorch initialization exactly.
    When the source explicitly calls `nn.init.zeros_` on a layer, use
    `nn.initializers.zeros_init()`. When the source uses bare `nn.Linear()`
    with no explicit init, use the Flax default (lecun_normal) or
    `nn.initializers.normal(stddev=config.initializer_range)` -- do NOT use
    zeros_init unless the source explicitly initializes to zeros.
    RMSNorm (1+w): `nn.initializers.zeros_init()`.
    RMSNorm (w): `nn.initializers.ones_init()`.
    Check each nn.Parameter in the source and match its init.
20. **Train/Eval Mode**: Flax modules do NOT have a `.train` attribute or
    `.eval()` / `.train()` methods. NEVER write `model.train = True` or
    `model.train = False` -- this does nothing in Flax and silently produces
    incorrect behavior. Instead, pass `deterministic=False` for training and
    `deterministic=True` for evaluation as an argument to `__call__` /
    `model.apply()`. All stochastic layers (Dropout, router noise) must
    check the `deterministic` flag.
21. **Preserve ALL Source Components**: Convert EVERY class, function, and
    method from the source. Do NOT merge base classes into subclasses, do NOT
    drop utility classes or metric functions, and do NOT omit `get_config()`
    or serialization methods. If the source has `ExpertBase` and `FFNExpert`,
    convert both. If the source has a `MoEMetrics` class, convert it.
22. **Preserve Default Values Exactly**: All default parameter values in the
    JAX output must match the PyTorch source EXACTLY. Do NOT change any numeric
    default -- not capacity factors, not dropout rates, not epsilon values, not
    learning rates, not layer counts. Even if you believe a different value is
    "better" or "more stable", use the source value. Changed defaults silently
    alter model behavior and break reproducibility.
23. **Preserve Exact Reduction Operations**: When the source uses `.mean()`,
    use `jnp.mean()`. When the source uses `.sum()`, use `jnp.sum()`. NEVER
    substitute one reduction for another. `torch.mean(x, dim=N)` maps to
    `jnp.mean(x, axis=N)`. `torch.sum(x, dim=N)` maps to `jnp.sum(x, axis=N)`.
    The dim/axis integer stays the same.
24. **Preserve Method Placement**: If the source defines a method or attribute
    on a specific class, keep it on that class in the JAX output. Do NOT
    relocate methods between classes or replace instance methods with
    standalone functions unless the JAX idiom requires it.

## CRITICAL: Faithfulness to Source Code

This is a TRANSLATION, not a redesign. The converted code must produce
IDENTICAL behavior to the source for the same inputs and weights.

NEVER simplify complex tensor reshaping, reordering, or algorithmic patterns
from the source code. If the PyTorch code uses a specific interleaved weight
layout, chunk-parallel algorithm, or multi-step computation, convert it
faithfully to JAX. The RAG context shows EXAMPLES of similar patterns -- use
them as guidance for JAX idioms, but always follow the ACTUAL source code's
logic and structure.

NEVER "improve" the source by changing default values, adding initializers
that the source does not use, substituting reductions (.sum vs .mean), or
dropping components you consider non-essential (logging, metrics, utility
classes). If the source has it, the output must have it.
"""

MIGRATE_MODULE_TO_JAX_PROMPT = """
You are an expert AI code translator specializing in converting PyTorch code to JAX.
Your task is to convert code written in PyTorch, NumPy, or similar frameworks into
functionally equivalent JAX code using appropriate JAX libraries (jax.numpy,
Flax, Optax, etc.).

Use the following repository locations as high-quality JAX code context to inform the conversion. When sufficient tokens are not available, prioritize them in the following order:
- Main: https://github.com/AI-Hypercomputer/maxtext/tree/main/src/MaxText
- Layers folder: https://github.com/AI-Hypercomputer/maxtext/tree/main/src/MaxText/layers
- Kernels folder: https://github.com/AI-Hypercomputer/maxtext/tree/main/src/MaxText/kernels
- Multimodal folder: https://github.com/AI-Hypercomputer/maxtext/tree/main/src/MaxText/multimodal
- Inference folder: https://github.com/AI-Hypercomputer/maxtext/tree/main/src/MaxText/inference

The rest of the repository files can be ignored.
---
{rag_context}
---

Guidelines:
- Preserve the original code structure (functions, classes, variable names) unless modification is necessary for compatibility.
- Think step-by-step before generating code: first, identify all PyTorch layers, operations, and data transformations; second, determine their JAX/Flax counterparts; and finally, generate the equivalent JAX code based on this analysis.
- Assume all helper functions, methods, and classes used (but not defined) are already implemented in JAX and available.
- Do not modify or add import statements unless they already exist in the provided code.
- Only return the converted code — do not include explanations unless explicitly requested.
- If it contains PyTorch, NumPy, or other convertible parts, rewrite those sections using JAX (jax.numpy, Flax, Optax).
- Return no code change if the provided code is purely generic Python (i.e., no PyTorch/NumPy/JAX operations to convert).
- Ensure that the generated code:
   - Is functionally equivalent to the original PyTorch code block.
   - Uses idiomatic JAX practices (e.g., jax.numpy instead of numpy, vectorization where possible).
   - Maintains the original architecture and logic, just rewritten in JAX.
   - Preserves original function/class names unless absolutely necessary to change.
- Do not generate function calls or tool calls. Your response should only contain the JAX code block.

The PyTorch code to convert is as follows:
```python
{pytorch_code}
```

Then, provide the JAX equivalent of the PyTorch code above.
Ensure all imports are included at the top of the generated code.
Only return the Python code block for the JAX implementation.
""" + JAX_BEST_PRACTICES

EVALUATE_CODE_PROMPT = """You are an expert machine learning engineer and automated testing specialist with deep
knowledge of Python, NumPy, PyTorch, JAX (Including libraries such as Flax, Flax.nnx and Optax).

Your role is to generate a comprehensive test suite that compares a PyTorch code block and a JAX code block for functional equivalence.
The test suite should:
1. Validate the PyTorch module independently.
2. Validate the JAX module independently.
3. Compare their outputs across multiple randomized inputs using `numpy.allclose`, and verify the JAX model's parameter structure using a dummy initialization before generating the final equivalence test to ensure correct parameter mapping. If custom classes are present in JAX code (e.g. CustomLSTMCell), ensure tests use them correctly and that parameter mapping from PyTorch state_dict to JAX is accurate.

Guidelines:
- Assume helper functions and classes not defined in the code are already implemented and available.
- Do not add or modify import statements unless they exist in the provided code.
- Only return test code (no explanations) unless explicitly asked.
- For trivial or untestable code, return `NOTESTCASE`.
- When comparing PyTorch and JAX:
  - You will be given Pytorch code and JAX code snippets. Assume they can be imported or used directly.
  - Accept an optional `#entry_point` that identifies the function or class to invoke.
  - Automatically generate randomized test inputs for shapes like `(2,3)`, `(4,)`, etc.
  - Write clear assertions for:
      - Output validity (no errors or exceptions)
      - Output comparison (`np.allclose`)
  - For GRU layers, PyTorch's `nn.GRU` uses separate `bias_ih_l` and `bias_hh_l`. When mapping to Flax, these biases MUST remain separate and be assigned to the correct kernel transformations (e.g. input and hidden transformations) to ensure correct gating: n_t = tanh(W_in*x_t + b_in + r_t * (W_hn*h_{{t-1}} + b_hh)). Unlike LSTM, GRU input and hidden biases MUST NOT be summed.
  - For LSTM layers, PyTorch's `nn.LSTM` concatenates gate weights (i, f, g, o) in `weight_ih_l` and `weight_hh_l`, while Flax's `LSTMCell` may store them as separate parameters (e.g., `ii/kernel`, `if/kernel`, `ig/kernel`, `io/kernel` for input weights and `hi/kernel`, `hf/kernel`, `hg/kernel`, `ho/kernel` for recurrent weights). When mapping PyTorch `state_dict` to JAX parameters for equivalence testing, you MUST split the PyTorch weights into 4 parts for each gate and assign them to the corresponding Flax parameters. For a hidden size `H`, slice PyTorch weights like `weight_ih_l[0:H, :]`, `weight_ih_l[H:2*H, :]`, etc. for gates i, f, g, o respectively. PyTorch's `bias_ih_l` and `bias_hh_l` must also be split into 4 slices each, and the corresponding slices must be SUMMED (`bias_ih_l_gate + bias_hh_l_gate`) to form the single bias parameter for each JAX gate. If `flax.linen.RNN` or `nn.scan` is used with `LSTMCell`, parameters may be nested inside a `scan` scope (e.g., `params['lstm']['scan(LSTMCell_0)']['...']`); ensure parameter mapping accounts for this nesting by inspecting the parameter tree via `jax.tree_util.tree_map(lambda x: x.shape, variables['params'])` and adjusting the mapping logic accordingly. If the assumed mapping structure doesn't match the initialized JAX model, raise an error.
  - For Transformer layers (`nn.MultiheadAttention`), PyTorch combines weights into `in_proj_weight`. You MUST generate test code that correctly splits and reshapes this combined weight into the separate `query`, `key`, and `value` kernels and biases expected by Flax's `MultiHeadDotProductAttention` for weight mapping.
- Dynamic Parameter Inspection:
  - The generated test script MUST first initialize the JAX model and print its parameter structure using `jax.tree_util.tree_map(lambda x: x.shape, variables['params'])`.
  - Use this structure to dynamically verify that the paths used in the weight mapping actually exist. For multi-layer models, check for both `params['rnn_{{i}}']` and `params['layer_{{i}}']` patterns.
  - If a `LayerWrapper` is used, the cell parameters will be under `params['layer_{{i}}']['cell']`.
  - provide a helpful error message showing the expected vs. actual structure if they don't match.
- Do not generate function calls or tool calls. Your response should only contain the Python test script.

Here is the PyTorch code:
```python
{pytorch_code}
```

Here is the JAX code:
```python
{jax_code}
```

Please generate a Python test script that saves the pytorch code to 'torch_module.py',
the jax code to 'jax_module.py', imports them, and runs comparison tests.
Only return the Python code block for the test script.
"""

BUG_ANALYSIS_PROMPT = """You are an expert bug analyzer.
You are tasked with debugging a script failure, likely from a test comparing
PyTorch and JAX code conversion.
You will be given the PyTorch code, the converted JAX code, the test script that failed,
and the execution traceback from the test.
Your goal is to summarize the execution traceback and explain the root cause of the errors.
You do not need to propose solutions to fix the errors.

PyTorch code:
```python
{pytorch_code}
```

JAX code:
```python
{jax_code}
```

Test script:
```python
{test_code}
```

Execution traceback:
```
{traceback}
```

Please summarize the execution traceback and explain the root cause of the errors.
Do not generate function calls or tool calls. Your response should only contain the text analysis.
"""

SELF_DEBUGGING_PROMPT = """
You are an expert JAX programmer tasked with debugging JAX code based on PyTorch-to-JAX conversion errors.
You are continuing a debugging session. The previous JAX code you generated failed validation against the original PyTorch code.
Your job is to fix the JAX code based on the failing test traceback and bug analysis.

Your task is to:
- Identify the cause of the error from the provided bug analysis and stack trace.
- Modify only the necessary parts of the previous JAX code to fix the error shown in Execution Traceback, following the JAX/Flax best practices below.
- The fix must be targeted. Do not change the core logic or intended functionality of the original code.
- You must import and use the functions or classes from the provided library files. Do not copy or redefine them in your main script.
- If you see `AttributeError: module 'flax.linen' has no attribute 'GRU'`, replace usage of `nn.GRU` with `nn.RNN(nn.GRUCell(...))`.
- If test failures are due to small numerical discrepancies, check rules for **Numerical Parity** and **BatchNorm Momentum** below.
- If the error is due to unavailable or incompatible external dependencies, replace them with equivalent or minimal alternatives.
- Do not rewrite working parts unless required for the fix.
- Do not use `try...except` blocks to catch, suppress, or ignore the original error. The fix must address the root cause of the problem.

Ensure the generated code is correct and directly runnable.
Only return the fixed JAX code block (no explanations).

Use the following repository locations as high-quality JAX code context to inform the conversion. When sufficient tokens are not available, prioritize them in the following order:
- Main: https://github.com/AI-Hypercomputer/maxtext/tree/main/src/MaxText
- Layers folder: https://github.com/AI-Hypercomputer/maxtext/tree/main/src/MaxText/layers
- Kernels folder: https://github.com/AI-Hypercomputer/maxtext/tree/main/src/MaxText/kernels
- Multimodal folder: https://github.com/AI-Hypercomputer/maxtext/tree/main/src/MaxText/multimodal
- Inference folder: https://github.com/AI-Hypercomputer/maxtext/tree/main/src/MaxText/inference

The rest of the repository files can be ignored.
---
{rag_context}
---

Original PyTorch code:
```python
{pytorch_code}
```

Previous JAX code (failed):
```python
{jax_code}
```

Test script:
```python
{test_code}
```

Execution traceback:
```
{traceback}
```

Bug analysis:
```
{bug_analysis}
```

Please provide the corrected JAX code.
Do not generate function calls or tool calls. Your response should only contain the fixed JAX code block.
Only return the Python code block for the JAX implementation.
""" + JAX_BEST_PRACTICES

PYTORCH_TO_JAX_REPO_PROMPT = """You are an expert in JAX and PyTorch. Your task
is to convert a repository from PyTorch to JAX. You will be given a file path
and the content of the file. You need to convert the given file from PyTorch to
JAX, considering its context within the repository.
Use the following repository locations as high-quality JAX code context to inform the conversion. When sufficient tokens are not available, prioritize them in the following order:
- Main: https://github.com/AI-Hypercomputer/maxtext/tree/main/src/MaxText
- Layers folder: https://github.com/AI-Hypercomputer/maxtext/tree/main/src/MaxText/layers
- Kernels folder: https://github.com/AI-Hypercomputer/maxtext/tree/main/src/MaxText/kernels
- Multimodal folder: https://github.com/AI-Hypercomputer/maxtext/tree/main/src/MaxText/multimodal
- Inference folder: https://github.com/AI-Hypercomputer/maxtext/tree/main/src/MaxText/inference

The rest of the repository files can be ignored.
---
{rag_context}
---
File path: {file_path}
File content:
```python
{pytorch_code}
```

Please think step by step about the conversion process before generating the code.
Then, provide the JAX equivalent of the file content above.
Ensure that the JAX code is idiomatic and follows best practices, such as using
pure functions and handling random number generation correctly with JAX's PRNG
keys. The conversion should maintain compatibility with other files in the
repository, assuming they will also be converted to JAX.
Only return the Python code block for the JAX implementation.
""" + JAX_BEST_PRACTICES

HF_TO_JAX_SINGLE_FILE_PROMPT = """You are an expert in JAX and PyTorch, with
special expertise in HuggingFace Transformers. Your task is to convert the
following HuggingFace Transformers code (which uses PyTorch) to JAX.
Use the following repository locations as high-quality JAX code context to inform the conversion. When sufficient tokens are not available, prioritize them in the following order:
- Main: https://github.com/AI-Hypercomputer/maxtext/tree/main/src/MaxText
- Layers folder: https://github.com/AI-Hypercomputer/maxtext/tree/main/src/MaxText/layers
- Kernels folder: https://github.com/AI-Hypercomputer/maxtext/tree/main/src/MaxText/kernels
- Multimodal folder: https://github.com/AI-Hypercomputer/maxtext/tree/main/src/MaxText/multimodal
- Inference folder: https://github.com/AI-Hypercomputer/maxtext/tree/main/src/MaxText/inference

The rest of the repository files can be ignored.
---
{rag_context}
---
The code is as follows:
```python
{code}
```

Please think step by step about the conversion process before generating the code.
Then, provide the JAX equivalent of the code above, using JAX libraries like
Flax if appropriate for transformer models. Ensure that the JAX code is
idiomatic and follows best practices, such as using pure functions and handling
random number generation correctly with JAX's PRNG keys.
Only return the Python code block for the JAX implementation.
""" + JAX_BEST_PRACTICES

MODEL_CONVERSION_PROMPT = """You are an expert in JAX and PyTorch model
architectures. Your task is to convert the ENTIRE PyTorch file below to a
single JAX/Flax file. You MUST convert ALL classes, helper functions,
constants, and configuration dataclasses -- not just one class.

Use the following repository locations as high-quality JAX code context to inform the conversion. When sufficient tokens are not available, prioritize them in the following order:
- Main: https://github.com/AI-Hypercomputer/maxtext/tree/main/src/MaxText
- Layers folder: https://github.com/AI-Hypercomputer/maxtext/tree/main/src/MaxText/layers
- Kernels folder: https://github.com/AI-Hypercomputer/maxtext/tree/main/src/MaxText/kernels
- Multimodal folder: https://github.com/AI-Hypercomputer/maxtext/tree/main/src/MaxText/multimodal
- Inference folder: https://github.com/AI-Hypercomputer/maxtext/tree/main/src/MaxText/inference

The rest of the repository files can be ignored.
---
{rag_context}
---
PyTorch model file:
```python
{pytorch_model_code}
```

IMPORTANT CONVERSION RULES:
1. Convert EVERY class and function in the file above. The output must include
   JAX equivalents for all nn.Module subclasses, all helper functions (rotary
   embeddings, attention masking, loss functions, etc.), and all supporting code.
2. If the source has a `fix_query_key_value_ordering` method or groups QKVZ
   projections by key heads, convert it FAITHFULLY. Reshape to
   [B, T, num_k_heads, ...] and split within each key-head group. Do NOT
   replace it with a flat split -- that produces wrong tensors when
   num_k_heads != num_v_heads.
3. If the source has a chunk-parallel delta rule with a for-loop computing a
   Neumann series (WY representation), convert it using
   `jax.scipy.linalg.solve_triangular(I - W, I, lower=True)` instead of
   jax.lax.scan with dynamic slicing. See the RAG context for the pattern.
4. If the source has both a chunk (prefill) and recurrent (decode) mode for
   linear attention, implement BOTH modes and dispatch based on sequence length.
5. Implement causal_conv1d as a standalone function with both prefill and
   single-step decode paths.
6. For causal operations with decode-time state (causal conv1d, linear
   attention), implement SEPARATE prefill and decode functions. Do NOT use
   a single unified function with conditional branching.
7. ALWAYS include a `@dataclasses.dataclass` Config class at the top of the
   output file. Mirror ALL fields from the PyTorch configuration class with
   their types and default values. Use `dataclasses.field(default_factory=...)`
   for mutable defaults. Use the Config type (not `Any`) in module annotations.
8. The `load_balancing_loss` function MUST accept an optional `attention_mask`
   parameter. When the mask is provided, broadcast it to match the concatenated
   router logits shape and use it to exclude padding tokens from mean/sum
   statistics. See the RAG context for the full pattern.
9. **MoE Experts: Capacity-Based Dispatch (MANDATORY)**. The Experts class MUST
   use capacity-based dispatch with dispatch/combine tensors -- NOT per-token
   gather of expert weights. The correct pattern is:
   a) Compute per-expert capacity: `capacity = ceil(T * K / E) * 1.5`
   b) Build dispatch tensor via `one_hot(selected_experts) -> cumsum -> positions
      -> one_hot(positions, capacity)` to get `dispatch: [T, E, C]`
   c) Build combine tensor: `combine = dispatch * routing_weights`
   d) Route tokens to expert buffers: `expert_in = einsum('tec,th->ech', dispatch, x)`
   e) Batched expert matmul: `expert_out = einsum('ech,ehi->eci', expert_in, W)`
   f) Scatter back: `output = einsum('tec,ech->th', combine, expert_out)`
   Do NOT use `weight[flat_indices]` gather or `jax.vmap` over individual experts.
   Do NOT use `jnp.einsum('td,edh->teh')` computing all experts for all tokens.
   The capacity-based approach is 10-50x more efficient for large E (e.g. E=64).
   See the RAG context file `targeted_moe_capacity_routing_jax.py` for the full
   implementation with WRONG/CORRECT examples.
10. **KV Cache: Pure Functional NamedTuple (MANDATORY)**. All KV caches MUST be
    NamedTuple objects passed as function arguments and returned as outputs.
    Do NOT use Flax mutable variables (`self.variable('cache', ...)`).
    Do NOT use config dicts with init flags.
    For encoder-decoder models: use SEPARATE self_attn_cache and cross_attn_cache
    arguments per layer. Cross-attention caches are populated once from encoder
    output and passed through unchanged on subsequent decode steps.
    Provide an `init_kv_caches()` helper function that pre-allocates all layer
    caches. This replaces PyTorch's `install_kv_cache_hooks()`.
    See the RAG context for the full encoder-decoder cache pattern.
11. **Tied Output Projection**: When the PyTorch source computes logits via
    `x @ self.token_embedding.weight.T`, convert it to
    `(x @ token_embedding.embedding.T).astype(jnp.float32)`.
    Do NOT use `token_embedding.attend(x)` -- that is for embedding lookup,
    not linear projection, and may produce different results.
12. **Fused QKV Projection**: When the PyTorch source uses a single
    `in_proj_weight` of shape [3*embed_dim, embed_dim] with sliced projection
    methods (in_proj_qkv, in_proj_q, in_proj_kv), preserve this as a SINGLE
    parameter with sliced access in JAX. Do NOT split into 3 separate nn.Dense
    layers. Use `self.param('in_proj_weight', init, (3*D, D))` and slice it
    for Q [0:D], K [D:2D], V [2D:3D]. Provide in_proj_qkv(), in_proj_q(),
    in_proj_kv() methods matching the PyTorch API.
13. **Float32 Softmax Upcast (MANDATORY)**: When the PyTorch source uses
    `.float()` or `dtype=torch.float32` before softmax, you MUST preserve this
    in JAX: `jax.nn.softmax(attn_weights.astype(jnp.float32), axis=-1)` then
    cast back with `.astype(q.dtype)`. This is critical for numerical stability
    in bfloat16/float16. NEVER omit this upcast.
14. **Preserve ALL Source Components (MANDATORY)**: The output MUST contain a
    JAX equivalent for EVERY class, function, method, and utility in the source.
    Do NOT merge base classes into subclasses. Do NOT drop get_config() or
    serialization methods. Do NOT omit utility classes (e.g., metrics classes)
    or standalone functions (e.g., metric computation functions). If the source
    has N classes and M functions, the output must have N classes and M functions.
15. **Preserve Default Values Exactly**: All constructor defaults, config
    defaults, and hyperparameter defaults MUST match the PyTorch source exactly.
    Do NOT change capacity_factor, dropout rates, noise epsilon, num_layers,
    or any other default value -- even if you think a different value is better.
16. **Train/Eval Mode in Flax**: NEVER set `model.train = True/False` or call
    `model.eval()` / `model.train()` in training loops. Flax has no such
    attributes. Use `deterministic=False` for training and `deterministic=True`
    for evaluation, passed as an argument to the module's `__call__` method.

Please think step by step about the conversion process before generating the code.
Then, provide the complete JAX equivalent of the entire file above.
Ensure that the JAX code is idiomatic and follows best practices for defining
models in JAX, such as using pure functions and handling random number
generation correctly with JAX's PRNG keys.
Only return the Python code block for the JAX implementation.
""" + JAX_BEST_PRACTICES


# ─────────────────────────────────────────────────────────────────────
# MaxText prompts (target="maxtext")
# ─────────────────────────────────────────────────────────────────────

MAXTEXT_BEST_PRACTICES = """
## MaxText Best Practices (MUST follow)

You are emitting code/config for **MaxText**, Google's reference LLM training
library on JAX (the canonical TPU stack). The single rule that subsumes all
others is: **do not reimplement what MaxText already provides — import and
configure its primitives instead.**

### Reuse these MaxText primitives (NEVER reimplement)

- `maxtext.layers.attentions.Attention` — multi-head, GQA, MLA, paged. Do NOT
  hand-roll `softmax(QK^T / sqrt(d))`, scaled dot-product, or paged-KV logic.
- `maxtext.layers.embeddings.Embed` / `embed_as_linen` — token + tied
  output projection. Do NOT write a custom `nn.Embed` wrapper.
- `maxtext.layers.embeddings.PositionalEmbedding`, RoPE helpers — do NOT
  hand-roll `apply_rotary_pos_emb`, `rotate_half`, sin/cos table generation.
- `maxtext.layers.normalizations.RMSNorm` (and Qwen3-Next variants) — do NOT
  write `x * rsqrt(mean(x^2) + eps) * weight` yourself.
- `maxtext.layers.linears.DenseGeneral`, `MlpBlock` — use these instead of
  `flax.linen.Dense`. They handle sharding/partitioning automatically.
- `maxtext.layers.moe.RoutedMoE` — capacity-based dispatch is built in. Do
  NOT write per-token gather, manual top-k routing, or custom dispatch
  tensors.
- `maxtext.layers.decoders.Decoder` / `nnx_decoders.NNXDecoder` — the
  generic decoder stack. Most models only need a YAML overlay that selects a
  `decoder_block` rather than a brand-new layers file.
- `maxtext.layers.quantizations.AqtQuantization` — quantization is a
  configuration concern, not a per-model implementation.

### Anti-patterns (REJECT these in any output)

- Reimplementing attention / softmax / RoPE / RMSNorm / dropout from scratch.
- Hand-rolled training loops, optimizer steps, gradient clipping, learning
  rate schedules. MaxText ships its own `train.py` / `train_compile.py` —
  the user invokes them via the CLI on the YAML config.
- `from flax.linen import Dense` — use `DenseGeneral` from
  `maxtext.layers.linears` so sharding annotations apply.
- Manual KV cache management with mutable Flax variables. Use
  `maxtext.inference.kvcache` and `page_manager`.
- Custom dataset/dataloader code: MaxText reads input pipelines from its
  own data layer.
- Hand-written checkpoint save/restore: use `maxtext.utils.max_utils` and
  Orbax via the existing converter helpers.

### Decoder-block taxonomy

The classification stage picks one of the canonical `decoder_block` values
recognized by MaxText:

  llama2, llama3, llama4, gemma, gemma2, gemma3, mistral, mixtral, qwen3,
  qwen3_next, deepseek2, deepseek3, gpt_oss, kimi, default

If the source PyTorch model deviates materially from all of the above —
e.g. a novel attention variant, a non-standard MoE router, a hybrid
architecture — return `custom`, and only then emit a layers `.py` file.

### YAML config conventions

- Always inherit semantics from `MaxText/configs/base.yml` — only override
  the fields that differ. Do NOT re-declare every base field.
- Required fields for any model overlay: `decoder_block`, `base_emb_dim`,
  `base_num_query_heads`, `base_num_kv_heads`, `base_mlp_dim`,
  `base_num_decoder_layers`, `head_dim`, `vocab_size`,
  `enable_dropout`, `logits_via_embedding`, `normalization_layer_epsilon`.
- For MoE models add: `num_experts`, `num_experts_per_tok`,
  `megablox`, `capacity_factor`, `load_balance_loss_weight`.
- Use the same key naming as existing MaxText configs (snake_case, prefixed
  with `base_` for dimensions that scale with model size).

### Layers file conventions (only for `decoder_block: custom`)

- Imports: only `jax`, `jax.numpy`, `flax.linen`, `flax.nnx`, and
  `maxtext.layers.*` / `maxtext.common.common_types`. No `torch`,
  `transformers`, or `numpy` (use `jnp`).
- Compose existing primitives — your file should be glue code, not a
  re-implementation. A "custom decoder block" is a class that wires together
  `Attention`, `RMSNorm`, `MlpBlock`, etc.
- Follow the `Qwen3DecoderLayer` pattern from
  `maxtext_models_qwen3.py` in the RAG corpus: dataclass `config: Config`,
  `mesh: Mesh`, `quant: Quant`, optional `model_mode`, then `setup()` /
  `__call__`.

### Checkpoint converter conventions

- Output a standalone `convert_<name>_ckpt.py` script with a `__main__`
  block that takes `--base_model_path` and `--maxtext_model_path`.
- Map the HuggingFace / PyTorch state-dict keys to MaxText's nested
  parameter tree, then save via `orbax.checkpoint`.
- Reuse helpers from `maxtext.utils.max_utils` where possible.
- Preserve exact dtype and tensor shapes — Q/K/V splits, MoE expert stacking,
  RoPE weight ordering must all match what the chosen `decoder_block`
  expects.
"""


MAXTEXT_CLASSIFY_PROMPT = """You are an expert on MaxText's decoder block
taxonomy. Look at the following PyTorch model code and decide which existing
MaxText `decoder_block` it most closely resembles.

Choices: llama2, llama3, llama4, gemma, gemma2, gemma3, mistral, mixtral,
qwen3, qwen3_next, deepseek2, deepseek3, gpt_oss, kimi, default, custom.

Pick `custom` ONLY when the architecture differs materially from every
listed family — a novel attention variant, an unusual MoE router, a hybrid
SSM/attention stack, etc. When in doubt, prefer the closest standard family
and rely on the YAML overlay to capture the differences.

Reference signals to consider:
- Attention type: standard MHA, GQA (num_kv_heads != num_heads), MLA,
  sliding-window, hybrid linear/full.
- Normalization: pre-norm vs post-norm, RMSNorm vs LayerNorm, Qwen3-style
  q/k norms.
- MLP: vanilla MLP vs SwiGLU vs MoE (and which router).
- Positional encoding: RoPE (and which variant), ALiBi, none.
- Tied vs untied output projection.

## Reference snippets (RAG):
{rag_context}

## PyTorch source:
```python
{pytorch_code}
```

Return ONLY a single JSON object with this exact shape — no markdown:
{{"decoder_block": "<one of the listed values>", "justification": "<one
sentence>"}}
"""


MAXTEXT_YAML_PROMPT = """You are an expert MaxText configuration author.
Emit a YAML config overlay that drops into `MaxText/configs/models/` and
selects the chosen `decoder_block`.

Follow these rules:
- The file inherits semantics from `MaxText/configs/base.yml`. ONLY emit
  fields that override the base.
- Use snake_case keys, exactly matching MaxText's existing model overlays.
- All dimension fields are prefixed `base_` (e.g. `base_emb_dim`,
  `base_num_query_heads`).
- Required keys: `decoder_block`, `base_emb_dim`, `base_num_query_heads`,
  `base_num_kv_heads`, `base_mlp_dim`, `base_num_decoder_layers`, `head_dim`,
  `vocab_size`. Add MoE keys when applicable.
- Numeric values must be derived faithfully from the source PyTorch config
  — do NOT round, do NOT substitute "reasonable" defaults.
- No comments other than a single header line giving the model name.

## Classification result
decoder_block: {decoder_block}
justification: {justification}

## Reference MaxText configs (RAG):
{rag_context}

## PyTorch source:
```python
{pytorch_code}
```

## Source-derived hints (may be incomplete; cross-check against the source):
{dim_hints}

Return ONLY the YAML content — no markdown fences, no explanation.
"""


MAXTEXT_LAYERS_PROMPT = """You are an expert MaxText layers author. The
classifier judged this model to be `custom` — the existing MaxText decoder
blocks are not a close enough fit, so you must emit a small `.py` file
under `MaxText/layers/` that defines the new decoder block.

CRITICAL RULES — non-negotiable:
1. NEVER reimplement attention, RoPE, RMSNorm, softmax, dropout, an MoE
   router, or a training loop. Import them from `maxtext.layers.*`.
2. The only allowed imports are: `jax`, `jax.numpy as jnp`, `flax.linen as
   nn`, `flax.nnx`, and `maxtext.*`. No `torch`, no `transformers`, no
   bare `numpy` for compute.
3. Use `maxtext.layers.linears.DenseGeneral` (or `MlpBlock`) — never
   `flax.linen.Dense` directly. Sharding annotations live on `DenseGeneral`.
4. Mirror the structure of `Qwen3DecoderLayer` in the RAG context: dataclass
   fields `config: Config`, `mesh: Mesh`, `quant: Quant`, optional
   `model_mode`, then `setup()` and `__call__`.
5. Your file should be small glue code that composes primitives — measured
   in dozens of lines, not hundreds. If you find yourself writing more than
   ~150 lines, you are almost certainly reimplementing something.

{maxtext_best_practices}

## Classification result
decoder_block: custom
justification: {justification}

## Reference MaxText layer files (RAG):
{rag_context}

## PyTorch source:
```python
{pytorch_code}
```

Return ONLY the Python code for the new layers file — no markdown fences,
no explanation.
"""


MAXTEXT_CKPT_CONVERTER_PROMPT = """You are an expert at writing MaxText
checkpoint converters. Emit a standalone Python script that maps a
HuggingFace / PyTorch checkpoint into MaxText's Orbax format for the chosen
`decoder_block`.

Conventions:
- File is named `convert_<name>_ckpt.py` and lives under `MaxText/utils/`.
- Has a `__main__` block accepting `--base_model_path` and
  `--maxtext_model_path` (and any other flags MaxText's existing converters
  use).
- Reuse helpers from `maxtext.utils.max_utils` and `orbax.checkpoint`.
- The key mapping must respect the chosen decoder block's parameter tree
  exactly — Q/K/V splits, MoE expert stacking, RoPE inverse-frequency
  ordering, embedding tie/untie, etc.
- Preserve dtype and shape; do NOT silently cast to float32.

This is a best-effort artifact. If the source does not provide enough
information to write a faithful converter, emit a skeleton with TODOs at
the unresolved points rather than guessing.

## Classification result
decoder_block: {decoder_block}

## Reference MaxText converters (RAG):
{rag_context}

## PyTorch source:
```python
{pytorch_code}
```

## YAML config overlay just emitted:
```yaml
{yaml_config}
```

Return ONLY the Python code for the converter — no markdown fences, no
explanation.
"""


MAXTEXT_VALIDATION_PROMPT = """You are an expert MaxText reviewer. Compare
the ORIGINAL PyTorch source with a CONVERTED MaxText artifact (a YAML
config or a layers `.py` file) and identify every faithfulness deviation
or MaxText anti-pattern.

A deviation is any place where the MaxText output:
- Changes a numeric value, default, dimension, or layer count from the source.
- Drops a feature the source has.
- Reimplements a MaxText primitive (attention, RoPE, RMSNorm, softmax, MoE
  router, KV cache) instead of importing it from `maxtext.layers.*`.
- Imports from `torch`, `transformers`, or any non-MaxText / non-JAX
  library.
- Uses `flax.linen.Dense` where `maxtext.layers.linears.DenseGeneral` is
  required.
- Embeds a custom training loop or optimizer step.

## Original PyTorch Source:
```python
{pytorch_code}
```

## Converted MaxText Output:
```python
{target_code}
```

## Categories to flag:
- "default_value", "missing_component", "reimplemented_primitive",
  "forbidden_import", "wrong_layer_class", "custom_training_loop",
  "dimension_mismatch", "dropped_feature".

Severity:
- "high" — model wiring is wrong or a primitive was reimplemented.
- "medium" — a default/dimension was changed.
- "low" — cosmetic.

Use exact verbatim snippets (max 3 lines) for `source_snippet` and
`output_snippet`. Use "MISSING" if the output lacks the component.

Return ONLY a JSON array of deviations. Empty array if none. No markdown.
Each deviation has keys: category, severity, source_snippet, output_snippet,
corrected_snippet, fix.
"""


MAXTEXT_REPAIR_PROMPT = """You are an expert MaxText developer. You have
been given a converted MaxText artifact (YAML or layers `.py`) along with a
list of faithfulness deviations and anti-pattern flags.

## Original PyTorch Source (for reference):
```python
{pytorch_code}
```

## Current MaxText Output:
```python
{target_code}
```
{rag_section}
## Deviations to Fix:
{deviations_text}

## CRITICAL RULES:
1. NEVER reimplement a MaxText primitive — fix by importing from
   `maxtext.layers.*` instead.
2. NEVER introduce `torch`, `transformers`, or `flax.linen.Dense`.
3. NEVER add a custom training loop, optimizer step, or dataloader.
4. Preserve EVERY existing class/function/import that is correct. Only
   change what the deviation specifies.
5. If a deviation says the current behaviour is acceptable or recommended,
   skip it.
6. For YAML overlays: keep the same field ordering and the same set of
   keys; only edit values that the deviation explicitly identifies.

Return ONLY the complete fixed artifact (Python or YAML). No markdown fences,
no explanation.
"""


# Selector mapping: prompt name -> {target -> prompt template}.
_PROMPT_REGISTRY = {
    "MIGRATE_MODULE_TO_JAX_PROMPT": {
        "jax": MIGRATE_MODULE_TO_JAX_PROMPT,
    },
    "MODEL_CONVERSION_PROMPT": {
        "jax": MODEL_CONVERSION_PROMPT,
    },
    "VALIDATION_PROMPT": {
        # JAX validation prompt lives in validation_agent.py for historical
        # reasons; the selector returns the MaxText variant only.
        "maxtext": MAXTEXT_VALIDATION_PROMPT,
    },
    "REPAIR_PROMPT": {
        "maxtext": MAXTEXT_REPAIR_PROMPT,
    },
    "MAXTEXT_CLASSIFY_PROMPT": {
        "maxtext": MAXTEXT_CLASSIFY_PROMPT,
    },
    "MAXTEXT_YAML_PROMPT": {
        "maxtext": MAXTEXT_YAML_PROMPT,
    },
    "MAXTEXT_LAYERS_PROMPT": {
        "maxtext": MAXTEXT_LAYERS_PROMPT,
    },
    "MAXTEXT_CKPT_CONVERTER_PROMPT": {
        "maxtext": MAXTEXT_CKPT_CONVERTER_PROMPT,
    },
}


def get_prompt(name: str, target: str = "jax") -> str | None:
  """Selects a prompt template by (name, target).

  Args:
    name: The logical prompt identifier (e.g. "MIGRATE_MODULE_TO_JAX_PROMPT",
      "VALIDATION_PROMPT", "MAXTEXT_YAML_PROMPT").
    target: The conversion target — "jax" (default) or "maxtext".

  Returns:
    The prompt template string, or None if no entry matches. Callers that
    need a hard requirement should check for None and raise.
  """
  by_target = _PROMPT_REGISTRY.get(name)
  if not by_target:
    return None
  if target in by_target:
    return by_target[target]
  # Fall back to JAX for any name registered there.
  return by_target.get("jax")
