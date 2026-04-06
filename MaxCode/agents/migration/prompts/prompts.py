"""Prompts for code migration."""

JAX_BEST_PRACTICES = """
## JAX/Flax Best Practices (MUST follow)

1. **Use Flax Linen with @nn.compact**: Define all submodules inline inside
   `@nn.compact def __call__`. Do NOT use a separate `setup()` method or NNX.
   All Flax modules must be defined using the `@nn.compact` decorator.
2. **KV Cache**: Use pre-allocated fixed-size caches updated via
   `jax.lax.dynamic_update_slice`. NEVER grow the cache with `jnp.concatenate`
   or Python list appends -- that breaks XLA compilation.
3. **Causal Conv1d**: Use `padding="VALID"` with explicit left-padding
   (`jnp.pad` with `pad_width` on the time axis). Do NOT use `padding="SAME"`
   as it is non-causal and leaks future information. Implement as a standalone
   function with both prefill (full sequence) and decode (single-step with
   conv_state) paths. Use `jax.lax.conv_general_dilated` with
   `feature_group_count=channels` for depthwise convolution.
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

## CRITICAL: Faithfulness to Source Code

NEVER simplify complex tensor reshaping, reordering, or algorithmic patterns
from the source code. If the PyTorch code uses a specific interleaved weight
layout, chunk-parallel algorithm, or multi-step computation, convert it
faithfully to JAX. The RAG context shows EXAMPLES of similar patterns -- use
them as guidance for JAX idioms, but always follow the ACTUAL source code's
logic and structure.
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

Please think step by step about the conversion process before generating the code.
Then, provide the complete JAX equivalent of the entire file above.
Ensure that the JAX code is idiomatic and follows best practices for defining
models in JAX, such as using pure functions and handling random number
generation correctly with JAX's PRNG keys.
Only return the Python code block for the JAX implementation.
""" + JAX_BEST_PRACTICES
