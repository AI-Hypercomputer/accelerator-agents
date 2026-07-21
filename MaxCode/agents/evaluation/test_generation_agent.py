"""Agent for generating equivalence tests."""

from typing import Any

from agents import base
from agents import utils

TEST_GENERATION_PROMPT = """You are an Expert Software Development Engineer in Test (SDET) tasked with generating an equivalence test between a PyTorch model and its migrated JAX/Flax NNX counterpart.
You will be given the JAX/Flax NNX code and the original PyTorch code. You need to generate a Python test script using `absltest` that verifies:
1. The JAX NNX model can be instantiated and run with dummy inputs.
2. The JAX NNX model, when loaded with weights from the PyTorch model, produces numerically close outputs given the same inputs.

You should use `absl.flags` to allow the user to specify the path to a pickle file containing a dictionary with keys 'input', 'output', 'state_dict', and 'intermediates', generated from the original PyTorch model.

The JAX code is saved in a module named `{jax_module_name}`. In your generated code, import it as `import {jax_module_name} as jax_model`.

The JAX code is:
```python
{jax_code}
```

The PyTorch code is:
```python
{pytorch_code}
```

Follow these instructions to generate the test file:
1.  **Setup & Imports**: Import `jax`, `flax`, `flax.nnx`, `numpy`, `torch` (for unpickling), `pickle`, `os`, `json`, `absltest`, and `absl.flags`. Import the JAX module as `import {jax_module_name} as jax_model`.
2.  **Flag Definition**: Define a flag named `pickle_path` using `flags.DEFINE_string` which defaults to `model.pkl`, e.g., `_PICKLE_PATH = flags.DEFINE_string("pickle_path", "model.pkl", "Path to oracle data.")`.
3.  **Model Configuration Loading**:
    *   Load the configurations from `model_configs.json` which is located in the parent directory of the test script:
        ```python
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_configs.json")
        with open(config_path, "r") as f:
            model_configs = json.load(f)
        ```
    *   Locate the specific configuration dictionary for the target class under test (e.g. `class_config = model_configs["<ClassName>"]`). Extract `init_kwargs = class_config.get("init_kwargs", {{}})`.
    *   If a `config` key is present in `init_kwargs` and its value is a dictionary, dynamically convert it to a namespace so attribute lookups (e.g. `config.hidden_size`) succeed:
        ```python
        from types import SimpleNamespace
        for k, v in list(init_kwargs.items()):
            if k == "config" and isinstance(v, dict):
                init_kwargs[k] = SimpleNamespace(**v)
        ```
4.  **Test 1: `test_jax_model_validity()`**:
    *   Instantiate the JAX NNX model using the loaded `init_kwargs`: `model = jax_model.MyClass(**init_kwargs, rngs=nnx.Rngs(42))` (or without `rngs` if the constructor doesn't accept RNGs; handle TypeError fallback if needed).
    *   Generate dummy input and run the model. Wrap the *entire* input generation and forward call block inside a `try ... except Exception:` block.
    *   In the `try` block:
        *   Inspect `input_shape` from `class_config`. If it is a list of lists (e.g. `[[1, 10, 64], [1, 10]]`), generate a list of dummy JAX tensors of ones for each shape, and call the model unpacking the arguments: `output = model(*dummy_inputs)`.
        *   If it is a single list of dimensions, generate a single dummy JAX tensor of ones, and call: `output = model(dummy_input)`.
    *   In the `except Exception:` block, fallback to loading inputs directly from the oracle pickle file (specified by `_PICKLE_PATH.value`), matching how `test_equivalence` extracts JAX inputs from `oracle_data['input']`, and execute the model.
    *   Assert that the output contains no NaNs.
5.  **Test 2: `test_equivalence()`**:
    *   Load the pickle file specified by `_PICKLE_PATH.value`. The pickle file contains a dictionary with keys 'input', 'output', 'state_dict', and 'intermediates'.
    *   If PyTorch input is a tuple, use the first element.
    *   Convert PyTorch input tensor to a Numpy array using `.detach().numpy()`. If it's 4D (NCHW), transpose it to NHWC format for JAX `(0, 2, 3, 1)`.
    *   Instantiate the JAX NNX model using the loaded `init_kwargs` (same as Test 1).
    *   Load PyTorch `state_dict` and overwrite the target JAX NNX parameter state values directly. Expose the flat JAX NNX parameter state via `params = nnx.state(model, nnx.Param)`. Map and transpose PyTorch weights:
        *   Convolution weights: PyTorch `(Out, In, H, W)` -> JAX NNX `(H, W, In, Out)` via `(2, 3, 1, 0)`.
        *   Linear weights: PyTorch `(Out, In)` -> JAX NNX `(In, Out)` via `(1, 0)`.
        *   Assign biases and parameters directly over instance properties (e.g., `model.dense.kernel.value = ...`). Account for state variable nesting by mapping PyTorch layers directly.
    *   **Numerical Verification of Intermediates**: The test MUST extract intermediate activations directly from the model instance state (e.g. intermediate variables via `intermediates = nnx.state(model, nnx.Intermediate)`). It must then iterate through the `intermediates` dictionary provided in the oracle data (captured via PyTorch hooks) and use `np.testing.assert_allclose` to verify each JAX intermediate against its PyTorch counterpart. Ensure `err_msg=f"Mismatch in layer: {{layer_name}}"` is used for precise error reporting.
    *   Assert `np.testing.assert_allclose(jax_output, torch_output.detach().numpy(), atol=1e-5)` to check for numerical equivalence of the final output.
6.  Include `absltest.main()` runner block.

Output **only** the Python code block for the test file. Do not include any text before or after the code.
"""


class TestGenerationAgent(base.Agent):
  """Agent for generating equivalence tests."""

  def __init__(self, model: Any):
    """Initializes the agent."""
    super().__init__(
        model=model,
        agent_domain=utils.AgentDomain.EVALUATION,
    )

  def run(
      self, jax_code: str, pytorch_code: str, jax_module_name: str = "jax_model"
  ) -> str:
    """Runs the agent to generate equivalence tests."""
    return self.generate(
        TEST_GENERATION_PROMPT,
        prompt_vars={
            "jax_code": jax_code,
            "pytorch_code": pytorch_code,
            "jax_module_name": jax_module_name,
        },
    )
