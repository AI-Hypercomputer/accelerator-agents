"""Agent for generating equivalence tests."""

from typing import Any

from agents import base
from agents import utils

TEST_GENERATION_PROMPT = """You are an Expert Software Development Engineer in Test (SDET) tasked with generating an equivalence test between a PyTorch model and its migrated JAX/Flax counterpart.
You will be given the JAX/Flax code and the original PyTorch code. You need to generate a Python test script using `absltest` that verifies:
1. The JAX model can be instantiated and run with dummy inputs.
2. The JAX model, when loaded with weights from the PyTorch model, produces numerically close outputs given the same inputs.

You should use `absl.flags` to allow the user to specify the path to a pickle file containing a dictionary with keys 'input', 'output', and 'state_dict', generated from the original PyTorch model.

The JAX code is:
```python
{jax_code}
```

The PyTorch code is:
```python
{pytorch_code}
```

Follow these instructions to generate the test file:
1.  **Setup & Imports**: Import `jax`, `flax`, `numpy`, `torch` (for unpickling), `pickle`, `os`, `absltest`, and `absl.flags`.
2.  **Flag Definition**: Define a flag named `pickle_path` using `flags.DEFINE_string` which defaults to `model.pkl`, e.g., `_PICKLE_PATH = flags.DEFINE_string("pickle_path", "model.pkl", "Path to oracle data.")`.
3.  **Test 1: `test_jax_model_validity()`**:
    *   Instantiate the JAX model.
    *   Generate a dummy JAX input tensor with a shape compatible with the model (batch size 1, NHWC format for 4D tensors).
    *   Run `model.init()` and `model.apply()` with the dummy input.
    *   Assert that the output has the expected shape and contains no NaNs.
4.  **Test 2: `test_equivalence()`**:
    *   Load the pickle file specified by `_PICKLE_PATH.value`. The pickle file contains a dictionary with keys 'input', 'output', and 'state_dict'.
    *   If PyTorch input is a tuple, use the first element.
    *   Convert PyTorch input tensor to a Numpy array using `.detach().numpy()`. If it's 4D (NCHW), transpose it to NHWC format for JAX `(0, 2, 3, 1)`.
    *   Instantiate the JAX model.
    *   Load PyTorch `state_dict` and create a JAX `params` dictionary by mapping and transposing weights:
        *   Convolution weights: PyTorch `(Out, In, H, W)` -> Flax `(H, W, In, Out)`. Transpose with `(2, 3, 1, 0)`.
        *   Linear weights: PyTorch `(Out, In)` -> Flax `(In, Out)`. Transpose with `(1, 0)`.
        *   Copy biases and other parameters without transpose.
        *   The JAX params structure may be nested, e.g., `{{'params': {{'Conv_0': {{'kernel': ..., 'bias': ...}}}}}}`. Map PyTorch weights to the correct Flax names and structure.
    *   Run JAX `model.apply()` using the converted parameters and input.
    *   Assert `np.testing.assert_allclose(jax_output, torch_output.detach().numpy(), atol=1e-5)` to check for numerical equivalence.
5.  Include `absltest.main()` runner block.

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

  def run(self, jax_code: str, pytorch_code: str) -> str:
    """Runs the agent to generate equivalence tests."""
    return self.generate(
        TEST_GENERATION_PROMPT,
        prompt_vars={"jax_code": jax_code, "pytorch_code": pytorch_code},
    )
