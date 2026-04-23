"""Agent for generating model configurations for evaluation."""

from typing import Any

from agents import base
from agents import utils

CONFIG_GENERATION_PROMPT = """You are an Expert ML Engineer tasked with analyzing PyTorch model definitions.
Your goal is to determine the necessary arguments for instantiation (`init_kwargs`) and a plausible input tensor shape (`input_shape`) for each `nn.Module` subclass in the provided code.

Analyze the following PyTorch code:
```python
{code}
```

Output **only** a valid JSON object containing the class name of each `nn.Module` as a key.
Each class entry must contain:
1. `init_kwargs`: A dictionary with reasonable default arguments needed to instantiate the class. If no arguments are needed, provide an empty dictionary.
2. `input_shape`: A list of integers representing the dimensions for a `torch.randn` dummy input tensor, including a batch size of 1. If the model expects multiple inputs, provide a list of lists.

Example output for a simple model:
```json
{{
  "MyModel": {{
    "init_kwargs": {{
      "num_classes": 10
    }},
    "input_shape": [1, 3, 224, 224]
  }}
}}
```

Do not include any text before or after the JSON object.
"""


class ConfigGenerationAgent(base.Agent):
  """Agent for generating model configurations for evaluation."""

  def __init__(self, model: Any):
    """Initializes the agent."""
    super().__init__(
        model=model,
        agent_domain=utils.AgentDomain.EVALUATION,
    )

  def run(self, code: str) -> str:
    """Runs the agent to generate model configurations."""
    return self.generate(
        CONFIG_GENERATION_PROMPT,
        prompt_vars={"code": code},
    )
