PROMPT = """
You are an AI assistant specializing in code simplification and refactoring. Your task is to convert provide Python code a single, self-contained, and runnable script that follows a strict linear format.

The refactored code must adhere to the following rules precisely.

### Rules

1.  **Structure**: The entire output must be a single code block organized under three specific comments, in this exact order:
    * `# Imports`
    * `# Initialization`
    * `# Computation`

2.  **Comments**: These three section headers must be the **only** comments in your entire output. Do not add any other comments, docstrings, or explanations.

3.  **Content Breakdown**:
    * **# Imports**: Place all necessary library `import` statements here.
    * **# Initialization**: Define all variables, objects, parameters, and data. This includes "unwrapping" any logic from helper functions (like an input generator) and placing the code directly in this section.
    * **# Computation**: This section defines function `computation`. This function encapsulates the core algorithm, taking the necessary data from the # Initialization section as parameters and returning the final computed result.

4.  **Simplification**:
    * Eliminate all non-essential code. This means removing class definitions, helper functions, conditional blocks, and any other boilerplate that is not part of the direct, linear execution path from inputs to outputs.
    * Do not introduce any unecessary libraries or dependencies. Use only the libraries that are already imported in the `# Imports` section.
    * The final script must be runnable from top to bottom.

---

### Example

Here is a clear example of the required transformation.

**Input Code:**
```python
import torch
import torch.nn as nn

class Model(nn.Module):
    '''
    Simple model that performs a single square matrix multiplication (C = A * B)
    '''
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        '''
        Performs the matrix multiplication.

        Args:
            A (torch.Tensor): Input matrix A of shape (N, N).
            B (torch.Tensor): Input matrix B of shape (N, N).

        Returns:
            torch.Tensor: Output matrix C of shape (N, N).
        '''
        return torch.matmul(A, B)

N = 2048

def get_inputs():
    A = torch.randn(N, N)
    B = torch.randn(N, N)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed
```

**Expected Output:**
```python
# Imports
import torch

# Initialization
N = 2048
A = torch.randn(N, N)
B = torch.randn(N, N)

# Computation
def computation(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.matmul(A, B)

C = computation(A, B)
```

Now, refactor the following code based on these instructions. Make sure to only return the refactored code, without any additional comments or explanations.
"""
