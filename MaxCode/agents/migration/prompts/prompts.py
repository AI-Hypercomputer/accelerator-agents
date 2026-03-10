"""Prompts for code migration."""

PYTORCH_TO_JAX_SINGLE_FILE_PROMPT = """You are an expert in JAX and PyTorch.
Your task is to convert the following PyTorch code to JAX.
If it is helpful, you can use the following JAX code snippets as context for
functionality that might be similar to your conversion task:
---
{rag_context}
---
The PyTorch code to convert is as follows:
```python
{pytorch_code}
```

Please think step by step about the conversion process before generating the code.
Then, provide the JAX equivalent of the PyTorch code above.
Ensure that the JAX code is idiomatic and follows best practices, such as using
pure functions and handling random number generation correctly with JAX's PRNG
keys. Only return the Python code block for the JAX implementation.
"""

PYTORCH_TO_JAX_REPO_PROMPT = """You are an expert in JAX and PyTorch. Your task
is to convert a repository from PyTorch to JAX. You will be given a file path
and the content of the file. You need to convert the given file from PyTorch to
JAX, considering its context within the repository.
If it is helpful, you can use the following JAX code snippets as context for
functionality that might be similar to your conversion task:
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
"""

HF_TO_JAX_SINGLE_FILE_PROMPT = """You are an expert in JAX and PyTorch, with
special expertise in HuggingFace Transformers. Your task is to convert the
following HuggingFace Transformers code (which uses PyTorch) to JAX.
If it is helpful, you can use the following JAX code snippets as context for
functionality that might be similar to your conversion task:
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
"""

MODEL_CONVERSION_PROMPT = """You are an expert in JAX and PyTorch model
architectures. Your task is to convert the following PyTorch model definition
to a JAX-based equivalent, using libraries such as Flax.
If it is helpful, you can use the following JAX code snippets as context for
functionality that might be similar to your conversion task:
---
{rag_context}
---
PyTorch model:
```python
{pytorch_model_code}
```

Please think step by step about the conversion process before generating the code.
Then, provide the JAX equivalent of the model definition above.
Ensure that the JAX code is idiomatic and follows best practices for defining
models in JAX, such as using pure functions and handling random number
generation correctly with JAX's PRNG keys.
Only return the Python code block for the JAX implementation.
"""
