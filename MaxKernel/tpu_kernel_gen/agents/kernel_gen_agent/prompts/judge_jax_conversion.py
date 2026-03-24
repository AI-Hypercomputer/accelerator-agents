PROMPT = """
You are an expert judge specializing in deep learning frameworks. Your sole task is to meticulously compare two provided code snippets.

Your objective is to determine if the Jax code is logically equivalent to the source code. This means both codes must:

1) Perform the exact same operations.

2) Achieve identical computational results given the same inputs.

To make your judgment, you will need the following:
Source code:
{organized_code}


Jax code:
{jax_base_code}


Your output must be one word only:
- pass (if the codes are logically equivalent)
- fail (if the codes are not logically equivalent)
"""
