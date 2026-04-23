"""Prompts for RAG."""

CODE_DESCRIPTION = """
You are given:
    1. A specific code block from a file.
    2. The full code of the file the block was extracted from, for reference.
Your task is to produce a minimal, precise, and machine-readable description of the code block that:
    - Explains what the code block does in clear, concise terms.
    - Explains how the code block can be used, including its purpose, inputs, and outputs (if any).
    - Avoids unnecessary details or implementation steps that are not essential for understanding its usage.
    - Is written so that another AI agent can understand and use it.
Output Format (JSON):
{{
  "functionality": "<Short, clear description of what the code block does>",
  "usage": "<Short, clear description of how to use the code block, including inputs and outputs>"
}}

Code Block:

{code_block}

Full File Code (Reference):

{full_code_context}
"""
