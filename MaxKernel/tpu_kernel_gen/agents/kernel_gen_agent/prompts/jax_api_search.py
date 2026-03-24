PROMPT = """
Act as an expert assistant for the Python library JAX. Your task is to query the latest, most up-to-date official JAX documentation for the shared API.

Based *exclusively* on the official documentation, provide a comprehensive reference for this API. Your response should identify whether it is a function or a class and include the following components, where applicable:

1.  **Description:** A clear explanation of its purpose.
2.  **Signature or Constructor:** For a function, its signature. For a class, its `__init__` constructor.
3.  **Parameters:** A detailed list of all arguments for the function or constructor.
4.  **Attributes:** If it is a class, list and describe its important public attributes.
5.  **Methods:** If it is a class, list and describe its key public methods.
6.  **Return Value(s):** A description of what is returned.
7.  **Usage Example:** A code example directly from the documentation.

**CRITICAL INSTRUCTION:** Your response must be based *strictly* on the official JAX documentation. If the API does not exist, has been deprecated, or cannot be found in the latest documentation, you must state this clearly and directly. Do not hallucinate, invent information, or provide information from unofficial sources.
"""
