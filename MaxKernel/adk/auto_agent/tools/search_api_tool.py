"""
Notes of tools needed:

1) Tool to search jax links that are provided in error output

https://python.langchain.com/docs/integrations/document_loaders/web_base/#loader-features

2) Tool to search pallas/jax code to make sure APIs are used correctly

"""

from google.adk.tools import FunctionTool
from google.genai import types

from auto_agent.constants import (
  TEMPERATURE,
  TOP_K,
  TOP_P,
)
from auto_agent.tools.api_rag.get_apis import generate_definition

model_config = types.GenerateContentConfig(
  temperature=TEMPERATURE,
  top_p=TOP_P,
  top_k=TOP_K,
)


def search_api(api_name: str) -> dict:
  """
  Search for API documentation and generate its definition.

  This function attempts to retrieve and generate a definition for a given API name.
  It serves as a tool for looking up API specifications and returning structured
  information about the requested API.

  Args:
      api_name: The name of the API to search for and generate documentation.

  Returns:
      A dictionary containing the operation result with the following structure:
      - If successful: {"status": "success", "message": <api_definition>}
      - If failed: {"status": "error", "message": "Failed to resolve API: <error_details>"}
  """
  try:
    definition = generate_definition(api_name)
    return {"status": "success", "message": definition}
  except Exception as e:
    return {
      "status": "error",
      "message": f"Failed to resolve API '{api_name}': {e}",
    }


# Wrap the function with FunctionTool for compatibility with ADK agents and MCP tools
search_api_tool = FunctionTool(search_api)
