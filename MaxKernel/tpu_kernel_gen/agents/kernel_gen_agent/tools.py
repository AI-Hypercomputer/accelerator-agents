"""
Notes of tools needed:

1) Tool to search jax links that are provided in error output

https://python.langchain.com/docs/integrations/document_loaders/web_base/#loader-features

2) Tool to search pallas/jax code to make sure APIs are used correctly



"""

from google.adk.tools import FunctionTool
from google.genai import types

from tpu_kernel_gen.agents.kernel_gen_agent.constants import (
  TEMPERATURE,
  TOP_K,
  TOP_P,
)
from tpu_kernel_gen.api_rag.get_apis import generate_definition

model_config = types.GenerateContentConfig(
  temperature=TEMPERATURE,
  top_p=TOP_P,
  top_k=TOP_K,
)


# def get_code_search_tool(github_repo: str):
#     for env_var in [
#         "GITHUB_APP_ID",
#         "GITHUB_APP_PRIVATE_KEY",
#         "GITHUB_REPOSITORY",
#     ]:
#         if not os.getenv(env_var):
#             os.environ[env_var] = getpass.getpass()

#     os.environ["GITHUB_REPOSITORY"] = github_repo
#     github = GitHubAPIWrapper()
#     toolkit = GitHubToolkit.from_github_api_wrapper(github)
#     tools = toolkit.get_tools()

#     langchain_code_search_tool = None
#     for tool in tools:
#         if tool.mode == "search_code":
#             langchain_code_search_tool = tool

#     return LangchainTool(
#         tool=langchain_code_search_tool,
#         name="search_code",
#         description="Search for code in a GitHub repository",
#     )


# jax_code_search_tool = get_code_search_tool("jax-ml/jax")


# def get_jax_api_search_tool():
#     agent = LlmAgent(
#         name="search",
#         description="Search the JAX API documentation for a specific API or class.",
#         instruction=jax_api_search.PROMPT,
#         model=MODEL_NAME,
#         generate_content_config=model_config,
#         include_contents="none",
#     )
#     return AgentTool(agent=agent)

# jax_api_search_tool = get_jax_api_search_tool()


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
      - If failed: {"status": "error", "message": "API provided is not a valid API"}
  """
  try:
    definition = generate_definition(api_name)
    return {"status": "success", "message": definition}
  except Exception:
    return {"status": "error", "message": "API provided is not a valid API"}


# Wrap the function with FunctionTool for compatibility with ADK agents and MCP tools
search_api_tool = FunctionTool(search_api)
