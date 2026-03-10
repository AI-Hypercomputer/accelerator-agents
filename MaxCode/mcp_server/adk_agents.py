"""ADK agent definitions."""

from tools import evaluation_tool
from tools import migration_tool
from google.adk.agents.llm_agent import LlmAgent as Agent
from google.adk.models.google_llm import Gemini

migration_agent = Agent(
    name="migration_agent",
    model=Gemini(),
    description=(
        "Handles end-to-end code migration tasks, such as converting PyTorch"
        " to JAX, generating oracle data, and creating equivalence tests."
    ),
    instruction=(
        "Your task is to migrate code, generate evaluation data, and"
        " generate equivalence tests by calling the `migrate_module` tool."
        " Extract the following parameters from the user request: `path`"
        " (the source file or directory), `destination` (the output directory),"
        " and `api_key` (the Google AI API key). Ensure all three parameters"
        " are present before calling the tool."
    ),
    tools=[migration_tool.migrate_module_tool],
)

evaluation_agent = Agent(
    name="evaluation_agent",
    model=Gemini(),
    description=(
        "Handles the generation of evaluation configurations and scripts."
    ),
    instruction=(
        "Your task is to generate model configurations for evaluation by"
        " calling the generate_model_configs tool with parameters extracted"
        " from the user request. Call the generate_equivalence_tests tool when"
        " the user requests to generate tests or verify equivalence between"
        " JAX and PyTorch models."
    ),
    tools=[
        evaluation_tool.generate_model_configs_tool,
        evaluation_tool.generate_equivalence_tests_tool,
    ],
)

master_agent = Agent(
    name="primary_agent",
    model=Gemini(),
    description="Main orchestrator agent for code migration.",
    instruction=(
        "You are the Primary Agent. You are in charge of a team of agents that"
        " can help with software development tasks. If the user request is"
        " about code migration, delegate the task to the 'migration_agent'. If"
        " the user request is about generating data configs, benchmarking, or"
        " evaluation, delegate the task to the 'evaluation_agent'. If the user"
        " asks about other topics, say that you cannot help."
    ),
    sub_agents=[migration_agent, evaluation_agent],
)
