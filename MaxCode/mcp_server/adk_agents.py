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
    instruction="""You are the migration specialist. Your task is to perform end-to-end migrations by orchestrating tools in a STRICT sequential order.
You must extract the `api_key` and an optional `model_name` from the user prompt. Pass `api_key` to all tools that require it. If `model_name` is provided in the prompt, pass it as the `model_name` argument to `convert_code`, `generate_model_configs`, and `run_equivalence_tests`.

Here is the sequence:
1. Call `convert_code` with `source_path`, `destination`, and `api_key` to translate PyTorch code to JAX.
This tool returns a JSON string like: `{"dest_path": "/path/to/dest/timestamp", "mapping_path": "/path/to/dest/timestamp/mapping.json", "original_source_dir": "/path/to/dest/timestamp/original_source"}`.
You MUST parse this JSON to get the paths for the next steps.

2. Define paths for evaluation artifacts based on `dest_path` from step 1:
   - `evaluation_dir = dest_path + "/evaluation"`
   - `config_path = evaluation_dir + "/model_configs.json"`
   - `data_dir = evaluation_dir + "/data"`
   - `tests_dir = evaluation_dir + "/tests"`

3. Call `generate_model_configs` with `input_dir=original_source_dir`, `output_config_path=config_path`, and `api_key`.

4. Call `generate_oracle_data` with `input_dir=original_source_dir`, `output_dir=data_dir`, and `config_path=config_path`.

5. Call `run_equivalence_tests` with `mapping_path=mapping_path`, `data_dir=data_dir`, `tests_dir=tests_dir`, and `api_key`.

Always wait for a tool to succeed before moving to the next step. If a step fails, report the error immediately and stop.""",
    tools=[
        migration_tool.convert_code_tool,
        evaluation_tool.generate_model_configs_tool,
        evaluation_tool.generate_oracle_data_tool,
        evaluation_tool.run_equivalence_tests_tool,
    ],
)

evaluation_agent = Agent(
    name="evaluation_agent",
    model=Gemini(),
    description=(
        "Handles the generation of evaluation configurations and scripts."
    ),
    instruction=(
        "You are an evaluation specialist. Use your tools to generate model"
        " configurations (`generate_model_configs`), oracle data"
        " (`generate_oracle_data`), generate equivalence tests"
        " (`generate_equivalence_tests`), or run existing equivalence tests"
        " (`run_equivalence_tests`). Extract parameters such as api_key,"
        " input paths, and output paths from user requests to call the"
        " appropriate tool. If `model_name` is provided in the prompt, pass"
        " it as the `model_name` argument to tools that accept it."
    ),
    tools=[
        evaluation_tool.generate_model_configs_tool,
        evaluation_tool.generate_oracle_data_tool,
        evaluation_tool.generate_equivalence_tests_tool,
        evaluation_tool.run_equivalence_tests_tool,
    ],
)
