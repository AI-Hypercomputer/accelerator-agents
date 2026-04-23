import os
import json
import logging
import pathlib
from tools import migration_tool, evaluation_tool
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Set environment variables
api_key = os.environ.get("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = api_key or ""
os.environ["ALLOYDB_HOST"] = "dummy"
os.environ["ALLOYDB_PASS"] = "dummy"

source_path = "/usr/local/google/home/katiao/accelerator-agents/MaxCode/evaluation/code_agent/pytorch_references/level2/self_attention.py"
destination = "/tmp/migrated_attention"

def run_migration():
    print(f"Step 1: Converting {source_path} to JAX...")
    convert_result_json = migration_tool.convert_code(
        source_path=source_path,
        destination=destination,
        api_key=api_key
    )
    convert_result = json.loads(convert_result_json)
    if "error" in convert_result:
        print(f"Error in convert_code: {convert_result['error']}")
        return

    dest_path = convert_result["dest_path"]
    mapping_path = convert_result["mapping_path"]
    original_source_dir = convert_result["original_source_dir"]

    print(f"Converted code saved to {dest_path}")

    # Step 2: Define paths for evaluation artifacts
    evaluation_dir = os.path.join(dest_path, "evaluation")
    config_path = os.path.join(evaluation_dir, "model_configs.json")
    data_dir = os.path.join(evaluation_dir, "data")
    tests_dir = os.path.join(evaluation_dir, "tests")

    # Step 3: Generate model configs
    print("Step 3: Generating model configurations...")
    config_result = evaluation_tool.generate_model_configs(
        input_dir=original_source_dir,
        output_config_path=config_path,
        api_key=api_key
    )
    print(config_result)

    # Step 4: Generate oracle data
    print("Step 4: Generating oracle data...")
    oracle_result = evaluation_tool.generate_oracle_data(
        input_dir=original_source_dir,
        output_dir=data_dir,
        config_path=config_path
    )
    print(oracle_result)

    # Step 5: Run equivalence tests
    print("Step 5: Running equivalence tests...")
    test_result = evaluation_tool.run_equivalence_tests(
        mapping_path=mapping_path,
        data_dir=data_dir,
        tests_dir=tests_dir,
        api_key=api_key
    )
    print(test_result)

if __name__ == "__main__":
    run_migration()
