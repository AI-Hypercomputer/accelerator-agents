"""
Step 4: Convert the merged PyTorch model to JAX using MaxCode.

This script runs the full MaxCode migration pipeline on the merged model
file from Step 3:

  1. Loads the merged PyTorch source (all model files in one)
  2. Converts it to JAX/Flax using Gemini with RAG context
  3. Validates the output against the PyTorch source for faithfulness
  4. Auto-repairs any deviations found during validation
  5. Re-validates the repaired output
  6. Saves the final JAX file

Using a single merged file gives the LLM full context of all model
components and their dependencies, producing higher quality output
than converting files independently.

Requires:
  - GOOGLE_API_KEY environment variable
  - Step 2 completed (RAG database populated)
  - Step 3 completed (merged model file created)

Usage:
    python step4_convert.py
"""

import logging
import os
import time
from config import MERGED_FILE, MERGED_UTILS_FILE, OUTPUT_DIR, REPO_URL, setup, require_api_key

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    api_key = require_api_key()
    setup()

    import models
    from agents.migration.primary_agent import PrimaryAgent
    from rag import vector_db

    # Pre-flight checks
    if not os.path.isfile(MERGED_FILE):
        print("ERROR: Merged model file not found. Run step3_merge.py first.")
        raise SystemExit(1)

    db_path = vector_db.RAG_DB_FILE
    if not os.path.exists(db_path):
        print("ERROR: RAG database not found. Run step2_populate_rag.py first.")
        raise SystemExit(1)

    print("=" * 70)
    print("Step 4: Convert PyTorch to JAX")
    print("=" * 70)
    print(f"  Source: {MERGED_FILE}")
    print(f"  Output: {OUTPUT_DIR}")
    print()

    # Initialize agent with RAG and validation enabled
    gemini_flash = models.GeminiTool(
        model_name=models.GeminiModel.GEMINI_2_5_FLASH,
        api_key=api_key,
    )
    agent = PrimaryAgent(model=gemini_flash, api_key=api_key, validate=True)

    # Use best available model for migration
    migration_model = None
    for model_enum in [
        models.GeminiModel.GEMINI_3_1_PRO_PREVIEW,
        models.GeminiModel.GEMINI_2_5_PRO,
        models.GeminiModel.GEMINI_2_5_FLASH,
    ]:
        try:
            candidate = models.GeminiTool(model_name=model_enum, api_key=api_key)
            candidate("test")
            migration_model = candidate
            print(f"  Migration model: {model_enum.value}")
            break
        except Exception:
            continue

    if migration_model is None:
        print("  ERROR: No Gemini model available.")
        raise SystemExit(1)

    agent._single_file_agent._model = migration_model
    agent._model_conversion_agent._model = migration_model

    # Run migration
    print(f"\n  Converting (this may take several minutes)...\n")
    t0 = time.time()
    results = agent.run(MERGED_FILE)
    elapsed = time.time() - t0
    jax_code = list(results.values())[0]

    print(f"\n  Migration completed in {elapsed:.1f}s")

    # Save output — derive filename from repo URL
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    repo_name = REPO_URL.rstrip("/").rsplit("/", 1)[-1].replace("-", "_")
    out_path = os.path.join(OUTPUT_DIR, f"{repo_name}_jax.py")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(jax_code)
    lines = jax_code.count("\n") + 1
    print(f"  Output: {out_path} ({lines} lines)")

    # ------------------------------------------------------------------
    # Convert utility files (if any)
    # ------------------------------------------------------------------
    if os.path.isfile(MERGED_UTILS_FILE):
        print("\n" + "-" * 70)
        print("  Converting utility files...")
        print(f"  Source: {MERGED_UTILS_FILE}")
        with open(MERGED_UTILS_FILE, "r", encoding="utf-8") as f:
            utils_code = f.read()
        utils_lines_in = utils_code.count("\n") + 1
        print(f"  Input: {utils_lines_in} lines")

        t1 = time.time()
        utils_jax = agent._single_file_agent.run(utils_code)
        utils_jax = agent._fill_missing_components(utils_code, utils_jax)
        utils_elapsed = time.time() - t1

        print(f"  Utility conversion completed in {utils_elapsed:.1f}s")

        utils_out_path = os.path.join(OUTPUT_DIR, f"{repo_name}_utils_jax.py")
        with open(utils_out_path, "w", encoding="utf-8") as f:
            f.write(utils_jax)
        utils_lines_out = utils_jax.count("\n") + 1
        print(f"  Output: {utils_out_path} ({utils_lines_out} lines)")
    else:
        print("\n  No merged utility file found — skipping utility conversion.")

    # Validation summary
    validation_results = agent.get_validation_results()
    if validation_results:
        for file_path, result in validation_results.items():
            found = result["deviations_found"]
            remaining = result["remaining_deviations_count"]
            print(f"\n  Validation: {found} deviations found, {remaining} remaining after repair")
    else:
        print("\n  No deviations found - output is faithful!")

    print("\n" + "=" * 70)
    print("Done! JAX output:")
    print(f"  {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
