"""
Step 3: Convert PyTorch code to JAX using MaxCode.

This script runs the full MaxCode migration pipeline on the cloned repo:

  1. Auto-discovers all Python files and builds a dependency graph
  2. Converts each file in topological order (dependencies first)
  3. Validates each converted file against the PyTorch source
  4. Auto-repairs any deviations found during validation
  5. Re-validates the repaired output
  6. Saves all converted JAX files preserving the original directory structure

The migration uses Gemini Pro (or Flash as fallback) with RAG context from
the database populated in Step 2. The validation agent checks for common
conversion errors like wrong initialization, dropped features, incorrect
reduction operations, and missing components.

Requires:
  - GOOGLE_API_KEY environment variable
  - Step 1 completed (repo cloned)
  - Step 2 completed (RAG database populated)

Usage:
    python step3_convert.py
"""

import os
import time
from config import REPO_DIR, OUTPUT_DIR, RAG_SOURCE_DIR, setup, require_api_key

def main():
    api_key = require_api_key()
    setup()

    import models
    from agents.migration.primary_agent import PrimaryAgent
    from rag import vector_db

    # Pre-flight checks
    if not os.path.isdir(REPO_DIR):
        print("ERROR: Repository not found. Run step1_clone_repo.py first.")
        raise SystemExit(1)

    db_path = vector_db.RAG_DB_FILE
    if not os.path.exists(db_path):
        print("ERROR: RAG database not found. Run step2_populate_rag.py first.")
        raise SystemExit(1)

    print("=" * 70)
    print("Step 3: Convert PyTorch to JAX")
    print("=" * 70)
    print(f"  Source: {REPO_DIR}")
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
    results = agent.run(REPO_DIR)
    elapsed = time.time() - t0

    print(f"\n  Converted {len(results)} files in {elapsed:.1f}s")

    # Save output files
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n  Saving output files:")
    for src_path, jax_code in results.items():
        rel_path = os.path.relpath(src_path, REPO_DIR)
        out_path = os.path.join(OUTPUT_DIR, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(jax_code)
        lines = jax_code.count("\n") + 1
        print(f"    {rel_path} ({lines} lines)")

    # Validation summary
    validation_results = agent.get_validation_results()
    if validation_results:
        print("\n  Validation summary:")
        total_found = 0
        total_remaining = 0
        for file_path, result in validation_results.items():
            name = os.path.relpath(file_path, REPO_DIR)
            found = result["deviations_found"]
            remaining = result["remaining_deviations_count"]
            total_found += found
            total_remaining += remaining
            status = "OK" if remaining == 0 else f"{remaining} remaining"
            print(f"    {name}: {found} found, {status}")
        print(f"\n  Total: {total_found} deviations found, {total_remaining} remaining after repair")
    else:
        print("\n  No deviations found - all outputs are faithful!")

    print("\n" + "=" * 70)
    print("Done! JAX output:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
