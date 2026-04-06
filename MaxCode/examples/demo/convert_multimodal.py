"""
Demo: Convert Multimodal-Transformer (PyTorch) to JAX using MaxCode.

This script demonstrates the full MaxCode pipeline:
  1. Clone a PyTorch repo from GitHub
  2. Merge source files into a single file
  3. Populate the RAG database with reference docs
  4. Run migration (PyTorch -> JAX) with automatic validation + repair
  5. Save the JAX output

Usage:
    cd MaxCode/examples/demo
    export GOOGLE_API_KEY=<your-key>
    python convert_multimodal.py
"""

import os
import sys
import time

# ---------------------------------------------------------------------------
# Setup paths — resolve MaxCode root relative to this script
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAXCODE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, MAXCODE_DIR)

if "HOME" not in os.environ:
    os.environ["HOME"] = os.environ.get("USERPROFILE", os.path.expanduser("~"))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPO_URL = "https://github.com/yaohungt/Multimodal-Transformer"
REPO_DIR = os.path.join(SCRIPT_DIR, "Multimodal-Transformer")
SOURCE_FILES = [
    "modules/position_embedding.py",
    "modules/multihead_attention.py",
    "modules/transformer.py",
    "src/models.py",
]
MERGED_FILE = os.path.join(SCRIPT_DIR, "merged_model.py")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
RAG_SOURCE_DIR = os.path.join(MAXCODE_DIR, "rag", "sources")

# ---------------------------------------------------------------------------
# Imports (after sys.path is set)
# ---------------------------------------------------------------------------
import models
from agents.migration.primary_agent import PrimaryAgent
from rag import vector_db


def merge_source_files(repo_dir, file_list, output_path):
    """Merge multiple PyTorch source files into a single file."""
    print("\n--- Merging source files ---")
    merged = '"""\nMerged model file from Multimodal-Transformer.\n'
    merged += f"Source files: {', '.join(file_list)}\n"
    merged += '"""\n\n'

    # Collect all imports first
    import_lines = set()
    code_sections = []

    for rel_path in file_list:
        full_path = os.path.join(repo_dir, rel_path)
        print(f"  Reading {rel_path}")
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()

        section_lines = []
        for line in content.split("\n"):
            stripped = line.strip()
            # Skip relative imports (from .xxx or from ..xxx)
            if stripped.startswith("from .") or stripped.startswith("from .."):
                continue
            # Collect standard imports
            if stripped.startswith("import ") or stripped.startswith("from "):
                import_lines.add(line)
            else:
                section_lines.append(line)

        code_sections.append(
            f"\n# {'=' * 70}\n# From {rel_path}\n# {'=' * 70}\n"
            + "\n".join(section_lines)
        )

    merged += "\n".join(sorted(import_lines)) + "\n"
    merged += "\n".join(code_sections)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(merged)
    print(f"  Merged file: {output_path} ({len(merged)} chars)")
    return merged


def main():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: Set GOOGLE_API_KEY environment variable first.")
        print("  export GOOGLE_API_KEY=<your-key>")
        sys.exit(1)

    print("=" * 70)
    print("MaxCode Demo: Multimodal-Transformer (PyTorch -> JAX)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Clone the repo (if not already present)
    # ------------------------------------------------------------------
    if not os.path.isdir(REPO_DIR):
        print(f"\n[Step 1] Cloning {REPO_URL} ...")
        ret = os.system(f"git clone {REPO_URL} {REPO_DIR}")
        if ret != 0:
            print("ERROR: git clone failed.")
            sys.exit(1)
    else:
        print(f"\n[Step 1] Repo already cloned: {REPO_DIR}")

    # Show source files
    print("\nSource files to convert:")
    for f in SOURCE_FILES:
        full = os.path.join(REPO_DIR, f)
        lines = sum(1 for _ in open(full, encoding="utf-8"))
        print(f"  {f} ({lines} lines)")

    # ------------------------------------------------------------------
    # Step 2: Merge source files into a single file
    # ------------------------------------------------------------------
    print(f"\n[Step 2] Merging source files...")
    merge_source_files(REPO_DIR, SOURCE_FILES, MERGED_FILE)

    # ------------------------------------------------------------------
    # Step 3: Populate RAG database with reference docs
    # ------------------------------------------------------------------
    print(f"\n[Step 3] Populating RAG database...")
    db_path = vector_db.RAG_DB_FILE
    if os.path.exists(db_path):
        os.remove(db_path)

    gemini_flash = models.GeminiTool(
        model_name=models.GeminiModel.GEMINI_2_5_FLASH,
        api_key=api_key,
    )
    agent = PrimaryAgent(model=gemini_flash, api_key=api_key, validate=True)

    t0 = time.time()
    agent._rag_agent.build_from_directory(RAG_SOURCE_DIR)
    elapsed = time.time() - t0

    ids, names, texts, files, embeddings = vector_db.load_all_documents(db_path)
    print(f"  RAG DB: {len(ids)} documents loaded in {elapsed:.1f}s")

    # ------------------------------------------------------------------
    # Step 4: Run migration with validation
    # ------------------------------------------------------------------
    print(f"\n[Step 4] Running migration + validation...")

    # Use best available model for migration
    migration_model = None
    for model_enum in [
        models.GeminiModel.GEMINI_2_5_PRO,
        models.GeminiModel.GEMINI_2_5_FLASH,
    ]:
        try:
            candidate = models.GeminiTool(model_name=model_enum, api_key=api_key)
            candidate("test")
            migration_model = candidate
            print(f"  Using {model_enum.value} for migration")
            break
        except Exception:
            continue

    if migration_model is None:
        print("  ERROR: No Gemini model available.")
        sys.exit(1)

    # Swap to the migration model for conversion
    agent._single_file_agent._model = migration_model
    agent._model_conversion_agent._model = migration_model

    t0 = time.time()
    results = agent.run(MERGED_FILE)
    elapsed = time.time() - t0
    jax_code = list(results.values())[0]

    print(f"  Migration completed in {elapsed:.1f}s")
    print(f"  Output: {len(jax_code)} chars")

    # ------------------------------------------------------------------
    # Step 5: Save output and show results
    # ------------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "multimodal_transformer_jax.py")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(jax_code)
    print(f"\n[Step 5] Output saved to: {out_path}")

    # Show validation results
    validation_results = agent.get_validation_results()
    if validation_results:
        for file_path, result in validation_results.items():
            found = result["deviations_found"]
            remaining = result["remaining_deviations_count"]
            print(f"\n  Validation: {found} deviations found, {remaining} remaining after repair")
    else:
        print("\n  No deviations found - output is faithful!")

    print("\n" + "=" * 70)
    print("Done! JAX output ready at:")
    print(f"  {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
