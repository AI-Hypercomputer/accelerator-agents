"""
Step 2: Populate the RAG (Retrieval-Augmented Generation) database.

This script builds a vector database of JAX/Flax reference documents that
MaxCode uses during migration. The database contains two types of documents:

  - Generic references (24 docs): JAX/Flax API docs, MaxText examples,
    flash-linear-attention implementations, and Flax attention patterns.

  - Targeted patterns (24 docs): WRONG/CORRECT/WHY examples for common
    conversion mistakes like incorrect cosine similarity, wrong einsum
    dimensions, missing weight initialization, and broken MoE routing.

Each document is embedded using Google's Gemini embedding model and stored
in a local SQLite database. During migration (Step 3), MaxCode retrieves
the most relevant documents for each file being converted.

Requires: GOOGLE_API_KEY environment variable.

Usage:
    python step2_populate_rag.py
"""

import os
import time
from config import RAG_SOURCE_DIR, setup, require_api_key

def main():
    api_key = require_api_key()
    setup()

    import models
    from agents.migration.primary_agent import PrimaryAgent
    from rag import vector_db

    print("=" * 70)
    print("Step 2: Populate RAG Database")
    print("=" * 70)
    print(f"  Source: {RAG_SOURCE_DIR}")
    print()

    # Count docs by category
    generic = targeted = 0
    for root, _, files in os.walk(RAG_SOURCE_DIR):
        for f in files:
            if not f.endswith(".py"):
                continue
            if "targeted" in f:
                targeted += 1
            else:
                generic += 1
    print(f"  Reference documents: {generic} generic + {targeted} targeted = {generic + targeted} total")
    print()

    # Clear old database and rebuild
    db_path = vector_db.RAG_DB_FILE
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"  Cleared old database: {db_path}")

    gemini_flash = models.GeminiTool(
        model_name=models.GeminiModel.GEMINI_2_5_FLASH,
        api_key=api_key,
    )

    # PrimaryAgent initializes the RAG agent internally
    agent = PrimaryAgent(model=gemini_flash, api_key=api_key)

    print(f"\n  Embedding documents (this takes ~1-2 minutes)...\n")
    t0 = time.time()
    agent._rag_agent.build_from_directory(RAG_SOURCE_DIR)
    elapsed = time.time() - t0

    # Verify
    ids, names, texts, files, embeddings = vector_db.load_all_documents(db_path)
    print(f"\n  RAG database: {len(ids)} documents indexed in {elapsed:.1f}s")
    print(f"  Database path: {db_path}")
    print("\nStep 2 complete.")


if __name__ == "__main__":
    main()
