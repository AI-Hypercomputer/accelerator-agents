"""Step 4 (API version): Convert the merged PyTorch model to JAX using MaxCode Python API.

This script tests the new high-level Python API defined in interface/api.py.
"""

import logging
import os
import time
from config import MERGED_FILE, OUTPUT_DIR, setup, require_api_key
from interface import api
import models

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)


def main():
  api_key = require_api_key()
  setup()

  # Pre-flight checks
  if not os.path.isfile(MERGED_FILE):
    print("ERROR: Merged model file not found. Run step3_merge.py first.")
    raise SystemExit(1)

  print("=" * 70)
  print("Step 4 (API): Convert PyTorch to JAX")
  print("=" * 70)
  print(f"  Source: {MERGED_FILE}")
  print(f"  Output: {OUTPUT_DIR}")
  print()

  # Find best available model
  migration_model_name = models.GeminiModel.GEMINI_3_1_PRO_PREVIEW.value
  for model_enum in [
      models.GeminiModel.GEMINI_3_1_PRO_PREVIEW,
      models.GeminiModel.GEMINI_2_5_PRO,
      models.GeminiModel.GEMINI_2_5_FLASH,
  ]:
    try:
      candidate = models.GeminiTool(model_name=model_enum, api_key=api_key)
      candidate("test")
      migration_model_name = model_enum.value
      print(f"  Using model: {migration_model_name}")
      break
    except Exception:
      continue

  config = api.ConvertConfig(
      source_path=MERGED_FILE,
      destination=OUTPUT_DIR,
      api_key=api_key,
      model_name=migration_model_name,
      validate=True,
  )

  print(f"\n  Converting via API (this may take several minutes)...\n")
  t0 = time.time()
  result = api.convert(config)
  elapsed = time.time() - t0

  print(f"\n  Migration completed in {elapsed:.1f}s")
  print(f"  Results saved to: {result.dest_path}")
  print(f"  Mapping file: {result.mapping_path}")

  if result.validation_path:
    print(f"  Validation results: {result.validation_path}")
  if result.verification_scorecard_path:
    print(f"  Verification scorecard: {result.verification_scorecard_path}")
    if result.verification_summary:
      print(f"  Summary score: {result.verification_summary}")

  print("\n" + "=" * 70)
  print("Done!")
  print("=" * 70)


if __name__ == "__main__":
  main()
