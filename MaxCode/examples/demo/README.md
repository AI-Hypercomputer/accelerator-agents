# MaxCode Demo: PyTorch to JAX Migration

End-to-end demo converting any PyTorch repository to JAX/Flax using MaxCode. By default it converts [Multimodal-Transformer](https://github.com/yaohungt/Multimodal-Transformer), but you can point it at any repo.

## Prerequisites

- Python 3.12+
- A Google AI API key ([get one here](https://aistudio.google.com/apikey))

## Setup

```bash
# Create and activate a virtual environment
python -m venv venv

# Linux / macOS / Git Bash
source venv/bin/activate

# Windows CMD
venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt

# Set your API key
export GOOGLE_API_KEY=<your-key>          # Linux / macOS / Git Bash
set GOOGLE_API_KEY=<your-key>             # Windows CMD
```

## Run the Demo

The demo is split into five steps. Run them in order:

```bash
# Step 1: Clone the PyTorch repo from GitHub
python step1_clone_repo.py                      # default: Multimodal-Transformer
python step1_clone_repo.py https://github.com/openai/whisper   # or any repo

# Step 2: Build the RAG database with JAX/Flax reference docs
python step2_populate_rag.py

# Step 3: Auto-detect model files, filter by import graph, and merge
python step3_merge.py

# Step 4: Convert to JAX with automatic validation and repair
python step4_convert.py
# -- OR convert to MaxText (YAML config + JAX layers + checkpoint converter) --
python step4_convert_maxtext.py

# Step 5: Verify conversion quality (scorecard)
python step5_verify.py
```

## What Each Step Does

### Step 1 — Clone Repository
Clones the target PyTorch repo and lists all Python files found.
Accepts an optional URL argument (defaults to Multimodal-Transformer).
The chosen URL is saved to `.repo_url` so subsequent steps (3-5)
automatically use the same repo without needing to set an environment
variable. If already cloned, this step is skipped.

### Step 2 — Populate RAG Database
Builds a vector database of JAX/Flax reference documents:
- **Generic references**: Flax API docs, MaxText examples, attention patterns
- **Targeted patterns**: WRONG/CORRECT/WHY examples for common conversion mistakes
  (detach/stop_gradient, dtype casts, dead code, initialization consistency,
  bare-layer initializer faithfulness, sum-vs-mean reduction correctness, etc.)

Each document is embedded using Gemini and stored in a local SQLite database.
During conversion, MaxCode retrieves the most relevant documents for context.

### Step 3 — Auto-Detect, Filter, and Merge Model Files
Scans the repository to find all files that define `nn.Module` subclasses
(the actual model code). Non-model files like datasets, training scripts,
and utilities are automatically excluded.

An import-graph analysis then filters out dead-code modules — files that
contain `nn.Module` classes but are never transitively imported by the main
model entry point. Only files reachable from the entry point are included
in the merge. This prevents unused code from confusing the LLM during
conversion.

The remaining files are merged in dependency order (leaves first, entry
point last) so classes are defined before they are used.

### Step 4 — Convert to JAX
Runs the full migration pipeline on the merged model file:
1. Converts PyTorch code to JAX/Flax using Gemini with RAG context
2. Validates the output against the PyTorch source for faithfulness
3. Auto-repairs any deviations (wrong init, dropped features, incorrect ops)
4. Saves the final JAX file

### Step 4 (MaxText) — Convert to MaxText
An alternative to the plain JAX path that targets Google's
[MaxText](https://github.com/AI-Hypercomputer/maxtext) TPU training stack.
Produces up to three artifacts:
- **YAML config overlay** — always emitted; maps model dimensions onto MaxText's
  config schema.
- **JAX layers file** — whether this file is emitted depends on the classifier
  result:
  - *Known block with built-in MaxText implementation* (e.g. `llama3`, `gemma2`):
    **not emitted** — MaxText already has the JAX code, so the YAML overlay alone
    is enough.
  - *Known block without a built-in implementation* (e.g. `qwen3_next`):
    **emitted** — the block is recognised but MaxText has no JAX code for it yet,
    so a layers file is generated.
  - *No known block matches*: **emitted** — the architecture is novel, so a full
    custom layers file is generated.
- **Checkpoint converter** — best-effort script to convert HuggingFace / PyTorch
  weights into an Orbax checkpoint consumable by MaxText.

### Step 5 — Verify Conversion Quality
Produces a scorecard measuring how complete and correct the conversion is:
- **Completeness** (AST-based, no LLM): compares classes, methods, and
  standalone functions between the PyTorch source and JAX output by name.
- **Correctness** (LLM-based, optional): runs the ValidationAgent to detect
  deviations and computes a weighted score (high=5, medium=3, low=1 penalty
  per deviation). Known false positives — low-severity `method_placement`,
  `missing_component`, and `dropped_feature` deviations that represent
  legitimate Flax idioms — are automatically filtered out of the score.

If `GOOGLE_API_KEY` is not set, the correctness check is skipped and only
the completeness score is reported. Results (including full deviation details
and filtered false positives) are saved to `output/verification_scorecard.json`.

## Output

After running, results are saved to a timestamped subdirectory under `output/`.

**JAX path** (`step4_convert.py`):
```
output/<timestamp>/<repo_name>_jax.py
```

**MaxText path** (`step4_convert_maxtext.py`):
```
output/<timestamp>/MaxText/configs/models/<model>.yml    # YAML config overlay
output/<timestamp>/MaxText/layers/<model>.py             # JAX layers (when applicable)
output/<timestamp>/utils/convert_<model>_ckpt.py         # checkpoint converter
```

## File Overview

| File | Purpose |
|------|---------|
| `config.py` | Shared paths and setup (resolves repo URL from env var, `.repo_url` file, or default) |
| `step1_clone_repo.py` | Clone any PyTorch repo (accepts optional URL argument) |
| `step2_populate_rag.py` | Build the RAG reference database |
| `step3_merge.py` | Auto-detect model files, filter by import graph, and merge |
| `step4_convert.py` | Run migration + validation + repair |
| `step4_convert_maxtext.py` | Convert to MaxText (YAML + layers + ckpt converter) |
| `step5_verify.py` | Verify conversion quality (scorecard) |
| `requirements.txt` | Python dependencies |
