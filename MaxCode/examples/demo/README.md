# MaxCode Demo: PyTorch to JAX Migration

End-to-end demo converting [Multimodal-Transformer](https://github.com/yaohungt/Multimodal-Transformer) from PyTorch to JAX/Flax using MaxCode.

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

The demo is split into four steps. Run them in order:

```bash
# Step 1: Clone the PyTorch repo from GitHub
python step1_clone_repo.py

# Step 2: Build the RAG database with JAX/Flax reference docs
python step2_populate_rag.py

# Step 3: Auto-detect model files and merge into a single file
python step3_merge.py

# Step 4: Convert to JAX with automatic validation and repair
python step4_convert.py
```

## What Each Step Does

### Step 1 — Clone Repository
Clones the Multimodal-Transformer repo and lists all Python files found.
If already cloned, this step is skipped.

### Step 2 — Populate RAG Database
Builds a vector database of 46 JAX/Flax reference documents:
- **24 generic references**: Flax API docs, MaxText examples, attention patterns
- **22 targeted patterns**: WRONG/CORRECT/WHY examples for common conversion mistakes

Each document is embedded using Gemini and stored in a local SQLite database.
During conversion, MaxCode retrieves the most relevant documents for context.

### Step 3 — Auto-Detect and Merge Model Files
Scans the repository to find all files that define `nn.Module` subclasses
(the actual model code). Non-model files like datasets, training scripts,
and utilities are automatically excluded. The detected model files are then
merged into a single file so the LLM has full context of all components
and their dependencies during conversion.

### Step 4 — Convert to JAX
Runs the full migration pipeline on the merged model file:
1. Converts PyTorch code to JAX/Flax using Gemini with RAG context
2. Validates the output against the PyTorch source for faithfulness
3. Auto-repairs any deviations (wrong init, dropped features, incorrect ops)
4. Saves the final JAX file

## Output

After running, the converted JAX file is saved to:
```
output/multimodal_transformer_jax.py
```

## File Overview

| File | Purpose |
|------|---------|
| `config.py` | Shared paths and setup |
| `step1_clone_repo.py` | Clone the PyTorch repo |
| `step2_populate_rag.py` | Build the RAG reference database |
| `step3_merge.py` | Auto-detect model files and merge |
| `step4_convert.py` | Run migration + validation + repair |
| `requirements.txt` | Python dependencies |
