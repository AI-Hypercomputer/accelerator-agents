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

The demo is split into three steps. Run them in order:

```bash
# Step 1: Clone the PyTorch repo from GitHub
python step1_clone_repo.py

# Step 2: Build the RAG database with JAX/Flax reference docs
python step2_populate_rag.py

# Step 3: Convert to JAX with automatic validation and repair
python step3_convert.py
```

## What Each Step Does

### Step 1 — Clone Repository
Clones the Multimodal-Transformer repo and lists all Python files that
MaxCode will discover and convert. If already cloned, this step is skipped.

### Step 2 — Populate RAG Database
Builds a vector database of 46 JAX/Flax reference documents:
- **24 generic references**: Flax API docs, MaxText examples, attention patterns
- **22 targeted patterns**: WRONG/CORRECT/WHY examples for common conversion mistakes

Each document is embedded using Gemini and stored in a local SQLite database.
During conversion, MaxCode retrieves the most relevant documents for context.

### Step 3 — Convert to JAX
Runs the full migration pipeline:
1. Auto-discovers all `.py` files and builds a dependency graph
2. Converts each file in topological order using Gemini with RAG context
3. Validates each output against the PyTorch source for faithfulness
4. Auto-repairs any deviations (wrong init, dropped features, incorrect ops)
5. Saves converted files preserving the original directory structure

## Output

After running, the converted JAX files are in the `output/` directory,
mirroring the original repo structure.

## File Overview

| File | Purpose |
|------|---------|
| `config.py` | Shared paths and setup |
| `step1_clone_repo.py` | Clone the PyTorch repo |
| `step2_populate_rag.py` | Build the RAG reference database |
| `step3_convert.py` | Run migration + validation + repair |
| `requirements.txt` | Python dependencies |
