# MaxCode Demo: PyTorch to JAX Migration

End-to-end demo converting [Multimodal-Transformer](https://github.com/yaohungt/Multimodal-Transformer) from PyTorch to JAX using MaxCode.

## Prerequisites

- Python 3.12+
- A Google AI API key ([get one here](https://aistudio.google.com/apikey))

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv

# Linux / macOS / Git Bash
source venv/bin/activate

# Windows CMD
venv\Scripts\activate.bat

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your API key
# Linux / macOS / Git Bash
export GOOGLE_API_KEY=<your-key>

# Windows CMD
set GOOGLE_API_KEY=<your-key>
```

## Run the Demo

```bash
python convert_multimodal.py
```

## What It Does

1. **Clone** the Multimodal-Transformer repo from GitHub
2. **Merge** 4 source files into a single input file
3. **Populate** the RAG database with 46 JAX/Flax reference documents
4. **Migrate** PyTorch code to JAX/Flax using Gemini
5. **Validate** the output for faithfulness and auto-repair deviations
6. **Save** the final JAX output to `output/multimodal_transformer_jax.py`

## Output

After running, the converted JAX code is saved to:

```
output/multimodal_transformer_jax.py
```
