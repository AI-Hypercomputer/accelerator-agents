# pylint: skip-file
# Gemma 4 model re-exports.
#
# Upstream source of truth: HuggingFace `transformers` ships Gemma 4 modeling
# code at `src/transformers/models/gemma4/` (confirmed 2026-04-22 on
# https://github.com/huggingface/transformers/tree/main/src/transformers/models/gemma4).
# We do NOT copy 2000+ lines of modeling here. Autoresearch iterations that
# need to modify the forward pass should subclass or monkey-patch in this
# folder.
#
# Architecture notes (from google/gemma-4-E4B config.json, 2026-04-22):
#   - model_type: "gemma4"
#   - architectures: ["Gemma4ForConditionalGeneration"] (multimodal)
#   - text-only causal LM class: `Gemma4ForCausalLM`
#   - 42 layers, hidden=2560, heads=8, kv_heads=2, head_dim=256
#   - intermediate_size=10240, vocab_size=262144, SW=512, ctx=128K
#   - Hybrid attention: interleaved sliding-window + global layers.
#   - KV sharing across 18 layers (`num_kv_shared_layers`).
#   - `final_logit_softcapping=30.0`, gelu_pytorch_tanh activation.
#
# For performance baselining we use the text-only causal LM path.

try:
  from transformers import (
      Gemma4Config,
      Gemma4ForCausalLM,
      Gemma4ForConditionalGeneration,
      Gemma4Model,
  )
except ImportError as e:  # pragma: no cover - surfaced at run time
  raise ImportError(
      "Gemma 4 requires a transformers version that ships "
      "`models/gemma4`. See requirements.txt."
  ) from e

# Tokenizer / processor: the model card uses `AutoProcessor` (multimodal) and
# `AutoTokenizer` (text-only). We expose both so train.py can pick.
from transformers import AutoProcessor, AutoTokenizer  # noqa: E402

__all__ = [
    "Gemma4Config",
    "Gemma4ForCausalLM",
    "Gemma4ForConditionalGeneration",
    "Gemma4Model",
    "AutoProcessor",
    "AutoTokenizer",
]
