"""Agent for converting a PyTorch model into a MaxText artifact set.

The MaxText conversion is staged across several LLM calls so each one stays
focused: classify the source, emit a YAML config overlay, optionally emit a
custom layers `.py` file, and best-effort emit a checkpoint converter. The
canonical MaxText deliverable is a YAML overlay; the layers and converter
artifacts are produced only when the source warrants them.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agents import base
from agents import utils
from agents.migration.model_conversion_agent import _strip_markdown_formatting
from agents.migration.prompts import prompts
from rag import rag_agent

logger = logging.getLogger(__name__)


# Decoder block families recognised by MaxText. The classifier is asked to
# pick from this list (or "custom" if nothing fits). Kept here so the agent
# can defensively map LLM output onto the canonical set.
_KNOWN_DECODER_BLOCKS = (
    "llama2", "llama3", "llama4",
    "gemma", "gemma2", "gemma3",
    "mistral", "mixtral",
    "qwen3", "qwen3_next",
    "deepseek2", "deepseek3",
    "gpt_oss", "kimi",
    "default", "custom",
)


@dataclass
class MaxTextArtifacts:
  """Paths (and metadata) for the artifacts produced by the MaxText path.

  All path fields are populated by the persistence layer in `interface/api.py`
  after this agent has produced the corresponding string content; the agent
  itself only fills `decoder_block` and the in-memory artifact bodies via
  `MaxTextRunResult` below.
  """
  config_yaml_path: str
  layers_py_path: Optional[str] = None
  ckpt_converter_path: Optional[str] = None
  decoder_block: str = "default"


@dataclass
class MaxTextRunResult:
  """In-memory result of a MaxTextConversionAgent run.

  Holds the raw content of every artifact plus the classification metadata.
  The persistence layer turns this into a `MaxTextArtifacts` instance and a
  flat string-to-string `results` dict for the standard write path.
  """
  decoder_block: str
  justification: str
  config_yaml: str
  layers_py: Optional[str] = None
  ckpt_converter_py: Optional[str] = None
  model_name: str = "model"


def _strip_yaml_formatting(text: str) -> str:
  """Strips markdown fences from a YAML response."""
  match = re.search(r"```(?:yaml|yml)?\n?(.*?)\n?```", text, re.DOTALL)
  if match:
    return match.group(1).strip()
  stripped = text.strip()
  if stripped.startswith("```"):
    first_nl = stripped.find("\n")
    if first_nl != -1:
      stripped = stripped[first_nl + 1:]
    if stripped.endswith("```"):
      stripped = stripped[:-3]
    return stripped.strip()
  return stripped


def _extract_dim_hints(pytorch_code: str) -> Dict[str, Any]:
  """Best-effort scan for common config attributes on a PyTorch config class.

  Walks AST assignments that look like `self.<name> = <Constant>` inside any
  `__init__`. Used purely as a hint passed to the YAML prompt — the LLM is
  still expected to verify the values against the source. Returns an empty
  dict on parse failure.
  """
  hints: Dict[str, Any] = {}
  try:
    tree = ast.parse(pytorch_code)
  except SyntaxError:
    return hints

  interesting = {
      "hidden_size", "num_attention_heads", "num_key_value_heads",
      "num_hidden_layers", "vocab_size", "intermediate_size",
      "head_dim", "max_position_embeddings", "rms_norm_eps",
      "rope_theta", "tie_word_embeddings", "num_experts",
      "num_experts_per_tok", "moe_intermediate_size",
      "router_aux_loss_coef", "n_routed_experts", "n_shared_experts",
      "first_k_dense_replace", "moe_layer_freq",
  }

  for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
      if len(node.targets) != 1:
        continue
      tgt = node.targets[0]
      if not isinstance(tgt, ast.Attribute):
        continue
      if not (isinstance(tgt.value, ast.Name) and tgt.value.id == "self"):
        continue
      if tgt.attr not in interesting:
        continue
      if isinstance(node.value, ast.Constant):
        # Don't overwrite an earlier sighting — first wins (usually the
        # default-bearing assignment).
        hints.setdefault(tgt.attr, node.value.value)
  return hints


def _format_dim_hints(hints: Dict[str, Any]) -> str:
  """Pretty-prints the dim hints for the prompt body."""
  if not hints:
    return "(no hints extracted; derive everything from the source)"
  return "\n".join(f"- {k}: {v}" for k, v in sorted(hints.items()))


def _normalize_decoder_block(value: str) -> str:
  """Snaps an LLM-emitted decoder block onto the known set.

  Accepts variants like "Llama-2", "llama_2", "Llama2" and maps them onto
  the canonical "llama2".
  """
  if not value:
    return "default"
  v = value.strip().lower().replace("-", "_")
  if v in _KNOWN_DECODER_BLOCKS:
    return v
  # Compare with all separators stripped so "llama-2", "llama_2", "llama 2"
  # all match the canonical "llama2".
  v_compact = v.replace("_", "")
  for known in _KNOWN_DECODER_BLOCKS:
    if v_compact == known.replace("_", ""):
      return known
  return "custom"


def _parse_classification(text: str) -> Dict[str, str]:
  """Parses the classifier's JSON response. Falls back to {custom, ""} on error."""
  raw = text.strip()
  json_match = re.search(r"```(?:json)?\n?(.*?)\n?```", raw, re.DOTALL)
  if json_match:
    raw = json_match.group(1).strip()
  try:
    obj = json.loads(raw)
  except json.JSONDecodeError:
    obj_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not obj_match:
      logger.warning("Classifier returned unparseable response; defaulting to 'custom'")
      return {"decoder_block": "custom", "justification": ""}
    try:
      obj = json.loads(obj_match.group(0))
    except json.JSONDecodeError:
      logger.warning("Classifier JSON sub-extract failed; defaulting to 'custom'")
      return {"decoder_block": "custom", "justification": ""}

  if not isinstance(obj, dict):
    return {"decoder_block": "custom", "justification": ""}
  return {
      "decoder_block": _normalize_decoder_block(str(obj.get("decoder_block", ""))),
      "justification": str(obj.get("justification", "")),
  }


def _format_rag_context(docs: List[Dict[str, Any]]) -> str:
  """Formats RAG docs for inclusion in a prompt body."""
  if not docs:
    return "(no reference snippets available)"
  blocks = []
  for d in docs:
    name = d.get("name", "unknown")
    text = d.get("text", "")
    blocks.append(f"### {name}\n```python\n{text}\n```")
  return "\n\n".join(blocks)


class MaxTextConversionAgent(base.Agent):
  """Stages classify -> YAML -> (layers) -> (ckpt converter) for MaxText output."""

  def __init__(
      self,
      model: Any,
      rag_agent_instance: rag_agent.RAGAgent,
  ):
    """Initializes the agent.

    Args:
      model: The LLM model.
      rag_agent_instance: RAG agent (expected to have `target='maxtext'`).
    """
    super().__init__(
        model=model,
        agent_domain=utils.AgentDomain.MIGRATION,
        agent_type=utils.AgentType.MODEL_CONVERSION,
    )
    self._rag_agent = rag_agent_instance

  # ---- Stage 1: classify -------------------------------------------------

  def _classify(self, pytorch_code: str) -> Dict[str, str]:
    """Picks the closest existing MaxText `decoder_block` for the source."""
    docs = self._rag_agent.retrieve_per_component_context(pytorch_code)
    rag_context = _format_rag_context(docs)
    prompt = prompts.get_prompt("MAXTEXT_CLASSIFY_PROMPT", "maxtext")
    response = self.generate(
        prompt,
        {"pytorch_code": pytorch_code, "rag_context": rag_context},
    )
    return _parse_classification(response)

  # ---- Stage 2: YAML overlay --------------------------------------------

  def _emit_yaml(
      self,
      pytorch_code: str,
      decoder_block: str,
      justification: str,
  ) -> str:
    """Emits the YAML config overlay for `MaxText/configs/models/`."""
    docs = self._rag_agent.retrieve_context(
        f"MaxText config overlay {decoder_block}", top_k=10
    )
    rag_context = _format_rag_context(docs)
    dim_hints = _format_dim_hints(_extract_dim_hints(pytorch_code))
    prompt = prompts.get_prompt("MAXTEXT_YAML_PROMPT", "maxtext")
    response = self.generate(
        prompt,
        {
            "pytorch_code": pytorch_code,
            "rag_context": rag_context,
            "decoder_block": decoder_block,
            "justification": justification,
            "dim_hints": dim_hints,
        },
    )
    return _strip_yaml_formatting(response)

  # ---- Stage 3 (conditional): custom layers file ------------------------

  def _emit_layers(
      self,
      pytorch_code: str,
      justification: str,
  ) -> Optional[str]:
    """Emits a small layers `.py` file when the architecture is custom."""
    docs = self._rag_agent.retrieve_per_component_context(pytorch_code)
    rag_context = _format_rag_context(docs)
    prompt = prompts.get_prompt("MAXTEXT_LAYERS_PROMPT", "maxtext")
    response = self.generate(
        prompt,
        {
            "pytorch_code": pytorch_code,
            "rag_context": rag_context,
            "justification": justification,
            "maxtext_best_practices": prompts.MAXTEXT_BEST_PRACTICES,
        },
    )
    code = _strip_markdown_formatting(response)
    if not code or len(code.strip()) < 40:
      logger.warning("MaxText layers stage returned suspiciously short output; skipping")
      return None
    return code

  # ---- Stage 4 (best-effort): checkpoint converter -----------------------

  def _emit_ckpt_converter(
      self,
      pytorch_code: str,
      decoder_block: str,
      yaml_config: str,
  ) -> Optional[str]:
    """Best-effort: emit a HF/PyTorch -> Orbax converter. Errors are swallowed."""
    try:
      docs = self._rag_agent.retrieve_context(
          f"MaxText checkpoint converter {decoder_block} state dict orbax",
          top_k=8,
      )
      rag_context = _format_rag_context(docs)
      prompt = prompts.get_prompt("MAXTEXT_CKPT_CONVERTER_PROMPT", "maxtext")
      response = self.generate(
          prompt,
          {
              "pytorch_code": pytorch_code,
              "rag_context": rag_context,
              "decoder_block": decoder_block,
              "yaml_config": yaml_config,
          },
      )
      code = _strip_markdown_formatting(response)
      if not code or len(code.strip()) < 60:
        logger.info("Checkpoint converter stage returned trivial output; skipping")
        return None
      return code
    except Exception as e:
      logger.warning("Checkpoint converter stage failed (best-effort): %s", e)
      return None

  # ---- Orchestration -----------------------------------------------------

  def run(
      self,
      pytorch_code: str,
      model_name: str = "model",
  ) -> MaxTextRunResult:
    """Runs all stages and returns the populated `MaxTextRunResult`.

    Args:
      pytorch_code: The merged or single-file PyTorch source.
      model_name: Stem used for output filenames (e.g. "qwen3_next").

    Returns:
      A `MaxTextRunResult` ready for persistence.
    """
    cls = self._classify(pytorch_code)
    decoder_block = cls["decoder_block"]
    justification = cls["justification"]
    logger.info("MaxText classification: decoder_block=%s, justification=%s",
                decoder_block, justification)

    yaml_config = self._emit_yaml(pytorch_code, decoder_block, justification)

    layers_py: Optional[str] = None
    if decoder_block == "custom":
      layers_py = self._emit_layers(pytorch_code, justification)

    ckpt_converter_py = self._emit_ckpt_converter(
        pytorch_code, decoder_block, yaml_config
    )

    return MaxTextRunResult(
        decoder_block=decoder_block,
        justification=justification,
        config_yaml=yaml_config,
        layers_py=layers_py,
        ckpt_converter_py=ckpt_converter_py,
        model_name=model_name,
    )
