# pylint: skip-file
"""Scan-over-layers wrapper for HuggingFace LlamaForCausalLM.

Pattern follows the canonical torchtitan example
(`raw/code/torchax/examples/train_llama_torchtitan/train_llama.py`):

  TransfomerWithScan(...) replaces the for-loop over decoder_layers with
  torchax.train.ScannedModule(layers, checkpoint_policy), which stacks the
  per-layer weights into shape (n_layers, ...) and uses jax.lax.scan to
  iterate. XLA's compile-time HBM analysis sees a single scan body's worth
  of buffers instead of the 32-layer-unrolled sum, often reducing the peak
  by several GiB at high seq/batch shapes.

Adapted to HF Llama: each LlamaDecoderLayer is wrapped in
`_WrappedDecoderLayer` to flatten the positional signature to
`(hidden, cos, sin)` (HF's forward has 7+ kwargs). attention_mask=None is
passed because the splash-attention override builds its own causal mask.
"""

from __future__ import annotations

import torch
import torchax
import torchax.train
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM,
)


class _WrappedDecoderLayer(torch.nn.Module):
  """Adapter: 3 positional args (hidden, cos, sin); returns hidden tensor.

  HF LlamaDecoderLayer's forward has many kwargs and accepts
  `position_embeddings` as a (cos, sin) tuple. ScannedModule's body calls
  `torch.func.functional_call(layer, weights, args)` which passes args
  positionally; flatten via this wrapper.
  """

  def __init__(self, layer):
    super().__init__()
    self.layer = layer

  def forward(self, hidden, cos, sin):
    # attention_mask=None — splash override builds its own causal mask.
    # Returns torch.Tensor (hidden_states); newer HF returns just the tensor.
    out = self.layer(
        hidden, attention_mask=None, position_embeddings=(cos, sin)
    )
    return out if not isinstance(out, tuple) else out[0]


class LlamaForCausalLMScan(torch.nn.Module):
  """Drop-in replacement for LlamaForCausalLM with scan-over-layers.

  State-dict layout (vs unscanned):

    Unscanned:
      model.layers.0.self_attn.q_proj.weight  (4096, 4096)
      model.layers.1.self_attn.q_proj.weight  (4096, 4096)
      ... × 32 layers

    Scanned (this class):
      scanned_layers.params.layer___self_attn___q_proj___weight  (32, 4096,
      4096)

  Use `model.sharding.SCAN_SHARDING_MAP` keyed on these stacked-name keys.
  """

  def __init__(self, config: LlamaConfig, checkpoint_policy=None):
    super().__init__()
    # Build the unscanned model first (on whatever default_dtype is set);
    # we copy the non-layer submodules and discard the layer ModuleList.
    orig = LlamaForCausalLM(config)
    self.config = config
    self.embed_tokens = orig.model.embed_tokens
    self.norm = orig.model.norm
    self.rotary_emb = orig.model.rotary_emb
    self.lm_head = orig.lm_head

    wrapped = [_WrappedDecoderLayer(l) for l in orig.model.layers]
    self.scanned_layers = torchax.train.ScannedModule(
        wrapped, checkpoint_policy
    )
    # If True, `forward` returns hidden_states pre-projection (the loss
    # path will apply lm_head + softmax + CE in a single fused kernel
    # via tokamax.linear_softmax_cross_entropy_loss).
    self.skip_lm_head = False

  def forward(self, input_ids):
    hidden = self.embed_tokens(input_ids)
    seq = input_ids.shape[1]
    position_ids = torch.arange(seq, device=input_ids.device).unsqueeze(0)
    cos, sin = self.rotary_emb(hidden, position_ids=position_ids)
    hidden = self.scanned_layers(hidden, cos, sin)
    hidden = self.norm(hidden)
    if self.skip_lm_head:
      return hidden
    return self.lm_head(hidden)


__all__ = ["LlamaForCausalLMScan"]
