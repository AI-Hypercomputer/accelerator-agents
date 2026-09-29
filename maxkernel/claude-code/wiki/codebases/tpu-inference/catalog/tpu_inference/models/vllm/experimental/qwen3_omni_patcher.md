---
title: 'Module: tpu_inference/models/vllm/experimental/qwen3_omni_patcher.py'
type: catalog
provenance: extracted
module: tpu_inference/models/vllm/experimental/qwen3_omni_patcher.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.models.vllm.experimental.qwen3_omni_patcher`/
symbols:
  apply_qwen3_omni_patches: apply_qwen3_omni_patches().
  maybe_apply_qwen3_omni_patches: maybe_apply_qwen3_omni_patches().
  logger: logger.
  _patched_qwen3_omni_forward: _patched_qwen3_omni_forward().
  _apply_pad_sequence_patch: _apply_pad_sequence_patch().
  _patched_process_audio_input: _patched_process_audio_input().
  _apply_pad_sequence_patch.patched_pad_sequence: _apply_pad_sequence_patch().patched_pad_sequence().
  _patched_qwen3_omni_embed_input_ids: _patched_qwen3_omni_embed_input_ids().
  _patched_rot_pos_emb: _patched_rot_pos_emb().
  _patched_vision_transformer_forward: _patched_vision_transformer_forward().
  _get_feat_extract_output_lengths: _get_feat_extract_output_lengths().
  _patched_audio_tower_forward: _patched_audio_tower_forward().
  is_qwen3_omni: is_qwen3_omni().
---
# Module: [`tpu_inference/models/vllm/experimental/qwen3_omni_patcher.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/experimental/qwen3_omni_patcher.py)

## Functions
- `_apply_pad_sequence_patch()` — [`L395`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/experimental/qwen3_omni_patcher.py#L395) — Globally patch pad_sequence with a pure-PyTorch sliced implementation.
- `_get_feat_extract_output_lengths(input_lengths: tuple[int, ...])` — [`L264`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/experimental/qwen3_omni_patcher.py#L264)
- `_patched_audio_tower_forward(self, input_features: torch.Tensor, feature_lens, aftercnn_lens)` — [`L293`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/experimental/qwen3_omni_patcher.py#L293) — A completely JIT-friendly, trace-safe implementation of AudioTower forward.
- `_patched_process_audio_input(vllm_model, audio_input)` — [`L276`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/experimental/qwen3_omni_patcher.py#L276) — JIT-friendly processing of audio inputs, keeping dynamic values static.
- `_patched_qwen3_omni_embed_input_ids(vllm_model, input_ids: torch.Tensor, multimodal_embeddings=None, *, is_multimodal: torch.Tensor | None = None)` — [`L42`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/experimental/qwen3_omni_patcher.py#L42) — A completely JIT-friendly implementation of embed_input_ids that avoids .cpu() and item() syncs.
- `_patched_qwen3_omni_forward(vllm_model, input_ids: torch.Tensor | None, positions: torch.Tensor, intermediate_tensors: IntermediateTensors | None = None, inputs_embeds: torch.Tensor | None = None, **kwargs: object)` — [`L106`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/experimental/qwen3_omni_patcher.py#L106) — Pure JIT-friendly forward passing deepstack embeddings statelessly.
- `_patched_rot_pos_emb(self, grid_thw)` — [`L139`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/experimental/qwen3_omni_patcher.py#L139) — JIT-friendly rot_pos_emb avoiding item() calls and dynamic tensor slicing.
- `_patched_vision_transformer_forward(self, x: torch.Tensor, grid_thw)` — [`L184`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/experimental/qwen3_omni_patcher.py#L184) — JIT-friendly forward avoiding .cpu().numpy() and dynamic list slicing on device.
- `apply_qwen3_omni_patches(vllm_model)` — [`L457`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/experimental/qwen3_omni_patcher.py#L457) — Apply Qwen3-Omni specific patches for stateless Deepstack support and JIT vision tower.
- `is_qwen3_omni(vllm_model)` — [`L512`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/experimental/qwen3_omni_patcher.py#L512) — Check if the given vLLM model is of architecture Qwen3OmniMoeThinkerForConditionalGeneration.
- `maybe_apply_qwen3_omni_patches(vllm_model: nn.Module)` — [`L518`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/experimental/qwen3_omni_patcher.py#L518)
- `patched_pad_sequence(sequences, batch_first=False, padding_value=0, padding_side="right")` — [`L408`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/experimental/qwen3_omni_patcher.py#L408)

## Module values
- `logger` — [`L39`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/experimental/qwen3_omni_patcher.py#L39)

