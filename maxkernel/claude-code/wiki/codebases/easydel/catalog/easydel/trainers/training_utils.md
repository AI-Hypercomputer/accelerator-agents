---
title: 'Module: easydel/trainers/training_utils.py'
type: catalog
provenance: extracted
module: easydel/trainers/training_utils.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.training_utils`/
symbols:
  update_state_respectfully: update_state_respectfully().
  minibatch_call: minibatch_call().
  update_metrics: update_metrics().
  make_assertions_and_get_sizes: make_assertions_and_get_sizes().
  resolve_straight_through_emulator: resolve_straight_through_emulator().
  filter_kwargs_for_callable: filter_kwargs_for_callable().
  sanitize_model_call_kwargs: sanitize_model_call_kwargs().
  make_default_tensor_straight_through: make_default_tensor_straight_through().
  normalize_generation_model_kwargs: normalize_generation_model_kwargs().
  compact_generation_model_kwargs: compact_generation_model_kwargs().
  prepare_generation_model_kwargs_for_call: prepare_generation_model_kwargs_for_call().
  validate_prompt_aligned_generation_model_kwargs: validate_prompt_aligned_generation_model_kwargs().
  extract_generation_model_kwargs: extract_generation_model_kwargs().
  repeat_prompt_aligned_model_kwargs: repeat_prompt_aligned_model_kwargs().
  slice_prompt_aligned_model_kwargs: slice_prompt_aligned_model_kwargs().
  strip_prompt_only_scoring_model_kwargs: strip_prompt_only_scoring_model_kwargs().
  GENERATION_MODEL_INPUT_KEYS: GENERATION_MODEL_INPUT_KEYS.
  make_default_tensor_straight_through.tensor_straight_through: make_default_tensor_straight_through().tensor_straight_through().
  minibatch_call.accumulate_gradients: minibatch_call().accumulate_gradients().
  minibatch_call.slice_minibatch: minibatch_call().slice_minibatch().
  FAST_COMPILE: FAST_COMPILE.
  _flatten_grouped_multimodal_model_value: _flatten_grouped_multimodal_model_value().
  _extend_prompt_scoring_value_to_sequence_length: _extend_prompt_scoring_value_to_sequence_length().
  repeat_prompt_aligned_model_value: repeat_prompt_aligned_model_value().
  slice_prompt_aligned_model_value: slice_prompt_aligned_model_value().
  FIXED_QUANTIZATION_BITS_BY_MODE.FIXED_QUANTIZATION_BITS_BY_MODE: FIXED_QUANTIZATION_BITS_BY_MODE.FIXED_QUANTIZATION_BITS_BY_MODE.
  QuantizationMode: QuantizationMode.
  SHARED_GENERATION_MODEL_INPUT_KEYS: SHARED_GENERATION_MODEL_INPUT_KEYS.
  infer_prompt_batch_axis: infer_prompt_batch_axis().
  SCAN_TRAINER: SCAN_TRAINER.
  AFFINE_SUPPORTED_BITS: AFFINE_SUPPORTED_BITS.
  PROMPT_ONLY_SCORING_MODEL_INPUT_KEYS: PROMPT_ONLY_SCORING_MODEL_INPUT_KEYS.
  _infer_batch_size: _infer_batch_size().
  GROUPED_MULTIMODAL_MODEL_INPUT_KEYS: GROUPED_MULTIMODAL_MODEL_INPUT_KEYS.
  PROMPT_SCORING_EXTENSION_KEYS: PROMPT_SCORING_EXTENSION_KEYS.
  _ste: _ste().
  make_default_tensor_straight_through._quantize_dequantize: make_default_tensor_straight_through()._quantize_dequantize().
  resolve_straight_through_emulator._default_emulator: resolve_straight_through_emulator()._default_emulator().
  update_state_respectfully.update_fn: update_state_respectfully().update_fn().
  update_state_respectfully.skip_fn: update_state_respectfully().skip_fn().
  minibatch_call.slice_minibatch._slice_leaf: minibatch_call().slice_minibatch()._slice_leaf().
  resolve_total_steps: resolve_total_steps().
---
# Module: [`easydel/trainers/training_utils.py`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_utils.py)

## Functions
- `_default_emulator(graphstate: tp.Any)` — [`L689`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_utils.py#L689)
- `_extend_prompt_scoring_value_to_sequence_length(key: str, value: tp.Any, *, prompt_length: int | None, target_sequence_length: int | None)` — [`L223`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_utils.py#L223) — Extend prompt-only token-type style tensors across generated text tokens.
- `_flatten_grouped_multimodal_model_value(key: str, value: tp.Any)` — [`L203`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_utils.py#L203) — Flatten grouped multimodal leaves before the actual model call.
- `_infer_batch_size(batch: tp.Any)` — [`L768`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_utils.py#L768) — Infer batch size from the most common leading dimension in the batch pytree.
- `_quantize_dequantize(y: jax.Array)` — [`L599`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_utils.py#L599)
- `_slice_leaf(arr)` — [`L891`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_utils.py#L891)
- `_ste(x: jax.Array, q: jax.Array)` — [`L534`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_utils.py#L534)
- `accumulate_gradients(acc, start_index)` — [`L913`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_utils.py#L913) — Accumulate gradients and metrics for each minibatch.
- `compact_generation_model_kwargs(kwargs: collections.abc.Mapping[str, tp.Any] | None)` — [`L186`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_utils.py#L186) — Drop ``None`` leaves from normalized generation model kwargs.
- `extract_generation_model_kwargs(batch: collections.abc.Mapping[str, tp.Any] | None, *, model_callable: tp.Callable[..., tp.Any] | None = None)` — [`L305`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_utils.py#L305) — Extract generation-related model inputs from a larger batch mapping.
- `filter_kwargs_for_callable(callable_obj: tp.Callable[..., tp.Any], kwargs: collections.abc.Mapping[str, tp.Any])` — [`L51`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_utils.py#L51) — Filter kwargs so only parameters accepted by ``callable_obj`` are forwarded.
- `infer_prompt_batch_axis(value: tp.Any, prompt_batch_size: int | None, *, key: str | None = None)` — [`L362`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_utils.py#L362) — Infer which axis of ``value`` is aligned with the prompt batch.
- `make_assertions_and_get_sizes(batch: dict, gradient_accumulation_steps: int, batch_partition_spec: PartitionSpec | None = None)` — [`L732`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_utils.py#L732) — Validates the input parameters and computes the batch size, minibatch size, and batch partition specification.
- `make_default_tensor_straight_through(quantization_mode: QuantizationMode, quantization_group_size: int | None = None, quantization_bits: int | None = None, *, quantization_block: int | None = None)` — [`L539`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_utils.py#L539) — Create a per-tensor STE quantization function.
- `minibatch_call(state: EasyDeLState, batch: dict, minibatch_size: int, grad_fn: tp.Callable[[jax.Array, dict], tuple[jax.Array, LossMetrics]])` — [`L860`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_utils.py#L860) — Processes batch in smaller chunks for gradient accumulation using jax.lax.scan.
- `normalize_generation_model_kwargs(kwargs: collections.abc.Mapping[str, tp.Any] | None, *, model_callable: tp.Callable[..., tp.Any] | None = None)` — [`L152`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_utils.py#L152) — Normalize model-side generation kwargs to a stable key set.
- `prepare_generation_model_kwargs_for_call(kwargs: collections.abc.Mapping[str, tp.Any] | None, *, target_sequence_length: int | None = None, prompt_length: int | None = None, flatten_grouped_multimodal: bool = True)` — [`L246`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_utils.py#L246) — Prepare generation kwargs for a model call without losing prompt grouping upstream.
- `repeat_prompt_aligned_model_kwargs(kwargs: collections.abc.Mapping[str, tp.Any] | None, repeat_factor: int, *, prompt_batch_size: int | None = None)` — [`L502`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_utils.py#L502) — Repeat prompt-aligned model kwargs to match completion-aligned batches.
- `repeat_prompt_aligned_model_value(value: tp.Any, repeat_factor: int, *, prompt_batch_size: int | None = None, key: str | None = None)` — [`L397`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_utils.py#L397) — Repeat a prompt-aligned model input along its actual batch axis.
- `resolve_straight_through_emulator(*, quantization_mode: QuantizationMode | None, quantization_group_size: int | None = None, quantization_bits: int | None = None, tensor_straight_through: tp.Callable[[jax.Array], jax.Array] | None, straight_through_emulator: tp.Callable[[tp.Any], tp.Any] | None, quantization_block: int | None = None)` — [`L642`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_utils.py#L642) — Resolve the graphstate-level straight-through emulator callable.
- `resolve_total_steps(*, forced_steps: int | None, total_data_len: int | None, batch_size: int, num_epochs: int, gradient_accumulation_steps: int, is_train: bool)` — [`L695`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_utils.py#L695) — Resolve total train/eval steps from config and dataset length.
- `sanitize_model_call_kwargs(kwargs: collections.abc.Mapping[str, tp.Any])` — [`L74`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_utils.py#L74) — Normalize model call kwargs to avoid known incompatible combinations.
- `skip_fn(args)` — [`L839`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_utils.py#L839)
- `slice_minibatch(tree, start_index)` — [`L888`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_utils.py#L888) — Extract one minibatch while leaving shared/global leaves untouched.
- `slice_prompt_aligned_model_kwargs(kwargs: collections.abc.Mapping[str, tp.Any], start: int, end: int, *, prompt_batch_size: int | None = None)` — [`L468`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_utils.py#L468) — Slice prompt-aligned model kwargs while preserving shared leaves.
- `slice_prompt_aligned_model_value(value: tp.Any, start: int, end: int, *, prompt_batch_size: int | None = None, key: str | None = None)` — [`L432`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_utils.py#L432) — Slice a prompt-aligned model input along its actual batch axis.
- `strip_prompt_only_scoring_model_kwargs(kwargs: collections.abc.Mapping[str, tp.Any] | None)` — [`L282`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_utils.py#L282) — Drop prompt-only sequence-control kwargs before full-sequence scoring.
- `tensor_straight_through(x: jax.Array)` — [`L634`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_utils.py#L634)
- `update_fn(args)` — [`L835`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_utils.py#L835)
- `update_metrics(metrics: LossMetrics, learning_rate_fn: tp.Callable, step: int | jax.Array, gradients: jax.Array | None)` — [`L783`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_utils.py#L783) — Updates the given metrics with the current learning rate and gradient norms.
- `update_state_respectfully(state: EasyDeLState, gradients: jax.Array, loss_config: LossConfig | None, metrics: LossMetrics)` — [`L813`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_utils.py#L813) — Updates the state of the model respectfully based on the provided gradients, loss configuration, and metrics.
- `validate_prompt_aligned_generation_model_kwargs(kwargs: collections.abc.Mapping[str, tp.Any] | None, *, prompt_batch_size: int | None)` — [`L327`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_utils.py#L327) — Validate that generation kwargs preserve prompt boundaries for GRPO-style scoring.

## Module values
- `AFFINE_SUPPORTED_BITS` — [`L41`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_utils.py#L41)
- `FAST_COMPILE` — [`L31`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_utils.py#L31)
- `FIXED_QUANTIZATION_BITS_BY_MODE` — [`L42`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_utils.py#L42)
- `GENERATION_MODEL_INPUT_KEYS` — [`L86`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_utils.py#L86)
- `GROUPED_MULTIMODAL_MODEL_INPUT_KEYS` — [`L121`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_utils.py#L121)
- `PROMPT_ONLY_SCORING_MODEL_INPUT_KEYS` — [`L141`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_utils.py#L141)
- `PROMPT_SCORING_EXTENSION_KEYS` — [`L132`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_utils.py#L132)
- `QuantizationMode` — [`L33`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_utils.py#L33)
- `SCAN_TRAINER` — [`L30`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_utils.py#L30)
- `SHARED_GENERATION_MODEL_INPUT_KEYS` — [`L114`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_utils.py#L114)

