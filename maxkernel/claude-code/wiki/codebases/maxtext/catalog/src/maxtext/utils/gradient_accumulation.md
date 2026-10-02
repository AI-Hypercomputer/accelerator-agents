---
title: 'Module: src/maxtext/utils/gradient_accumulation.py'
type: catalog
provenance: extracted
module: src/maxtext/utils/gradient_accumulation.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.utils.gradient_accumulation`/
symbols:
  gradient_accumulation_loss_and_grad: gradient_accumulation_loss_and_grad().
  gradient_accumulation_loss_and_grad._maybe_shard_with_name: gradient_accumulation_loss_and_grad()._maybe_shard_with_name().
  gradient_accumulation_loss_and_grad.convert_to_bf16: gradient_accumulation_loss_and_grad().convert_to_bf16().
  gradient_accumulation_loss_and_grad.accumulate_gradient: gradient_accumulation_loss_and_grad().accumulate_gradient().
  gradient_accumulation_loss_and_grad.reshape_to_microbatch_accumulations: gradient_accumulation_loss_and_grad().reshape_to_microbatch_accumulations().
  update_sharding_for_unreduced: update_sharding_for_unreduced().
  update_sharding_for_reduced: update_sharding_for_reduced().
---
# Module: [`src/maxtext/utils/gradient_accumulation.py`](../../../../../../../raw/code/maxtext/src/maxtext/utils/gradient_accumulation.py)

## Functions
- `_maybe_shard_with_name(inputs, sharding_names)` — [`L67`](../../../../../../../raw/code/maxtext/src/maxtext/utils/gradient_accumulation.py#L67) — Wrapper of maybe_shard_with_name with fixed shard_mode
- `accumulate_gradient(acc_grad_and_loss, data)` — [`L110`](../../../../../../../raw/code/maxtext/src/maxtext/utils/gradient_accumulation.py#L110)
- `convert_to_bf16(param)` — [`L95`](../../../../../../../raw/code/maxtext/src/maxtext/utils/gradient_accumulation.py#L95)
- `gradient_accumulation_loss_and_grad(_loss_fn, config, model, params, params_shardings, data, dropout_rng)` — [`L27`](../../../../../../../raw/code/maxtext/src/maxtext/utils/gradient_accumulation.py#L27) — Calculates gradients using gradient accumulation.
- `reshape_to_microbatch_accumulations(batch_arr)` — [`L140`](../../../../../../../raw/code/maxtext/src/maxtext/utils/gradient_accumulation.py#L140) — Reshape global batch to microbatches, assuming batch axis is leading.
- `update_sharding_for_reduced(sharding: NamedSharding)` — [`L189`](../../../../../../../raw/code/maxtext/src/maxtext/utils/gradient_accumulation.py#L189) — Add reduced on data axis of given NamedSharding
- `update_sharding_for_unreduced(sharding: NamedSharding)` — [`L196`](../../../../../../../raw/code/maxtext/src/maxtext/utils/gradient_accumulation.py#L196) — Add unreduced on data axis of given NamedSharding

