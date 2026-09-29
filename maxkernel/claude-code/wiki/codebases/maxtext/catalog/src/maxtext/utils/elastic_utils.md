---
title: 'Module: src/maxtext/utils/elastic_utils.py'
type: catalog
provenance: extracted
module: src/maxtext/utils/elastic_utils.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.utils.elastic_utils`/
symbols:
  elastic_retry: elastic_retry().
  live_devices: live_devices().
  is_scale_up_event: is_scale_up_event().
  elastic_manager.elastic_manager: elastic_manager.elastic_manager.
  ensure_elastic_manager_initialized: ensure_elastic_manager_initialized().
  pending_reinit_recorder: pending_reinit_recorder.
  pending_elastic_event_type: pending_elastic_event_type.
  record_elastic_event_start: record_elastic_event_start().
  record_elastic_wait_end_and_reinit_start: record_elastic_wait_end_and_reinit_start().
  should_use_elastic: should_use_elastic().
  get_devices_per_host: get_devices_per_host().
  chain_callbacks: chain_callbacks().
  elastic_enabled: elastic_enabled().
  record_elastic_reinit_end: record_elastic_reinit_end().
  get_local_batch_size: get_local_batch_size().
  live_slice_indices: live_slice_indices().
  maybe_elastic_scale_up: maybe_elastic_scale_up().
  clean_up_checkpoints: clean_up_checkpoints().
  chain_callbacks.wrapper: chain_callbacks().wrapper().
---
# Module: [`src/maxtext/utils/elastic_utils.py`](../../../../../../../raw/code/maxtext/src/maxtext/utils/elastic_utils.py)

## Functions
- `chain_callbacks(*funcs)` — [`L151`](../../../../../../../raw/code/maxtext/src/maxtext/utils/elastic_utils.py#L151) — Helper function to chain callbacks.
- `clean_up_checkpoints(checkpoint_dir: str)` — [`L71`](../../../../../../../raw/code/maxtext/src/maxtext/utils/elastic_utils.py#L71) — Cleans up incomplete checkpoints after an elastic event.
- `elastic_enabled(config)` — [`L61`](../../../../../../../raw/code/maxtext/src/maxtext/utils/elastic_utils.py#L61) — Returns whether elastic mode is enabled.
- `elastic_retry(config, callback_fn=None, pre_callback_fn=None)` — [`L161`](../../../../../../../raw/code/maxtext/src/maxtext/utils/elastic_utils.py#L161) — Decorator for elastic retry.
- `ensure_elastic_manager_initialized(config)` — [`L105`](../../../../../../../raw/code/maxtext/src/maxtext/utils/elastic_utils.py#L105) — Initializes elastic manager if it's not initialized and pathways is used.
- `get_devices_per_host(config)` — [`L135`](../../../../../../../raw/code/maxtext/src/maxtext/utils/elastic_utils.py#L135) — Dynamically calculates the number of chips per physical worker VM.
- `get_local_batch_size(config)` — [`L112`](../../../../../../../raw/code/maxtext/src/maxtext/utils/elastic_utils.py#L112) — Returns the local batch size based on the config.
- `is_scale_up_event(config)` — [`L208`](../../../../../../../raw/code/maxtext/src/maxtext/utils/elastic_utils.py#L208) — Returns whether a scale up event is detected.
- `live_devices(config=None)` — [`L117`](../../../../../../../raw/code/maxtext/src/maxtext/utils/elastic_utils.py#L117) — Returns the list of live devices.
- `live_slice_indices(config)` — [`L130`](../../../../../../../raw/code/maxtext/src/maxtext/utils/elastic_utils.py#L130) — Returns the set of live slice indices.
- `maybe_elastic_scale_up(config, checkpoint_manager)` — [`L218`](../../../../../../../raw/code/maxtext/src/maxtext/utils/elastic_utils.py#L218) — Waits for a checkpoint to finish before interrupting for scale up.
- `record_elastic_event_start(recorder, config)` — [`L31`](../../../../../../../raw/code/maxtext/src/maxtext/utils/elastic_utils.py#L31) — Records start of an elastic scale up event.
- `record_elastic_reinit_end()` — [`L53`](../../../../../../../raw/code/maxtext/src/maxtext/utils/elastic_utils.py#L53) — Records end of elastic reinitialization event.
- `record_elastic_wait_end_and_reinit_start(recorder)` — [`L40`](../../../../../../../raw/code/maxtext/src/maxtext/utils/elastic_utils.py#L40) — Records end of elastic slice event and start of reinitialization event.
- `should_use_elastic(config)` — [`L66`](../../../../../../../raw/code/maxtext/src/maxtext/utils/elastic_utils.py#L66) — Returns whether elastic training should be used.
- `wrapper()` — [`L154`](../../../../../../../raw/code/maxtext/src/maxtext/utils/elastic_utils.py#L154)

## Module values
- `elastic_manager` — [`L26`](../../../../../../../raw/code/maxtext/src/maxtext/utils/elastic_utils.py#L26)
- `pending_elastic_event_type` — [`L28`](../../../../../../../raw/code/maxtext/src/maxtext/utils/elastic_utils.py#L28)
- `pending_reinit_recorder` — [`L27`](../../../../../../../raw/code/maxtext/src/maxtext/utils/elastic_utils.py#L27)

