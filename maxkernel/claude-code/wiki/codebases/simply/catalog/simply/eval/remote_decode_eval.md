---
title: 'Module: simply/eval/remote_decode_eval.py'
type: catalog
provenance: extracted
module: simply/eval/remote_decode_eval.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.eval.remote_decode_eval`/
symbols:
  main: main().
  main.query_and_evaluate: main().query_and_evaluate().
  simply_service_stub: simply_service_stub().
  main._stats_history: main()._stats_history().
  main._stats_all_history: main()._stats_all_history().
  _SERVER_ADDRESS: _SERVER_ADDRESS.
  _EXPERIMENT_DIR: _EXPERIMENT_DIR.
  _EVALUATION: _EVALUATION.
  _DATASOURCE_NAME: _DATASOURCE_NAME.
  _SAVE_EVERY_N: _SAVE_EVERY_N.
  _N_REPEATS: _N_REPEATS.
  get_last_file: get_last_file().
  _MAX_DECODE_STEPS: _MAX_DECODE_STEPS.
---
# Module: [`simply/eval/remote_decode_eval.py`](../../../../../../raw/code/simply/simply/eval/remote_decode_eval.py)

## Functions
- `_stats_all_history()` — [`L228`](../../../../../../raw/code/simply/simply/eval/remote_decode_eval.py#L228)
- `_stats_history(history_path: epath.PathLike)` — [`L214`](../../../../../../raw/code/simply/simply/eval/remote_decode_eval.py#L214)
- `get_last_file(directory: epath.PathLike, pattern: str)` — [`L81`](../../../../../../raw/code/simply/simply/eval/remote_decode_eval.py#L81) — Returns the last file that matches the pattern.
- `main(argv: Sequence[str])` — [`L95`](../../../../../../raw/code/simply/simply/eval/remote_decode_eval.py#L95)
- `query_and_evaluate(index: int, example: Mapping[str, Any])` — [`L133`](../../../../../../raw/code/simply/simply/eval/remote_decode_eval.py#L133) — Queries the server and evaluates the response.
- `simply_service_stub()` — [`L71`](../../../../../../raw/code/simply/simply/eval/remote_decode_eval.py#L71) — Returns the stub to the server.

## Module values
- `_DATASOURCE_NAME` — [`L55`](../../../../../../raw/code/simply/simply/eval/remote_decode_eval.py#L55)
- `_EVALUATION` — [`L51`](../../../../../../raw/code/simply/simply/eval/remote_decode_eval.py#L51)
- `_EXPERIMENT_DIR` — [`L41`](../../../../../../raw/code/simply/simply/eval/remote_decode_eval.py#L41)
- `_MAX_DECODE_STEPS` — [`L45`](../../../../../../raw/code/simply/simply/eval/remote_decode_eval.py#L45)
- `_N_REPEATS` — [`L66`](../../../../../../raw/code/simply/simply/eval/remote_decode_eval.py#L66)
- `_SAVE_EVERY_N` — [`L62`](../../../../../../raw/code/simply/simply/eval/remote_decode_eval.py#L62)
- `_SERVER_ADDRESS` — [`L37`](../../../../../../raw/code/simply/simply/eval/remote_decode_eval.py#L37)

