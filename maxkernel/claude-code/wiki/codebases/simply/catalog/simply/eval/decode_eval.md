---
title: 'Module: simply/eval/decode_eval.py'
type: catalog
provenance: extracted
module: simply/eval/decode_eval.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.eval.decode_eval`/
symbols:
  main: main().
  main._init_fn: main()._init_fn().
  main._stats_history: main()._stats_history().
  main._stats_all_history: main()._stats_all_history().
  _BATCH_SIZE: _BATCH_SIZE.
  _EXPERIMENT_DIR: _EXPERIMENT_DIR.
  _VOCAB_NAME: _VOCAB_NAME.
  _ACTIVATION_DTYPE: _ACTIVATION_DTYPE.
  _EXPERIMENT_CONFIG: _EXPERIMENT_CONFIG.
  _MESH_SHAPE: _MESH_SHAPE.
  _CKPT_DIR: _CKPT_DIR.
  _CKPT_STEP: _CKPT_STEP.
  _CKPT_FORMAT: _CKPT_FORMAT.
  _SEED: _SEED.
  _PREFILL_SIZE: _PREFILL_SIZE.
  _MAX_SEQ_LEN: _MAX_SEQ_LEN.
  _MAX_DECODE_STEPS: _MAX_DECODE_STEPS.
  _INTERMEDIATE_DECODE_STEPS: _INTERMEDIATE_DECODE_STEPS.
  _LM_FORMAT: _LM_FORMAT.
  _EVALUATION: _EVALUATION.
  _DATASOURCE_NAME: _DATASOURCE_NAME.
  _SAVE_EVERY_N: _SAVE_EVERY_N.
  _TEMPERATURE: _TEMPERATURE.
  _TOP_K: _TOP_K.
  _TOP_P: _TOP_P.
  _N_REPEATS: _N_REPEATS.
  get_last_file: get_last_file().
---
# Module: [`simply/eval/decode_eval.py`](../../../../../../raw/code/simply/simply/eval/decode_eval.py)

## Functions
- `_init_fn()` — [`L230`](../../../../../../raw/code/simply/simply/eval/decode_eval.py#L230)
- `_stats_all_history()` — [`L424`](../../../../../../raw/code/simply/simply/eval/decode_eval.py#L424)
- `_stats_history(history_path: epath.PathLike)` — [`L410`](../../../../../../raw/code/simply/simply/eval/decode_eval.py#L410)
- `get_last_file(directory: epath.PathLike, pattern: str)` — [`L156`](../../../../../../raw/code/simply/simply/eval/decode_eval.py#L156) — Returns the last file that matches the pattern.
- `main(argv: Sequence[str])` — [`L170`](../../../../../../raw/code/simply/simply/eval/decode_eval.py#L170)

## Module values
- `_ACTIVATION_DTYPE` — [`L86`](../../../../../../raw/code/simply/simply/eval/decode_eval.py#L86)
- `_BATCH_SIZE` — [`L82`](../../../../../../raw/code/simply/simply/eval/decode_eval.py#L82)
- `_CKPT_DIR` — [`L61`](../../../../../../raw/code/simply/simply/eval/decode_eval.py#L61)
- `_CKPT_FORMAT` — [`L71`](../../../../../../raw/code/simply/simply/eval/decode_eval.py#L71)
- `_CKPT_STEP` — [`L65`](../../../../../../raw/code/simply/simply/eval/decode_eval.py#L65)
- `_DATASOURCE_NAME` — [`L132`](../../../../../../raw/code/simply/simply/eval/decode_eval.py#L132)
- `_EVALUATION` — [`L128`](../../../../../../raw/code/simply/simply/eval/decode_eval.py#L128)
- `_EXPERIMENT_CONFIG` — [`L50`](../../../../../../raw/code/simply/simply/eval/decode_eval.py#L50)
- `_EXPERIMENT_DIR` — [`L46`](../../../../../../raw/code/simply/simply/eval/decode_eval.py#L46)
- `_INTERMEDIATE_DECODE_STEPS` — [`L117`](../../../../../../raw/code/simply/simply/eval/decode_eval.py#L117)
- `_LM_FORMAT` — [`L124`](../../../../../../raw/code/simply/simply/eval/decode_eval.py#L124)
- `_MAX_DECODE_STEPS` — [`L111`](../../../../../../raw/code/simply/simply/eval/decode_eval.py#L111)
- `_MAX_SEQ_LEN` — [`L107`](../../../../../../raw/code/simply/simply/eval/decode_eval.py#L107)
- `_MESH_SHAPE` — [`L57`](../../../../../../raw/code/simply/simply/eval/decode_eval.py#L57)
- `_N_REPEATS` — [`L151`](../../../../../../raw/code/simply/simply/eval/decode_eval.py#L151)
- `_PREFILL_SIZE` — [`L99`](../../../../../../raw/code/simply/simply/eval/decode_eval.py#L99)
- `_SAVE_EVERY_N` — [`L139`](../../../../../../raw/code/simply/simply/eval/decode_eval.py#L139)
- `_SEED` — [`L93`](../../../../../../raw/code/simply/simply/eval/decode_eval.py#L93)
- `_TEMPERATURE` — [`L143`](../../../../../../raw/code/simply/simply/eval/decode_eval.py#L143)
- `_TOP_K` — [`L147`](../../../../../../raw/code/simply/simply/eval/decode_eval.py#L147)
- `_TOP_P` — [`L149`](../../../../../../raw/code/simply/simply/eval/decode_eval.py#L149)
- `_VOCAB_NAME` — [`L75`](../../../../../../raw/code/simply/simply/eval/decode_eval.py#L75)

