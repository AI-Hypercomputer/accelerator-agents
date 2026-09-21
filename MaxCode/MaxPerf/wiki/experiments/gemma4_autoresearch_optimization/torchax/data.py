# pylint: skip-file
"""WikiText loader + sequence packer for Gemma 4 fine-tune.

Chosen dataset: `wikitext-2-raw-v1` (small, standard, fast for smoke-testing
on a v6e-8 host). Flip to `wikitext-103-raw-v1` via `--dataset` when the
autoresearch loop wants a realistic dataset. This is a **performance
baseline** trainer — dataset choice is not a semantic knob.

Packs tokens to a fixed `seq_len` (next-token-prediction). Labels are
input_ids shifted by one, with -100 on the EOS slot so the loss ignores the
wraparound.

Not exercised — see the UNTESTED warning in `train.py`.
"""

from __future__ import annotations

from typing import Iterator, Optional, Tuple

import numpy as np


# Token ID for masked positions in labels (standard HF / torch convention).
IGNORE_INDEX = -100


def _iter_tokenize_pack(
    dataset,  # datasets.Dataset
    tokenizer,
    seq_len: int,
    text_field: str = "text",
) -> Iterator[np.ndarray]:
  """Stream rows -> tokens, emit fixed-length windows of `seq_len + 1`.

  The extra +1 is so callers can shift for labels without padding.
  Windows are non-overlapping. Skips empty rows. Uses `append` semantics
  (no separator token between rows — the standard wikitext-raw packer).
  """
  buf: list[int] = []
  window = seq_len + 1
  for row in dataset:
    text = row[text_field]
    if not text or not text.strip():
      continue
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not ids:
      continue
    buf.extend(ids)
    while len(buf) >= window:
      yield np.asarray(buf[:window], dtype=np.int32)
      buf = buf[window:]
  # Drop the final partial window — it is easier to reason about than
  # padding when the dataset can be re-shuffled across epochs.


def _make_batches(
    it: Iterator[np.ndarray],
    batch_size: int,
    seq_len: int,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
  """Group windows into (input_ids, labels) batches of shape (B, T)."""
  window = seq_len + 1
  batch = np.empty((batch_size, window), dtype=np.int32)
  filled = 0
  for w in it:
    batch[filled] = w
    filled += 1
    if filled == batch_size:
      input_ids = batch[:, :-1]
      labels = batch[:, 1:].astype(np.int64)
      # Labels are already the shifted input_ids; no need to mask for
      # wikitext packing since there are no pad tokens here.
      yield input_ids.copy(), labels.copy()
      filled = 0


def make_dataloader(
    seq_len: int,
    batch_size: int,
    tokenizer,
    split: str = "train",
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    streaming: bool = False,
    num_proc: Optional[int] = None,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
  """Return a generator of `(input_ids, labels)` numpy int32/int64 batches.

  Shapes: `(batch_size, seq_len)`.

  Parameters
  ----------
  seq_len : per-example sequence length (after packing).
  batch_size : global batch size (caller is responsible for dividing by DP
      when sharding — input-sharding over the `dp` mesh axis takes care of
      the per-device view).
  tokenizer : a HF tokenizer with `__call__` returning `input_ids`.
  split : dataset split name.
  dataset_name / dataset_config : forwarded to `datasets.load_dataset`.
  streaming : use HF streaming (avoids downloading the full wikitext to
      disk; useful for wikitext-103).
  num_proc : multiprocessing for the pre-tokenization pass. Unused on
      streaming.
  """
  try:
    from datasets import load_dataset
  except ImportError as e:  # pragma: no cover
    raise ImportError(
        "`datasets` is not installed. `pip install datasets`."
    ) from e

  ds = load_dataset(
      dataset_name, dataset_config, split=split, streaming=streaming
  )
  # We do row-level tokenization on the fly; HF `map` isn't worth the
  # overhead for a 4M-token dataset.
  it = _iter_tokenize_pack(ds, tokenizer, seq_len)
  yield from _make_batches(it, batch_size, seq_len)


__all__ = ["IGNORE_INDEX", "make_dataloader"]
