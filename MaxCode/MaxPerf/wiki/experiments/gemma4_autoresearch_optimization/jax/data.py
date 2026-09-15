# pylint: skip-file
"""WikiText loader + packer for the native-JAX Gemma 4 trainer.

Mirrors `../torchax/data.py` so the two trainers see identical batches.
"""

from __future__ import annotations

from typing import Iterator, Optional, Tuple

import numpy as np

IGNORE_INDEX = -100


def _iter_tokenize_pack(
    dataset,
    tokenizer,
    seq_len: int,
    text_field: str = "text",
) -> Iterator[np.ndarray]:
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


def _make_batches(
    it: Iterator[np.ndarray],
    batch_size: int,
    seq_len: int,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
  window = seq_len + 1
  batch = np.empty((batch_size, window), dtype=np.int32)
  filled = 0
  for w in it:
    batch[filled] = w
    filled += 1
    if filled == batch_size:
      input_ids = batch[:, :-1]
      labels = batch[:, 1:].astype(np.int64)
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
  try:
    from datasets import load_dataset
  except ImportError as e:
    raise ImportError(
        "`datasets` is not installed. `pip install datasets`."
    ) from e
  ds = load_dataset(
      dataset_name, dataset_config, split=split, streaming=streaming
  )
  it = _iter_tokenize_pack(ds, tokenizer, seq_len)
  yield from _make_batches(it, batch_size, seq_len)


__all__ = ["IGNORE_INDEX", "make_dataloader"]
