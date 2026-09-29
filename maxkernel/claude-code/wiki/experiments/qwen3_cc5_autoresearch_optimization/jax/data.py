"""WikiText loader + packer + synthetic loader for the native-JAX Qwen3 trainer.

Mirrors `../torchax/data.py` so the two lanes see identical batches. Pure numpy
(no torch) — yields (input_ids, labels) as int32/int64 numpy arrays.
"""

from __future__ import annotations

from typing import Iterator, Optional, Tuple

import numpy as np

IGNORE_INDEX = -100


def _iter_tokenize_pack(
    dataset, tokenizer, seq_len: int, text_field: str = "text"
):
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


def _make_batches(it, batch_size: int, seq_len: int):
  window = seq_len + 1
  batch = np.empty((batch_size, window), dtype=np.int32)
  filled = 0
  for w in it:
    batch[filled] = w
    filled += 1
    if filled == batch_size:
      yield batch[:, :-1].copy(), batch[:, 1:].astype(np.int64).copy()
      filled = 0


def make_dataloader(
    seq_len: int,
    batch_size: int,
    tokenizer,
    *,
    split: str = "train",
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    streaming: bool = False,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
  from datasets import load_dataset

  ds = load_dataset(
      dataset_name, dataset_config, split=split, streaming=streaming
  )
  yield from _make_batches(
      _iter_tokenize_pack(ds, tokenizer, seq_len), batch_size, seq_len
  )


def fake_dataloader(
    size: int, seq_len: int, batch_size: int, vocab_size: int = 151936
):
  """Synthetic random ints (matches the torchax sibling).

  Default vocab = Qwen3.
  """
  rng = np.random.default_rng(0)
  for _ in range(size):
    x = rng.integers(0, vocab_size, size=(batch_size, seq_len), dtype=np.int32)
    y = ((x.astype(np.int64) + 1) % vocab_size).astype(np.int64)
    yield x.copy(), y.copy()


__all__ = ["IGNORE_INDEX", "make_dataloader", "fake_dataloader"]
