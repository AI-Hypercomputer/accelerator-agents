# pylint: skip-file
"""WikiText loader + fixed-length packer for Llama 3 8B fine-tune."""

from __future__ import annotations
from typing import Iterator, Tuple
import numpy as np
import torch

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
      input_ids = batch[:, :-1].copy()
      labels = batch[:, 1:].copy()
      yield input_ids, labels
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
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
  """Yields (input_ids, labels) torch tensors of shape `(batch_size, seq_len)`.

  Concatenates every text row in the dataset, then packs into fixed-length
  windows. Labels are inputs shifted by one (causal LM).
  """
  from datasets import load_dataset

  ds = load_dataset(
      dataset_name, dataset_config, split=split, streaming=streaming
  )
  it = _iter_tokenize_pack(ds, tokenizer, seq_len)
  for input_ids, labels in _make_batches(it, batch_size, seq_len):
    yield torch.from_numpy(input_ids).to(torch.int64), torch.from_numpy(
        labels
    ).to(torch.int64)


def fake_dataloader(
    size: int, seq_len: int, batch_size: int, vocab_size: int = 128256
):
  """Synthetic random ints (matches canonical example).

  For perf-only smoke tests where loss values aren't meaningful.
  """
  for _ in range(size):
    x = torch.randint(
        0, vocab_size, (batch_size, seq_len), dtype=torch.int64, device="cpu"
    )
    yield x, (x + 1) % vocab_size


__all__ = ["IGNORE_INDEX", "make_dataloader", "fake_dataloader"]
