---
title: 'Module: easydel/trainers/packer.py'
type: catalog
provenance: extracted
module: easydel/trainers/packer.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.packer`/pack_sequences().
symbols:
  pack_sequences: ''
  pack_sequences.pack_examples: pack_examples().
  pack_sequences.pack_examples.get_position_ids: pack_examples().get_position_ids().
---
# Module: [`easydel/trainers/packer.py`](../../../../../../raw/code/EasyDeL/easydel/trainers/packer.py)

## Functions
- `get_position_ids(length, start_position=0)` — [`L125`](../../../../../../raw/code/EasyDeL/easydel/trainers/packer.py#L125) — Generate position IDs for a sequence.
- `pack_examples(examples)` — [`L107`](../../../../../../raw/code/EasyDeL/easydel/trainers/packer.py#L107) — Pack a batch of examples into fixed-length sequences.
- `pack_sequences(dataset: Dataset, max_length: int = 512, pad_token_id: int = 0, reset_position_ids: bool = False, num_proc: int | None = None)` — [`L34`](../../../../../../raw/code/EasyDeL/easydel/trainers/packer.py#L34) — Pack multiple sequences into fixed-length batches for efficient training.

