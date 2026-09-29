---
title: 'Module: src/alphafold3/model/msa_pairing.py'
type: catalog
provenance: extracted
module: src/alphafold3/model/msa_pairing.py
status: fresh
symbol_base: scip-python python alphafold3 0.0.0 `src.alphafold3.model.msa_pairing`/
symbols:
  remove_all_gapped_rows_from_all_seqs: remove_all_gapped_rows_from_all_seqs().
  create_paired_features: create_paired_features().
  deduplicate_unpaired_sequences: deduplicate_unpaired_sequences().
  _align_species: _align_species().
  choose_paired_unpaired_msa_crop_sizes: choose_paired_unpaired_msa_crop_sizes().
---
# Module: [`src/alphafold3/model/msa_pairing.py`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/msa_pairing.py)

## Functions
- `_align_species(all_species: Sequence[bytes], chains_species_to_rows: Sequence[Mapping[bytes, np.ndarray]], min_hits_per_species: Mapping[bytes, int])` — [`L40`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/msa_pairing.py#L40) — Aligns MSA row indices based on species.
- `choose_paired_unpaired_msa_crop_sizes(unpaired_msa: np.ndarray, paired_msa: np.ndarray | None, total_msa_crop_size: int, max_paired_sequences: int)` — [`L247`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/msa_pairing.py#L247) — Returns the sizes of the MSA crop and MSA_all_seq crop.
- `create_paired_features(chains: Sequence[MutableMapping[str, np.ndarray]], max_paired_sequences: int, nonempty_chain_ids: set[str], max_hits_per_species: int)` — [`L87`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/msa_pairing.py#L87) — Creates per-chain MSA features where the MSAs have been aligned.
- `deduplicate_unpaired_sequences(np_chains: Sequence[MutableMapping[str, np.ndarray]])` — [`L219`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/msa_pairing.py#L219) — Deduplicates unpaired sequences based on paired sequences.
- `remove_all_gapped_rows_from_all_seqs(chains_list: Sequence[dict[str, np.ndarray]], asym_ids: Sequence[float])` — [`L292`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/msa_pairing.py#L292) — Removes all gapped rows from all_seq feat based on selected asym_ids.

