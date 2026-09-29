---
title: 'Module: src/alphafold3/model/confidences.py'
type: catalog
provenance: extracted
module: src/alphafold3/model/confidences.py
status: fresh
symbol_base: scip-python python alphafold3 0.0.0 `src.alphafold3.model.confidences`/
symbols:
  fraction_disordered: fraction_disordered().
  has_clash: has_clash().
  get_ranking_score: get_ranking_score().
  pae_metrics: pae_metrics().
  chain_pair_pae: chain_pair_pae().
  pae_metrics.reduce_chain_pair_fn: pae_metrics().reduce_chain_pair_fn().
  pae_metrics.reduce_chain_pair_fn.inner: pae_metrics().reduce_chain_pair_fn().inner().
  chain_pairwise_predicted_tm_scores: chain_pairwise_predicted_tm_scores().
  weighted_mean: weighted_mean().
  windowed_solvent_accessible_area: windowed_solvent_accessible_area().
  pde_single: pde_single().
  reduce_chain_pair: reduce_chain_pair().
  get_iptm_xchain: get_iptm_xchain().
  weighted_nanmean: weighted_nanmean().
  _IPTM_WEIGHT: _IPTM_WEIGHT.
  predicted_tm_score: predicted_tm_score().
  MAX_ACCESSIBLE_SURFACE_AREA: MAX_ACCESSIBLE_SURFACE_AREA.
  _FRACTION_DISORDERED_WEIGHT: _FRACTION_DISORDERED_WEIGHT.
  _CLASH_PENALIZATION_WEIGHT: _CLASH_PENALIZATION_WEIGHT.
  rank_metric: rank_metric().
  chain_pair_pde: chain_pair_pde().
---
# Module: [`src/alphafold3/model/confidences.py`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidences.py)

## Functions
- `chain_pair_pae(*, num_tokens: int, asym_ids: np.ndarray, full_pae: np.ndarray, mask: np.ndarray | None = None, contact_probs: np.ndarray | None = None)` — [`L355`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidences.py#L355) — Compute predicted errors for all pairs of chains.
- `chain_pair_pde(num_tokens: int, asym_ids: np.ndarray, full_pde: np.ndarray)` — [`L299`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidences.py#L299) — Compute predicted distance errors for all pairs of chains.
- `chain_pairwise_predicted_tm_scores(tm_adjusted_pae: np.ndarray, pair_mask: np.ndarray, asym_id: np.ndarray)` — [`L634`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidences.py#L634) — Compute predicted TM (pTM) between each pair of chains independently.
- `fraction_disordered(struc: structure.Structure, rasa_disorder_cutoff: float = 0.581)` — [`L99`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidences.py#L99) — Compute fraction of protein residues that are disordered.
- `get_iptm_xchain(chain_pair_iptm: np.ndarray)` — [`L563`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidences.py#L563) — Cross chain aggregate ipTM.
- `get_ranking_score(ptm: float, iptm: float, fraction_disordered_: float, has_clash_: bool)` — [`L194`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidences.py#L194)
- `has_clash(struc: structure.Structure, cutoff_radius: float = 1.1, min_clashes_for_overlap: int = 100, min_fraction_for_overlap: float = 0.5)` — [`L140`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidences.py#L140) — Determine whether the structure has at least one clashing chain.
- `inner(agg_over_col)` — [`L529`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidences.py#L529)
- `pae_metrics(num_tokens: int, asym_ids: np.ndarray, full_pae: np.ndarray, mask: np.ndarray, contact_probs: np.ndarray, tm_adjusted_pae: np.ndarray)` — [`L485`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidences.py#L485) — PAE aggregate metrics.
- `pde_single(num_tokens: int, asym_ids: np.ndarray, full_pde: np.ndarray, contact_probs: np.ndarray)` — [`L243`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidences.py#L243) — Compute 1D PDE summaries.
- `predicted_tm_score(tm_adjusted_pae: np.ndarray, pair_mask: np.ndarray, asym_id: np.ndarray, interface: bool = False)` — [`L579`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidences.py#L579) — Computes predicted TM alignment or predicted interface TM alignment score.
- `rank_metric(full_pde: jnp.ndarray | np.ndarray, contact_probs: jnp.ndarray | np.ndarray)` — [`L209`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidences.py#L209) — Compute the metric that will be used to rank predictions, higher is better.
- `reduce_chain_pair(*, chain_pair_met: np.ndarray, num_chain_tokens: np.ndarray, agg_over_col: bool, agg_type: str, weight_method: str)` — [`L422`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidences.py#L422) — Compute 1D summaries from a chain-pair summary.
- `reduce_chain_pair_fn(chain_pair: np.ndarray)` — [`L528`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidences.py#L528)
- `weighted_mean(mask, value, axis)` — [`L239`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidences.py#L239)
- `weighted_nanmean(value: np.ndarray, mask: np.ndarray, axis: int)` — [`L333`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidences.py#L333) — Nan-mean with weighting -- empty slices return NaN.
- `windowed_solvent_accessible_area(cif: str, window: int = 25)` — [`L62`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidences.py#L62) — Implementation of AlphaFold-RSA.

## Module values
- `MAX_ACCESSIBLE_SURFACE_AREA` — [`L33`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidences.py#L33)
- `_CLASH_PENALIZATION_WEIGHT` — [`L59`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidences.py#L59)
- `_FRACTION_DISORDERED_WEIGHT` — [`L58`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidences.py#L58)
- `_IPTM_WEIGHT` — [`L57`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidences.py#L57)

