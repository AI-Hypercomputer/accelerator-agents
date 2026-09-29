---
title: 'Module: src/alphafold3/model/scoring/alignment.py'
type: catalog
provenance: extracted
module: src/alphafold3/model/scoring/alignment.py
status: fresh
symbol_base: scip-python python alphafold3 0.0.0 `src.alphafold3.model.scoring.alignment`/
symbols:
  align: align().
  deviations_from_coords: deviations_from_coords().
  rmsd_from_coords: rmsd_from_coords().
  transform_ls: transform_ls().
---
# Module: [`src/alphafold3/model/scoring/alignment.py`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/scoring/alignment.py)

## Functions
- `align(*, x: np.ndarray, y: np.ndarray, x_indices: np.ndarray, y_indices: np.ndarray)` — [`L61`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/scoring/alignment.py#L61) — Align x to y considering only included_idxs.
- `deviations_from_coords(decoy_coords: np.ndarray, gt_coords: np.ndarray, align_idxs: np.ndarray | None = None, include_idxs: np.ndarray | None = None)` — [`L105`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/scoring/alignment.py#L105) — Returns the raw per-atom deviations used in RMSD computation.
- `rmsd_from_coords(decoy_coords: np.ndarray, gt_coords: np.ndarray, align_idxs: np.ndarray | None = None, include_idxs: np.ndarray | None = None)` — [`L134`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/scoring/alignment.py#L134) — Computes the *aligned* RMSD of two Mx3 np arrays of coordinates.
- `transform_ls(x: np.ndarray, b: np.ndarray, *, allow_reflection: bool = False)` — [`L25`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/scoring/alignment.py#L25) — Find the least squares best fit rotation between two sets of N points.

