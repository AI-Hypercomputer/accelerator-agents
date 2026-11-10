---
id: STRAT-MLA-06
title: The SparseCore streaming figure is access-width dependent; it is not a single ceiling for wide-row indirect reads
applies_to: sparsecore
lever: layout
generality: unit-wide
phase: [plan]
verify: probe sc_gather_row_width_scaling
---

# STRAT-MLA-06 — SparseCore achieved bandwidth depends on access width; measure at your row width

**Rule.** A quoted SparseCore streaming bandwidth is a figure for a particular access width, not a
single ceiling. A gather whose rows span many whole DMA granules sustains substantially more than the
figure quoted for narrow reads. Measure achieved bandwidth at your row width before pricing a gather
against any streaming figure — and before concluding from a roofline built on it that you have
headroom or that you have none.

**Mechanism.** Cost is paid per DMA granule (read the granule size from `SparseCoreInfo`), so a read
touching part of a granule wastes the rest. Narrow or partial-granule reads sit at the bad end of the
curve; a wide contiguous row is many whole granules and sits at the good end. The two regimes are the
same hardware fact at different widths, which is why one number cannot serve both. Hypothesis
(the unit's ceiling is an unverified vendor figure; see [tpu_hardware_facts.md](../tpu_hardware_facts.md)
§2): the hardware ceiling sits far above typical achieved figures, so no narrow-read measurement is
close enough to the wall to serve as a roofline.

**Scope.** SparseCore indirect reads. The mechanism — granule utilization sets achieved bandwidth — is
general; any specific number is a property of one width, one geometry and one byte accounting. Settle
the byte accounting (exactly which bytes a row moves) before trusting an absolute figure; a ratio
across widths within one instrument is more transferable than any absolute.

**Does not apply to** narrow or partial-granule reads; there the narrow-read figure is the right one.

**How to check on your kernel.** Run probe `sc_gather_row_width_scaling` for the width curve, and
probe `sc_gather_width_curve_fixed_chunk` to confirm the curve's shape is not an artifact of chunk
size shrinking with row width. Read off achieved bandwidth at your row width, with your own byte
accounting.
