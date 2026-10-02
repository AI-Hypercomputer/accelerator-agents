---
title: 'Module: ejkernel/ops/utils/meta.py'
type: catalog
provenance: extracted
module: ejkernel/ops/utils/meta.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.ops.utils.meta`/
symbols:
  labels_to_configs: labels_to_configs().
  find_labels_in_lowered: find_labels_in_lowered().
  extract_labels_from_hlo_text: extract_labels_from_hlo_text().
  LABEL_RE: LABEL_RE.
  label: label().
  LABEL_PREFIXES: LABEL_PREFIXES.
---
# Module: [`ejkernel/ops/utils/meta.py`](../../../../../../../raw/code/ejkernel/ejkernel/ops/utils/meta.py)

## Functions
- `extract_labels_from_hlo_text(hlo_text: str)` — [`L126`](../../../../../../../raw/code/ejkernel/ejkernel/ops/utils/meta.py#L126) — Find all ejkernel operation labels in HLO text.
- `find_labels_in_lowered(lowered)` — [`L146`](../../../../../../../raw/code/ejkernel/ejkernel/ops/utils/meta.py#L146) — Extract operation labels from a JAX lowered computation.
- `label(op_id: str, call_hash: str, prefix: str | None = None)` — [`L96`](../../../../../../../raw/code/ejkernel/ejkernel/ops/utils/meta.py#L96) — Generate a standardised label string for an operation instance.
- `labels_to_configs(lowered, selector)` — [`L63`](../../../../../../../raw/code/ejkernel/ejkernel/ops/utils/meta.py#L63) — Extract labels from lowered computation and map them to configurations.

## Module values
- `LABEL_PREFIXES` — [`L59`](../../../../../../../raw/code/ejkernel/ejkernel/ops/utils/meta.py#L59)
- `LABEL_RE` — [`L60`](../../../../../../../raw/code/ejkernel/ejkernel/ops/utils/meta.py#L60)

