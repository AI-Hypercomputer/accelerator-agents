---
title: "aqt (google/aqt)"
type: codebase
tags: [aqt, quantization, pallas, mosaic-tpu, deprecated, qwix-superseded]
repo: google/aqt
commit: 9d1667e
status: deprecated
created: 2026-04-23
updated: 2026-04-23
---
<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->


Accurate Quantized Training — Google's earlier quantization framework, now **deprecated in favor of [qwix](qwix.md)**. Four Pallas files in the repo, all superseded. Ingested for lineage only; any new quantization hypothesis should target qwix, not aqt. Per-kernel detail in [§5 of the Pallas kernel directory](../analyses/pallas-kernel-directory/05-frameworks-quant.md).

## Architecture

- `aqt/jax/v2/pallas/pallas_call.py` — `pallas_call` + `dot_general` wrapper for AQT tensors. **Deprecated.**
- Rest of `aqt/` — QAT training machinery (out of scope).

## Key abstractions

| Component | File | Status |
|---|---|---|
| AQT-aware `pallas_call` | `aqt/jax/v2/pallas/pallas_call.py` | deprecated — use qwix |

## Performance-relevant surfaces

### Status: deprecated
Do not target AQT for new work. The same functionality plus improvements lives in [qwix](qwix.md). Read AQT only for lineage understanding when reading older maxtext / maxdiffusion code that still references it.

## Connections

- [Pallas kernel directory §5](../analyses/pallas-kernel-directory/05-frameworks-quant.md)
- [qwix](qwix.md) — successor.

## Sources

- `raw/code/aqt/aqt/jax/v2/pallas/pallas_call.py`
- Upstream: <https://github.com/google/aqt>
