---
title: 'Module: pallas_forge/tune/__init__.py'
type: catalog
provenance: extracted
module: pallas_forge/tune/__init__.py
status: fresh
symbol_base: scip-python python pallas-forge 0.0.0 `pallas_forge.tune`/
symbols:
  tune: tune().
  __all__: __all__.
---
# Module: [`pallas_forge/tune/__init__.py`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/__init__.py)

## Functions
- `tune(kernel_fn: Callable, input_fn: Callable[[dict[str, Any]], tuple], config: TuneConfig | dict | str | Path, *, strategy: str | SearchStrategy = "grid", n_warmup: int = 5, n_repeat: int = 20, flops_fn: Callable[[dict[str, Any]], int] | None = None, bytes_fn: Callable[[dict[str, Any]], int] | None = None, top_n_traces: int = 0, trace_output_dir: str = "/tmp/xprof_traces", verbose: bool = True)` — [`L34`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/__init__.py#L34) — Auto-tune a Pallas kernel over a configuration space. — documented in [pallas_forge-tune-config](../../../concepts/pallas_forge-tune-config.md)

## Module values
- `__all__` — [`L130`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/__init__.py#L130)

