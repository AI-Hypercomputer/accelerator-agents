from dataclasses import dataclass
from typing import Dict, List, Optional, Union


@dataclass
class KernelTask:
  task_id: str
  description: Optional[str] = None
  input_gen_code: Optional[str] = None
  atol: Optional[Union[float, List[float]]] = None
  rtol: Optional[Union[float, List[float]]] = None
  # When true, the harness sorts every output leaf along its last axis before
  # comparing reference and optimized outputs. Use this for outputs whose
  # order along the last axis is not part of the contract (e.g. top-k index
  # sets), so an otherwise correct kernel is not failed for emitting the same
  # elements in a different order. Defaults to element-wise comparison.
  sort_before_compare: bool = False
  # Buffers the kernel updates in place, as {input_index: output_index}:
  # the index of a dynamic argument of `computation()` and the index of the
  # flattened output leaf that returns the updated buffer with the same shape
  # and dtype (a KV cache, a recurrent state, ...). The harness donates each
  # listed input to the compiled call so XLA can alias it with that output
  # instead of copying it, as production callers do, and feeds the output
  # back in as the next iteration's input. The same map is applied to the
  # reference and the optimized kernel. Inputs that production code donates
  # only to free memory early (queries, new keys/values) do not belong here:
  # they alias no output, so donating them changes nothing in a benchmark.
  donate_args: Optional[Dict[int, int]] = None
