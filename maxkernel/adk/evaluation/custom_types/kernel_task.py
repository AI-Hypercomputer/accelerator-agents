from dataclasses import dataclass
from typing import List, Optional, Union


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
