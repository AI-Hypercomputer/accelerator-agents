from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EvaluationResult:
  task_id: str
  compiled_successfully: List[bool] = field(default_factory=list)
  numerically_correct: List[bool] = field(default_factory=list)
  max_abs_diff: List[Optional[float]] = field(default_factory=list)
  max_rel_diff: List[Optional[float]] = field(default_factory=list)
  reference_time_ms: List[float] = field(default_factory=list)
  optimized_time_ms: List[float] = field(default_factory=list)
  xprof_reference_time_ms: List[float] = field(default_factory=list)
  xprof_optimized_time_ms: List[float] = field(default_factory=list)
  error_trace: List[Optional[str]] = field(default_factory=list)
  logs: List[str] = field(default_factory=list)

  @property
  def speedup(self) -> List[Optional[float]]:
    res = []
    if not self.optimized_time_ms or not self.reference_time_ms:
      return []
    for opt, ref in zip(self.optimized_time_ms, self.reference_time_ms):
      if opt == 0 or ref == 0:
        res.append(None)
      else:
        res.append(ref / opt)
    return res

  @property
  def speed_up_xprof(self) -> List[Optional[float]]:
    res = []
    if not self.xprof_optimized_time_ms or not self.xprof_reference_time_ms:
      return []
    for opt, ref in zip(
      self.xprof_optimized_time_ms, self.xprof_reference_time_ms
    ):
      if opt == 0 or ref == 0:
        res.append(None)
      else:
        res.append(ref / opt)
    return res
