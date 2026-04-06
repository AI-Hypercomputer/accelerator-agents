from dataclasses import dataclass
from typing import Optional


@dataclass
class EvaluationResult:
  task_id: str
  compiled_successfully: bool = False
  numerically_correct: bool = False
  max_abs_diff: Optional[float] = None
  max_rel_diff: Optional[float] = None
  reference_time_ms: float = 0.0
  optimized_time_ms: float = 0.0
  error_trace: Optional[str] = None

  @property
  def speedup(self) -> Optional[float]:
    if self.optimized_time_ms == 0 or self.reference_time_ms == 0:
      return None
    return self.reference_time_ms / self.optimized_time_ms
