from dataclasses import dataclass
from typing import Optional


@dataclass
class KernelTask:
  task_id: str
  description: Optional[str] = None
  input_gen_code: Optional[str] = None
