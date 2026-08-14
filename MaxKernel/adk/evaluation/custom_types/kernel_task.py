from dataclasses import dataclass
from typing import List, Optional, Union


@dataclass
class KernelTask:
  task_id: str
  description: Optional[str] = None
  input_gen_code: Optional[str] = None
  atol: Optional[Union[float, List[float]]] = None
  rtol: Optional[Union[float, List[float]]] = None
