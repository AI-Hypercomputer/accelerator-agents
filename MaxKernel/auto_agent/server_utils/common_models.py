import logging
from typing import Optional
from pydantic import BaseModel

class CodeRequest(BaseModel):
  code: str
  timeout: Optional[int] = 30
  dependencies: Optional[dict] = None


class CodeResponse(BaseModel):
  output: str
  error: Optional[str] = None
  exit_code: int


class AutotuneRequest(BaseModel):
  code_template: str
  search_space: dict[str, list]
  timeout: Optional[int] = 300
  total_timeout: Optional[int] = None
  dependencies: Optional[dict] = None


class GetTpuVersionResponse(BaseModel):
  tpu_version: str


class GetBackendVersionResponse(BaseModel):
  backend_version: str


def extract_code(code: str) -> str:
  """Extracts code from markdown blocks if present."""
  code_content = code.strip()
  if code_content.startswith("```python") and code_content.endswith("```"):
    lines = code_content.split("\n")
    if lines[0].strip() == "```python":
      lines = lines[1:]
    if lines[-1].strip() == "```":
      lines = lines[:-1]
    return "\n".join(lines)
  elif code_content.startswith("```") and code_content.endswith("```"):
    lines = code_content.split("\n")
    if lines[0].strip().startswith("```"):
      lines = lines[1:]
    if lines[-1].strip() == "```":
      lines = lines[:-1]
    return "\n".join(lines)
  return code_content
