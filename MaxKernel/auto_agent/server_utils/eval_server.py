import asyncio
import logging
from enum import Enum
from typing import Optional

import aiohttp
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from auto_agent.constants import (
  EVAL_SERVER_PORT,
  TPU_TIMEOUT,
)
from auto_agent.server_utils.tpu_server import (
  CodeResponse,
  get_tpu_version,
)

logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
  datefmt="%Y-%m-%d %H:%M:%S",
)

app = FastAPI(title="Agent Evaluation Server", version="1.0.0")

backend_semaphore = asyncio.Semaphore(1)


class Backend:
  def __init__(self, name: str, ip: str, port: int, backend_type: str = "tpu"):
    self.name = name
    self.ip = ip
    self.port = port
    self.backend_type = backend_type  # "tpu" or "cpu"
    self.status = "available"

    # Get version info for display purposes
    if backend_type == "tpu":
      try:
        # Try to get the running loop (if we're already in an async context)
        loop = asyncio.get_running_loop()
        # We're in an async context, but __init__ is sync, so this shouldn't happen
        self.version = "TPU"
      except RuntimeError:
        # No running loop - we're in a sync context, which is expected
        # Use asyncio.run() for Python 3.10+ compatibility
        try:
          self.version = asyncio.run(get_tpu_version())
        except RuntimeError:
          # asyncio.run() failed (maybe loop already exists), fall back to creating one
          loop = asyncio.new_event_loop()
          asyncio.set_event_loop(loop)
          self.version = loop.run_until_complete(get_tpu_version())
          loop.close()
    else:
      self.version = "CPU"

  def __str__(self):
    return f"{self.name}: {self.version} ({self.ip}:{self.port})"

  def __repr__(self):
    return self.__str__()

  def set_status(self, status: str):
    if status not in ["available", "busy"]:
      raise ValueError("Status must be 'available' or 'busy'")
    self.status = status

  def get_status(self):
    return self.status


class EvalTypes(Enum):
  CORRECTNESS_TEST = "correctness_test"
  COMPILATION_TEST = "compilation_test"
  PERFORMANCE_TEST = "performance_test"
  UNIFIED_TEST = "unified_test"
  PROFILE = "profile"


class EvalRequest(BaseModel):
  eval_type: EvalTypes
  code: str
  timeout: Optional[int] = 30
  backend_type: Optional[str] = None  # "tpu", "cpu", or None for any available
  dependencies: Optional[dict] = None


class Evaluator:
  def __init__(self, cfg_path="eval_config.yaml"):
    with open(cfg_path, "r") as file:
      self.config = yaml.safe_load(file)

    self.backends = []

    if "backends" in self.config and self.config["backends"]:
      logging.info("Using 'backends' configuration format")
      for backend_config in self.config["backends"]:
        backend_obj = Backend(
          name=backend_config["name"],
          ip=backend_config["ip"],
          port=backend_config["port"],
          backend_type=backend_config.get("type", "tpu"),
        )
        self.backends.append(backend_obj)
    else:
      raise ValueError(
        "No backends configured in eval_config.yaml. Please use the 'backends' format."
      )

    logging.info(f"Evaluator initialized with backends: {self.backends}")

  def get_available_backend(self, backend_type: Optional[str] = None):
    """
    Get an available backend without blocking, optionally filtered by type.
    """
    for backend in self.backends:
      if backend.get_status() == "available":
        if backend_type is None or backend.backend_type == backend_type:
          return backend
    return None


evaluator = Evaluator()


@app.get("/health")
async def health_check():
  return {"status": "healthy"}


@app.post("/evaluate", response_model=CodeResponse)
async def evaluate(request: EvalRequest):
  if request.eval_type not in EvalTypes:
    raise HTTPException(status_code=400, detail="Invalid evaluation type")

  # Acquire backend, retry if busy
  while True:
    async with backend_semaphore:
      backend = evaluator.get_available_backend(
        backend_type=request.backend_type
      )
      if backend:
        backend.set_status("busy")
        backend_ip = backend.ip
        backend_port = backend.port
        backend_name = backend.name
        backend_type = backend.backend_type
        break
    await asyncio.sleep(0.1)

  try:
    # Start evaluation process
    requested_type_msg = (
      f" (requested: {request.backend_type})" if request.backend_type else ""
    )
    logging.info(
      f"Starting evaluation on {backend_type} backend '{backend_name}' ({backend_ip}:{backend_port}) for {request.eval_type.value}{requested_type_msg}"
    )

    # Send request to backend server
    async with aiohttp.ClientSession() as session:
      async with session.post(
        f"http://{backend_ip}:{backend_port}/{request.eval_type.value}",
        json={
          "eval_type": request.eval_type.value,
          "code": request.code,
          "timeout": request.timeout,
          "dependencies": request.dependencies,
        },
      ) as response:
        result = await response.json()
        logging.info(
          f"Received response from {backend_name}: {response.status}"
        )

        if response.status != 200:
          if response.status == 408:
            raise HTTPException(
              status_code=408,
              detail=f"Backend evaluation timed out. Timeout was set to {TPU_TIMEOUT} seconds.",
            )
          raise HTTPException(
            status_code=response.status,
            detail=result.get("detail", "Backend evaluation failed"),
          )

    # Mark backend as available after evaluation
    logging.info(
      f"Evaluation completed on {backend_name}, marking as available"
    )
    backend.set_status("available")

    return result

  except Exception as e:
    logging.error(f"Error occurred while evaluating on {backend_name}: {e}")
    backend.set_status("available")
    raise HTTPException(status_code=500, detail="Backend evaluation failed")


if __name__ == "__main__":
  import uvicorn

  uvicorn.run(app, host="0.0.0.0", port=EVAL_SERVER_PORT)
