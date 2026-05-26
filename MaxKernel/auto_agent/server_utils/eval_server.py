import asyncio
import atexit
import logging
import subprocess
import time
import uuid
from contextlib import asynccontextmanager
from enum import Enum
from typing import Optional

import aiohttp
import requests
import yaml
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

from auto_agent.constants import EVAL_SERVER_PORT

logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
  datefmt="%Y-%m-%d %H:%M:%S",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
  yield
  logging.info("Shutting down eval_server, cleaning up tunnels...")
  evaluator._cleanup_tunnels()


app = FastAPI(
  title="Agent Evaluation Server", version="1.0.0", lifespan=lifespan
)

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
      url = f"http://{self.ip}:{self.port}/get_tpu_version"
      try:
        resp = requests.post(url, timeout=10)
        if resp.status_code == 200:
          data = resp.json()
          if isinstance(data, dict):
            self.version = data.get("tpu_version", "TPU version not found")
          else:
            self.version = str(data)
        else:
          self.version = f"HTTP Error {resp.status_code}"
      except Exception as e:
        logging.warning(f"Failed to get TPU version from {url}: {e}")
        self.version = "Unknown TPU"
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
  AUTOTUNE = "autotune"


class EvalRequest(BaseModel):
  eval_type: EvalTypes
  timeout: int
  code: Optional[str] = None
  code_template: Optional[str] = None  # For autotune
  search_space: Optional[dict] = None  # For autotune
  backend_type: Optional[str] = None  # "tpu", "cpu", or None for any available
  dependencies: Optional[dict] = None
  total_timeout: Optional[int] = None  # For autotune
  client_wait_timeout: Optional[int] = None


class TaskStatus(str, Enum):
  QUEUED = "queued"
  RUNNING = "running"
  SUCCESS = "success"
  FAILED = "failed"
  TIMEOUT = "timeout"


class TaskResponse(BaseModel):
  task_id: str
  status: TaskStatus
  result: Optional[dict] = None
  error: Optional[str] = None


# For evaluation task tracking. task_id -> {status, request, result_or_error}
_tasks = {}


class Evaluator:
  def __init__(self, cfg_path="eval_config.yaml"):
    with open(cfg_path, "r") as file:
      self.config = yaml.safe_load(file)

    self.backends = []

    self.tunnels = []
    self._load_backends()

    logging.info(f"Evaluator initialized with backends: {self.backends}")

  def _load_backends(self):
    if "backends" in self.config and self.config["backends"]:
      logging.info("Using 'backends' configuration format")
      for backend_config in self.config["backends"]:
        # Check if gcloud TPU VM tunnel is requested
        self._create_tunnel(backend_config)

        backend_obj = Backend(
          name=backend_config["name"],
          ip=backend_config["ip"],
          port=backend_config["port"],
          backend_type=backend_config.get("type", "tpu"),
        )
        self.backends.append(backend_obj)

      # Register cleanup
      if self.tunnels:
        logging.info(f"Registering cleanup for {len(self.tunnels)} tunnels.")
        atexit.register(self._cleanup_tunnels)

    else:
      raise ValueError(
        "No backends configured in eval_config.yaml. Please use the 'backends' format."
      )

  def _create_tunnel(self, backend_config):
    if "tpu_vm" in backend_config:
      local_port = backend_config["port"]
      gt = backend_config["tpu_vm"]
      tpu_name = gt["tpu_name"]
      zone = gt["zone"]
      project = gt["project"]
      remote_port = gt["port"]

      logging.info(f"Constructing gcloud SSH tunnel to {tpu_name}...")

      cmd = [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "ssh",
        tpu_name,
        f"--zone={zone}",
        f"--project={project}",
        "--",
        "-N",
        "-L",
        f"{local_port}:localhost:{remote_port}",
      ]

      try:
        process = subprocess.Popen(cmd)
        self.tunnels.append(process)
        logging.info(
          f"Gcloud SSH tunnel started for {backend_config['name']} (PID: {process.pid})"
        )
        time.sleep(10)  # Blocking sleep as requested
      except Exception as e:
        logging.error(f"Failed to start gcloud SSH tunnel: {e}")

  def _cleanup_tunnels(self):
    if self.tunnels:
      logging.info("Cleaning up SSH tunnels...")
      for p in self.tunnels:
        try:
          p.terminate()
          p.wait(timeout=5)
          logging.info(f"Tunnel PID {p.pid} terminated.")
        except Exception as e:
          logging.error(f"Failed to terminate tunnel PID {p.pid}: {e}")
          try:
            p.kill()
          except:
            pass

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


# Asynchronous task polling architecture for evaluation.
@app.post("/evaluate", status_code=202)
async def evaluate(request: EvalRequest, background_tasks: BackgroundTasks):
  task_id = str(uuid.uuid4())
  _tasks[task_id] = {"status": TaskStatus.QUEUED, "request": request}
  background_tasks.add_task(run_evaluation_task, task_id, request)
  return {"task_id": task_id, "status": TaskStatus.QUEUED}


@app.get("/status/{task_id}", response_model=TaskResponse)
async def get_task_status(task_id: str):
  if task_id not in _tasks:
    raise HTTPException(status_code=404, detail="Task not found")
  task_info = _tasks[task_id]
  return {
    "task_id": task_id,
    "status": task_info["status"],
    "result": task_info.get("result"),
    "error": task_info.get("error"),
  }


async def run_evaluation_task(task_id: str, request: EvalRequest):
  _tasks[task_id]["status"] = TaskStatus.RUNNING
  try:
    result = await _perform_evaluation(request)
    _tasks[task_id]["status"] = TaskStatus.SUCCESS
    _tasks[task_id]["result"] = result
  except HTTPException as e:
    if e.status_code == 408:
      _tasks[task_id]["status"] = TaskStatus.TIMEOUT
    else:
      _tasks[task_id]["status"] = TaskStatus.FAILED
    _tasks[task_id]["error"] = e.detail
  except Exception as e:
    _tasks[task_id]["status"] = TaskStatus.FAILED
    _tasks[task_id]["error"] = str(e)


async def _perform_evaluation(request: EvalRequest):
  if request.eval_type not in EvalTypes:
    raise HTTPException(status_code=400, detail="Invalid evaluation type")

  # Acquire backend, retry if busy
  start_queue_time = time.time()
  max_queue_time = (
    request.client_wait_timeout
    if request.client_wait_timeout is not None
    else 3600 * 3
  )

  while True:
    if time.time() - start_queue_time > max_queue_time:
      raise HTTPException(
        status_code=408,
        detail="Timed out waiting for an available backend in queue",
      )

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
      f"Starting evaluation on {backend_type} backend '{backend_name}' "
      f"({backend_ip}:{backend_port}) for {request.eval_type.value}"
      f"{requested_type_msg}"
    )

    # Construct payload based on eval type
    payload = {
      "eval_type": request.eval_type.value,
      "timeout": request.timeout,
    }
    if request.eval_type == EvalTypes.AUTOTUNE:
      payload["code_template"] = request.code_template
      payload["search_space"] = request.search_space
      backend_timeout = request.total_timeout
      payload["total_timeout"] = request.total_timeout
      payload["dependencies"] = request.dependencies
    else:
      payload["code"] = request.code
      payload["dependencies"] = request.dependencies
      backend_timeout = request.timeout

    client_timeout = aiohttp.ClientTimeout(total=backend_timeout + 10)
    async with aiohttp.ClientSession(timeout=client_timeout) as session:
      async with session.post(
        f"http://{backend_ip}:{backend_port}/{request.eval_type.value}",
        json=payload,
      ) as response:
        result = await response.json()
        logging.info(
          f"Received response from {backend_name}: {response.status}"
        )

        if response.status != 200:
          if response.status == 408:
            raise HTTPException(
              status_code=408,
              detail=f"Backend evaluation timed out. Timeout was set to {backend_timeout} seconds.",
            )
          raise HTTPException(
            status_code=response.status,
            detail=result.get("detail", "Backend evaluation failed"),
          )

    # Mark backend as available after evaluation
    logging.info(
      f"Evaluation completed on {backend_name}, marking as available"
    )
    return result

  except HTTPException as e:
    logging.info(f"HTTPException occurred during evaluation: {e.detail}")
    raise e
  except Exception as e:
    logging.error(f"Error occurred while evaluating on {backend_name}: {e}")
    raise HTTPException(
      status_code=500, detail=f"Backend evaluation failed: {e}"
    )
  finally:
    backend.set_status("available")


if __name__ == "__main__":
  import uvicorn

  uvicorn.run(app, host="0.0.0.0", port=EVAL_SERVER_PORT)
