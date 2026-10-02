"""FastAPI server for executing MaxKernel TPU workloads."""

import asyncio
import base64
import contextlib
import enum
import io
import itertools
import json
import logging
import os
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from typing import Any, Dict, List, Optional

import fastapi
import pydantic
import uvicorn

logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
  datefmt="%Y-%m-%d %H:%M:%S",
)

MAX_RUNNING_TIMEOUT_MARGIN = (
  180  # 3 mins margin past specified timeout before force kill
)
DEFAULT_MAX_JOB_RUNTIME = 1800  # 30 mins hard max runtime if unspecified
MAX_QUEUE_WAIT_TIME = 7200  # 2 hours max waiting in queue before auto cancel
MAX_FINISHED_RETENTION_TIME = (
  3600  # keep finished jobs for 1 hr before memory pruning
)


class CodeRequest(pydantic.BaseModel):
  """Request body for running single TPU code snippet."""

  code: str
  timeout: Optional[int] = 180
  dependencies: Optional[Dict[str, str]] = None


class AutotuneRequest(pydantic.BaseModel):
  """Request body for autotuning TPU code snippets."""

  code_template: str
  search_space: Dict[str, List[Any]]
  timeout: Optional[int] = 120
  total_timeout: Optional[int] = 1800
  dependencies: Optional[Dict[str, str]] = None


class CodeResponse(pydantic.BaseModel):
  """Response body for code execution results."""

  output: str
  error: Optional[str] = None
  exit_code: int
  trace_tar_b64: Optional[str] = None


def _collect_trace_artifacts(temp_dir: str) -> Optional[str]:
  """Scans temp_dir and /tmp for trace files; returns base64 tar.gz."""
  try:
    if not temp_dir or not os.path.exists(temp_dir):
      return None

    trace_files = []
    for root, _, files in os.walk(temp_dir):
      for f in files:
        if (
          f.endswith((".xplane.pb", ".trace.json.gz", "perfetto_trace.json.gz"))
          or "plugins/profile" in root
        ):
          trace_files.append(os.path.join(root, f))

    tmp_dirs_to_clean = []
    if not trace_files:
      for entry in os.listdir("/tmp"):
        entry_path = os.path.join("/tmp", entry)
        if os.path.isdir(entry_path) and (
          "trace" in entry.lower()
          or "tensorboard" in entry.lower()
          or "profile" in entry.lower()
        ):
          try:
            mtime = os.path.getmtime(entry_path)
            if time.time() - mtime < 180:
              tmp_dirs_to_clean.append(entry_path)
              for root, _, files in os.walk(entry_path):
                for f in files:
                  if f.endswith(
                    (
                      ".xplane.pb",
                      ".trace.json.gz",
                      "perfetto_trace.json.gz",
                    )
                  ):
                    trace_files.append(os.path.join(root, f))
          except OSError:
            pass

    if not trace_files:
      return None

    # Prioritize .xplane.pb to avoid transmitting massive redundant JSON trace files
    has_xplane = any(f.endswith(".xplane.pb") for f in trace_files)
    if has_xplane:
      filtered = [f for f in trace_files if f.endswith(".xplane.pb")]
      if filtered:
        trace_files = filtered

    trace_files.sort(
      key=lambda f: os.path.getsize(f) if os.path.exists(f) else 0
    )

    buf = io.BytesIO()
    max_tar_bytes = 25 * 1024 * 1024  # 25 MB max limit
    current_bytes = 0

    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
      for fpath in trace_files:
        if not os.path.exists(fpath):
          continue
        fsize = os.path.getsize(fpath)
        if current_bytes + fsize > max_tar_bytes and current_bytes > 0:
          logging.warning(
            "Skipping trace file %s (%d bytes) to stay within %d MB limit.",
            fpath,
            fsize,
            max_tar_bytes // (1024 * 1024),
          )
          continue
        arcname = os.path.relpath(fpath, temp_dir)
        if arcname.startswith(".."):
          arcname = (
            os.path.basename(os.path.dirname(fpath))
            + "/"
            + os.path.basename(fpath)
          )
        tar.add(fpath, arcname=arcname)
        current_bytes += fsize

    buf.seek(0)
    raw_tar = buf.read()
    b64_str = base64.b64encode(raw_tar).decode("utf-8")
    logging.info(
      "Collected %d trace artifacts (%d compressed bytes, %d base64 bytes).",
      len(trace_files),
      len(raw_tar),
      len(b64_str),
    )

    for dpath in tmp_dirs_to_clean:
      try:
        shutil.rmtree(dpath, ignore_errors=True)
      except OSError:
        pass

    return b64_str
  except (OSError, tarfile.TarError) as e:
    logging.warning("Error collecting trace artifacts: %s", e)
    return None


class JobStatus(str, enum.Enum):
  """Status enum for TPU jobs."""

  QUEUED = "queued"
  RUNNING = "running"
  COMPLETED = "completed"
  FAILED = "failed"
  CANCELLED = "cancelled"


class JobSubmission(pydantic.BaseModel):
  """Submission payload for a TPU job."""

  action: str
  code_request: Optional[CodeRequest] = None
  autotune_request: Optional[AutotuneRequest] = None


class JobResponse(pydantic.BaseModel):
  """Status response payload for a TPU job."""

  job_id: str
  status: JobStatus
  action: str
  queue_position: int = 0
  created_at: float
  started_at: Optional[float] = None
  completed_at: Optional[float] = None
  result: Optional[CodeResponse] = None
  error: Optional[str] = None


class JobRecord:
  """In-memory tracking record for a submitted TPU job."""

  def __init__(self, job_id: str, action: str, submission: JobSubmission):
    self.job_id = job_id
    self.action = action
    self.submission = submission
    self.status = JobStatus.QUEUED
    self.created_at = time.time()
    self.started_at: Optional[float] = None
    self.completed_at: Optional[float] = None
    self.result: Optional[CodeResponse] = None
    self.error: Optional[str] = None
    self.done_event = asyncio.Event()
    self.current_process: Optional[asyncio.subprocess.Process] = None
    self.current_pgid: Optional[int] = None

  def kill_process(self):
    """Terminates active process and process group for this job."""
    if self.current_pgid is not None:
      try:
        os.killpg(self.current_pgid, signal.SIGKILL)
      except OSError:
        pass
    if self.current_process and self.current_process.returncode is None:
      try:
        self.current_process.kill()
      except OSError:
        pass


jobs: Dict[str, JobRecord] = {}
_job_queue: Optional[asyncio.Queue[str]] = None
compilation_semaphore = asyncio.Semaphore(4)
hardware_semaphore = asyncio.Semaphore(1)


def get_job_queue() -> asyncio.Queue[str]:
  """Returns or initializes the global asyncio job queue."""
  global _job_queue
  if _job_queue is None:
    _job_queue = asyncio.Queue()
  return _job_queue


def _get_queue_position(target_job_id: str) -> int:
  """Calculates the 0-based position of a job in the queue."""
  target_job = jobs.get(target_job_id)
  if not target_job or target_job.status in (
    JobStatus.COMPLETED,
    JobStatus.FAILED,
    JobStatus.CANCELLED,
  ):
    return 0
  if target_job.status == JobStatus.RUNNING:
    return 0

  pos = 0
  for j_id, j in jobs.items():
    if j.status in (JobStatus.QUEUED, JobStatus.RUNNING):
      if j.created_at < target_job.created_at or (
        j.created_at == target_job.created_at and j_id < target_job_id
      ):
        pos += 1
  return pos


def _sanitize_code(code: str) -> str:
  """Validates that code does not contain forbidden dangerous operations."""
  dangerous_patterns = [
    r"os\.system\s*\(",
    r"subprocess\.",
    r"os\.popen\s*\(",
  ]
  for pattern in dangerous_patterns:
    if re.search(pattern, code):
      raise fastapi.HTTPException(
        status_code=400,
        detail=(
          f"Security violation: Code contains forbidden pattern '{pattern}'"
        ),
      )
  return code


def _save_dependencies(dependencies: Optional[Dict[str, str]], temp_dir: str):
  """Saves optional dependency files into specified temporary directory."""
  if not dependencies:
    return
  base_dir = pathlib.Path(temp_dir).resolve()
  for filename, content in dependencies.items():
    if not filename:
      raise fastapi.HTTPException(
        status_code=400, detail="Invalid dependency filename"
      )
    normalized_filename = filename.replace("\\", "/")
    target_path = (base_dir / normalized_filename).resolve()
    if not target_path.is_relative_to(base_dir) or target_path == base_dir:
      raise fastapi.HTTPException(
        status_code=400, detail="Path traversal detected"
      )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(content)


def _handle_execution_result(output: str, error: Optional[str], exit_code: int):
  """Processes stdout/stderr to detect OOM and lock conflicts."""
  if error and "Out of memory" in error:
    error = (
      "CRITICAL: Out of Memory (OOM) encountered during execution.\n" + error
    )

  busy_match = None
  if error:
    busy_match = re.search(
      r"already in use by process with pid (\d+)", error, re.IGNORECASE
    )
  if not busy_match and output:
    busy_match = re.search(
      r"already in use by process with pid (\d+)", output, re.IGNORECASE
    )

  if busy_match:
    pid = int(busy_match.group(1))
    is_active = False
    ps_out = ""
    try:
      ps_out = (
        subprocess.check_output(
          ["ps", "-o", "user,state,cmd", "-p", str(pid)],
          stderr=subprocess.STDOUT,
        )
        .decode()
        .strip()
      )
      lines = ps_out.split("\n")
      if len(lines) > 1:
        parts = lines[1].split()
        if len(parts) > 1 and "Z" not in parts[1]:
          is_active = True
    except (subprocess.SubprocessError, OSError):
      pass

    if is_active:
      error = (
        "HUMAN INTERVENTION REQUIRED: Conflict! The TPU is currently locked"
        f" by active external process {pid}.\nDetails:\n{ps_out}\nAGENT"
        " INSTRUCTION: Halt execution and ask the human user to resolve this"
        " conflict. DO NOT attempt to kill the process yourself."
      )
      exit_code = 409
    else:
      logging.warning(
        "Detected STALE TPU lock by dead/zombie PID %d. Cleaning up"
        " lockfiles...",
        pid,
      )
      subprocess.call(
        "sudo rm -f /tmp/libtpu_lockfile /tmp/tpu_logs/*lock*", shell=True
      )
      error = (
        "STALE LOCK RECOVERED: The TPU was locked by a crashed/zombie"
        f" process (PID {pid}). The server has automatically cleaned the"
        " stale lock files. Please retry this request!"
      )
      exit_code = 503

  return output, error, exit_code


async def _execute_code_internal(
  request: CodeRequest, test_name: str, job: Optional[JobRecord] = None
) -> CodeResponse:
  """Executes Python code snippet in an isolated subprocess with retries."""
  logging.info("Starting %s", test_name)
  max_retries = 3
  output, error, exit_code = "", "", 1
  temp_dir = None
  for attempt in range(max_retries):
    temp_dir = None
    process = None
    pgid = None
    try:
      safe_code = _sanitize_code(request.code)
      temp_dir = tempfile.mkdtemp()
      temp_file_path = os.path.join(temp_dir, "run_code.py")
      _save_dependencies(request.dependencies, temp_dir)

      with open(temp_file_path, "w") as f:
        f.write(safe_code)

      process = await asyncio.create_subprocess_exec(
        sys.executable,
        temp_file_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=temp_dir,
        preexec_fn=os.setsid,
      )
      try:
        pgid = os.getpgid(process.pid)
      except OSError:
        pgid = None

      if job:
        job.current_process = process
        job.current_pgid = pgid

      try:
        stdout, stderr = await asyncio.wait_for(
          process.communicate(), timeout=request.timeout
        )
        output = stdout.decode("utf-8") if stdout else ""
        error = stderr.decode("utf-8") if stderr else None
        exit_code = process.returncode

        output, error, exit_code = _handle_execution_result(
          output, error, exit_code
        )

        if exit_code == 409 and attempt < max_retries - 1:
          logging.info(
            "TPU busy (attempt %d/%d). Waiting 5s for TPU to free up...",
            attempt + 1,
            max_retries,
          )
          await asyncio.sleep(5)
          continue
        if exit_code == 503 and attempt < max_retries - 1:
          logging.info("Stale lock cleared. Retrying execution immediately...")
          continue

        trace_b64 = _collect_trace_artifacts(temp_dir)
        return CodeResponse(
          output=output,
          error=error,
          exit_code=exit_code,
          trace_tar_b64=trace_b64,
        )

      except asyncio.TimeoutError:
        logging.error("%s timed out after %ds", test_name, request.timeout)
        return CodeResponse(
          output="",
          error=f"Code execution timed out after {request.timeout}s",
          exit_code=1,
        )
    except fastapi.HTTPException as e:
      return CodeResponse(output="", error=e.detail, exit_code=1)
    except OSError as e:
      return CodeResponse(
        output="", error=f"Execution error: {str(e)}", exit_code=1
      )
    finally:
      if job:
        job.current_process = None
        job.current_pgid = None
      if process and process.returncode is None:
        if pgid is not None:
          try:
            os.killpg(pgid, signal.SIGKILL)
          except OSError:
            pass
        try:
          process.kill()
          await process.wait()
        except OSError:
          pass
      if temp_dir and os.path.exists(temp_dir):
        try:
          shutil.rmtree(temp_dir)
        except OSError:
          pass
  return CodeResponse(
    output=output,
    error=error,
    exit_code=exit_code,
    trace_tar_b64=_collect_trace_artifacts(temp_dir) if temp_dir else None,
  )


async def _execute_autotune_internal(
  request: AutotuneRequest, job: Optional[JobRecord] = None
) -> CodeResponse:
  """Executes autotune parameter sweep over search space."""
  logging.info("Starting autotune execution")
  temp_dir = tempfile.mkdtemp()
  process = None
  pgid = None
  try:
    _save_dependencies(request.dependencies, temp_dir)
    keys = list(request.search_space.keys())
    values = list(request.search_space.values())
    combinations = list(itertools.product(*values))

    all_results = []
    start_time = time.time()
    for combo in combinations:
      if (
        request.total_timeout
        and (time.time() - start_time) > request.total_timeout
      ):
        logging.info(
          "Autotune total timeout reached (%ds). Stopping sweep...",
          request.total_timeout,
        )
        break
      cfg = dict(zip(keys, combo))

      try:
        code_content = request.code_template
        for k, v in cfg.items():
          code_content = code_content.replace(f"{{{k}}}", str(v))
        code_content = _sanitize_code(code_content)
      except fastapi.HTTPException:
        continue

      temp_file_path = os.path.join(temp_dir, "run_code.py")
      with open(temp_file_path, "w") as temp_file:
        temp_file.write(code_content)

      process = await asyncio.create_subprocess_exec(
        sys.executable,
        temp_file_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=temp_dir,
        preexec_fn=os.setsid,
      )
      try:
        pgid = os.getpgid(process.pid)
      except OSError:
        pgid = None

      if job:
        job.current_process = process
        job.current_pgid = pgid

      try:
        stdout, stderr = await asyncio.wait_for(
          process.communicate(), timeout=request.timeout
        )
        output = stdout.decode("utf-8") if stdout else ""
        error = stderr.decode("utf-8") if stderr else None
        exit_code = process.returncode

        output, error, exit_code = _handle_execution_result(
          output, error, exit_code
        )

        all_results.append(
          {
            "cfg": cfg,
            "exit_code": exit_code,
            "output": output,
            "error": error,
          }
        )
      except asyncio.TimeoutError:
        if pgid is not None:
          try:
            os.killpg(pgid, signal.SIGKILL)
          except OSError:
            pass
        try:
          process.kill()
          await process.wait()
        except OSError:
          pass
        all_results.append({"cfg": cfg, "status": "timeout"})
      finally:
        if job:
          job.current_process = None
          job.current_pgid = None

      await asyncio.sleep(0.5)

    return CodeResponse(
      output=json.dumps({"all_results": all_results}), exit_code=0
    )
  finally:
    if job:
      job.current_process = None
      job.current_pgid = None
    if process and process.returncode is None:
      if pgid is not None:
        try:
          os.killpg(pgid, signal.SIGKILL)
        except OSError:
          pass
      try:
        process.kill()
        await process.wait()
      except OSError:
        pass
    if temp_dir and os.path.exists(temp_dir):
      try:
        shutil.rmtree(temp_dir)
      except OSError:
        pass


async def _process_job_queue():
  """Worker loop that pulls and processes jobs from the asyncio queue."""
  q = get_job_queue()
  logging.info("TPU Job Queue worker task started.")
  while True:
    try:
      job_id = await q.get()
      job = jobs.get(job_id)
      if not job:
        q.task_done()
        continue

      if job.status == JobStatus.CANCELLED:
        q.task_done()
        continue

      job.status = JobStatus.RUNNING
      job.started_at = time.time()
      logging.info(
        "Worker processing job '%s' (action: %s)", job.job_id, job.action
      )

      try:
        async with hardware_semaphore:
          if job.status == JobStatus.CANCELLED:
            continue
          if job.action == "autotune":
            if job.submission.autotune_request:
              job.result = await _execute_autotune_internal(
                job.submission.autotune_request, job=job
              )
            else:
              job.result = CodeResponse(
                output="", error="Missing autotune_request", exit_code=1
              )
          else:
            if job.submission.code_request:
              job.result = await _execute_code_internal(
                job.submission.code_request, job.action, job=job
              )
            else:
              job.result = CodeResponse(
                output="", error="Missing code_request", exit_code=1
              )

          if job.status != JobStatus.CANCELLED:
            if job.result and job.result.exit_code == 0:
              job.status = JobStatus.COMPLETED
            else:
              job.status = JobStatus.FAILED
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.exception("Unexpected error processing job '%s': %s", job_id, e)
        if job.status != JobStatus.CANCELLED:
          job.status = JobStatus.FAILED
          job.error = str(e)
          job.result = CodeResponse(output="", error=str(e), exit_code=1)
      finally:
        job.completed_at = time.time()
        job.done_event.set()
        q.task_done()
        logging.info(
          "Finished job '%s' with status '%s'", job.job_id, job.status.value
        )

    except asyncio.CancelledError:
      break
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.error("Error in TPU job queue worker: %s", e)
      await asyncio.sleep(1)


async def _cleanup_stale_jobs():
  """Background loop enforcing job timeouts, TTL expiration, and pruning."""
  logging.info("TPU Job Lifecycle Cleanup Task started.")
  while True:
    try:
      await asyncio.sleep(15)
      now = time.time()
      jobs_to_delete = []

      for job_id, job in list(jobs.items()):
        # 1. Enforce running job max timeout
        if job.status == JobStatus.RUNNING and job.started_at:
          req_timeout = DEFAULT_MAX_JOB_RUNTIME
          if (
            job.submission.code_request and job.submission.code_request.timeout
          ):
            req_timeout = job.submission.code_request.timeout
          elif (
            job.submission.autotune_request
            and job.submission.autotune_request.total_timeout
          ):
            req_timeout = job.submission.autotune_request.total_timeout

          max_allowed = req_timeout + MAX_RUNNING_TIMEOUT_MARGIN
          if (now - job.started_at) > max_allowed:
            logging.error(
              "AUTO-CLEANUP: Job '%s' exceeded max runtime (%.1fs > %ds)."
              " Force killing!",
              job_id,
              now - job.started_at,
              max_allowed,
            )
            job.kill_process()
            job.status = JobStatus.FAILED
            job.error = (
              "AUTO CLEANUP: Job exceeded max allowed execution time"
              f" ({max_allowed}s) and was forcibly terminated."
            )
            job.result = CodeResponse(output="", error=job.error, exit_code=137)
            job.completed_at = now
            job.done_event.set()

        # 2. Enforce queued job TTL expiration
        elif job.status == JobStatus.QUEUED:
          if (now - job.created_at) > MAX_QUEUE_WAIT_TIME:
            logging.warning(
              "AUTO-CLEANUP: Job '%s' expired in queue after %.1fs."
              " Cancelling...",
              job_id,
              now - job.created_at,
            )
            job.status = JobStatus.CANCELLED
            job.error = (
              "QUEUE EXPIRED: Job waited too long in queue and was"
              " automatically cancelled."
            )
            job.completed_at = now
            job.done_event.set()

        # 3. Prune old finished job records from memory
        elif (
          job.status
          in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)
          and job.completed_at
        ):
          if (now - job.completed_at) > MAX_FINISHED_RETENTION_TIME:
            jobs_to_delete.append(job_id)

      for j_id in jobs_to_delete:
        jobs.pop(j_id, None)

    except asyncio.CancelledError:
      break
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.error("Error in TPU job cleanup task: %s", e)


background_tasks: set[asyncio.Task[Any]] = set()


@contextlib.asynccontextmanager
async def lifespan(app_instance: fastapi.FastAPI):
  """Lifecycle context manager replacing deprecated on_event handlers."""
  del app_instance
  logging.info(
    "Cleaning up orphan execution processes and stale lockfiles on startup..."
  )
  subprocess.call("pkill -9 -f run_cod[e].py || true", shell=True)
  subprocess.call(
    "sudo rm -f /tmp/libtpu_lockfile /tmp/tpu_logs/*lock*", shell=True
  )
  queue_task = asyncio.create_task(_process_job_queue())
  cleanup_task = asyncio.create_task(_cleanup_stale_jobs())
  background_tasks.add(queue_task)
  background_tasks.add(cleanup_task)
  queue_task.add_done_callback(background_tasks.discard)
  cleanup_task.add_done_callback(background_tasks.discard)
  yield


app = fastapi.FastAPI(title="MaxKernel TPU Execution Server", lifespan=lifespan)


@app.get("/health")
async def health_check():
  """Returns current health status of the server."""
  return {"status": "healthy", "service": "maxkernel-tpu-server"}


@app.post("/submit", response_model=JobResponse)
async def submit_job(submission: JobSubmission):
  """Submits a new job to the TPU execution queue."""
  valid_actions = [
    "compilation_test",
    "correctness_test",
    "performance_test",
    "profile",
    "autotune",
  ]
  if submission.action not in valid_actions:
    raise fastapi.HTTPException(
      status_code=400,
      detail=f"Invalid action '{submission.action}'. Valid: {valid_actions}",
    )

  job_id = f"job_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
  record = JobRecord(job_id, submission.action, submission)
  jobs[job_id] = record
  await get_job_queue().put(job_id)
  pos = _get_queue_position(job_id)
  logging.info(
    "Job '%s' submitted for action '%s'. Queue position: %d",
    job_id,
    submission.action,
    pos,
  )

  return JobResponse(
    job_id=job_id,
    status=record.status,
    action=record.action,
    queue_position=pos,
    created_at=record.created_at,
  )


@app.get("/job/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
  """Retrieves status and results for a specific job."""
  if job_id not in jobs:
    raise fastapi.HTTPException(
      status_code=404, detail=f"Job '{job_id}' not found"
    )
  record = jobs[job_id]
  return JobResponse(
    job_id=record.job_id,
    status=record.status,
    action=record.action,
    queue_position=_get_queue_position(job_id),
    created_at=record.created_at,
    started_at=record.started_at,
    completed_at=record.completed_at,
    result=record.result,
    error=record.error,
  )


@app.get("/queue")
async def get_queue():
  """Returns overview of queued and currently running jobs."""
  queued_jobs = [j for j in jobs.values() if j.status == JobStatus.QUEUED]
  running_jobs = [j for j in jobs.values() if j.status == JobStatus.RUNNING]
  return {
    "total_jobs": len(jobs),
    "queued_count": len(queued_jobs),
    "running_count": len(running_jobs),
    "queued_jobs": [
      {
        "job_id": j.job_id,
        "action": j.action,
        "position": _get_queue_position(j.job_id),
      }
      for j in queued_jobs
    ],
    "running_jobs": [
      {"job_id": j.job_id, "action": j.action} for j in running_jobs
    ],
  }


@app.post("/cancel")
async def cancel_jobs(job_id: Optional[str] = None):
  """Cancels one job, or every active job when `job_id` is omitted.

  The queue worker already honours `JobStatus.CANCELLED` (it skips cancelled
  jobs both on dequeue and after acquiring the hardware semaphore), and
  `JobRecord.kill_process()` already tears down the process group -- but
  nothing was reachable from outside the server to actually set that state.
  The MaxKernel emergency-stop protocol documents
  `tpu_client.py --cancel_job` as its recovery path, so without this endpoint
  a stuck job could only be cleared by killing the server.

  Args:
    job_id: Specific job to cancel. If omitted, all QUEUED/RUNNING jobs are
      cancelled.

  Returns:
    A summary of which jobs were cancelled.
  """
  if job_id is not None and job_id not in jobs:
    raise fastapi.HTTPException(
      status_code=404, detail=f"Job '{job_id}' not found"
    )

  targets = (
    [jobs[job_id]]
    if job_id is not None
    else [
      j
      for j in jobs.values()
      if j.status in (JobStatus.QUEUED, JobStatus.RUNNING)
    ]
  )

  cancelled = []
  skipped = []
  for job in targets:
    if job.status in (
      JobStatus.COMPLETED,
      JobStatus.FAILED,
      JobStatus.CANCELLED,
    ):
      skipped.append({"job_id": job.job_id, "status": job.status})
      continue
    job.status = JobStatus.CANCELLED
    job.error = "CANCELLED: Job was cancelled by client request."
    job.completed_at = time.time()
    job.kill_process()
    # Unblock anyone waiting in _submit_and_wait on this job.
    job.done_event.set()
    cancelled.append({"job_id": job.job_id, "action": job.action})
    logging.warning("Job '%s' cancelled by client request.", job.job_id)

  return {
    "cancelled_count": len(cancelled),
    "cancelled": cancelled,
    "skipped": skipped,
  }


async def _submit_and_wait(
  action: str,
  code_req: Optional[CodeRequest] = None,
  autotune_req: Optional[AutotuneRequest] = None,
) -> CodeResponse:
  """Helper to submit job and await completion synchronously."""
  submission = JobSubmission(
    action=action, code_request=code_req, autotune_request=autotune_req
  )
  job_id = f"job_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
  record = JobRecord(job_id, action, submission)
  jobs[job_id] = record
  await get_job_queue().put(job_id)
  await record.done_event.wait()
  if record.result:
    return record.result
  return CodeResponse(
    output="",
    error=record.error or "Execution failed without result",
    exit_code=1,
  )


@app.post("/compilation_test", response_model=CodeResponse)
async def compilation_test(request: CodeRequest):
  """Direct endpoint for compilation test."""
  return await _submit_and_wait("compilation_test", code_req=request)


@app.post("/correctness_test", response_model=CodeResponse)
async def correctness_test(request: CodeRequest):
  """Direct endpoint for correctness test."""
  return await _submit_and_wait("correctness_test", code_req=request)


@app.post("/performance_test", response_model=CodeResponse)
async def performance_test(request: CodeRequest):
  """Direct endpoint for performance test."""
  return await _submit_and_wait("performance_test", code_req=request)


@app.post("/profile", response_model=CodeResponse)
async def profile_test(request: CodeRequest):
  """Direct endpoint for profiling test."""
  return await _submit_and_wait("profile", code_req=request)


@app.post("/autotune", response_model=CodeResponse)
async def autotune(request: AutotuneRequest):
  """Direct endpoint for autotune sweep."""
  return await _submit_and_wait("autotune", autotune_req=request)


if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=8000)
