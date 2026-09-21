# pylint: skip-file
#!/usr/bin/env python3
import os

PATH = "/home/gvanica_google_com/vllm/vllm/config/vllm.py"

if not os.path.exists(PATH):
  print(f"Error: File {PATH} does not exist!")
  exit(1)

with open(PATH, "r") as f:
  content = f.read()

print(
    "=== Patching vllm.py to support class-based distributed_executor_backend"
    " check ==="
)

target = (
    "        executor_backend ="
    " self.parallel_config.distributed_executor_backend\n       "
    " executor_supports_async_sched = executor_backend in (\n           "
    ' "mp",\n            "uni",\n            "external_launcher",\n        )'
)

replacement = (
    "        executor_backend ="
    " self.parallel_config.distributed_executor_backend\n        backend_name ="
    " executor_backend if isinstance(executor_backend, str) else"
    ' getattr(executor_backend, "__name__", str(executor_backend))\n       '
    ' executor_supports_async_sched = backend_name in (\n            "mp",\n   '
    '         "uni",\n            "external_launcher",\n            "ray",\n   '
    '     ) or "RayDistributedExecutor" in backend_name'
)

if target in content:
  content = content.replace(target, replacement)
  with open(PATH, "w") as f:
    f.write(content)
  print("SUCCESS: Patched vllm.py successfully!")
else:
  # If already patched
  if "backend_name = executor_backend" in content:
    print("INFO: vllm.py is already patched.")
  else:
    print("ERROR: Target block not found in vllm.py!")
    exit(1)
