# pylint: skip-file
#!/usr/bin/env python3
import os

file_path = os.path.expanduser("~/bench_serving/backend_request_func.py")
with open(file_path, "r") as f:
  content = f.read()

target = """                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = chunk_bytes.decode("utf-8").removeprefix(
                            "data: ")
                        if chunk != "[DONE]":
                            data = json.loads(chunk)"""

replacement = """                    async for chunk_bytes in response.content:
                        if not chunk_bytes:
                            continue
                        lines = chunk_bytes.decode("utf-8").splitlines()
                        for line in lines:
                            line = line.strip()
                            if not line or not line.startswith("data: "):
                                continue
                            chunk = line.removeprefix("data: ")
                            if chunk == "[DONE]":
                                continue

                            print(f"[DEBUG RAW SSE] {chunk}")

                            try:
                                data = json.loads(chunk)
                            except Exception as e:
                                print(f"[DEBUG JSON ERROR] Failed to parse: {chunk}. Error: {e}")
                                continue"""

if target in content:
  content = content.replace(target, replacement)
  with open(file_path, "w") as f:
    f.write(content)
  print("SUCCESS: Patched backend_request_func.py successfully!")
else:
  print("ERROR: Target block not found in backend_request_func.py!")
