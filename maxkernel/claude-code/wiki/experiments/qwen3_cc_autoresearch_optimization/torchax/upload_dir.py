"""Upload a local directory tree to a gs:// prefix (HLO dumps, etc.).

Used in the GKE pod command to publish the XLA HLO dump (written to a local /tmp
dir during compile) to the experiment's GCS run folder, so xprof-mcp's HLO tools
and the experiment page can reach it. Uses the google-cloud-storage client (no
gsutil dependency).

    python upload_dir.py <local_dir> gs://<bucket>/<prefix>

Idempotent-ish: re-uploading overwrites same-named blobs. Safe to run even if
the
local dir is empty (prints a notice and exits 0) so it never fails the pod.
"""

from __future__ import annotations

import os
import sys


def main(local_dir: str, gcs_uri: str) -> int:
  if not gcs_uri.startswith("gs://"):
    print(f"[upload_dir] not a gs:// uri: {gcs_uri}", flush=True)
    return 2
  if not os.path.isdir(local_dir):
    print(
        f"[upload_dir] no such dir (nothing to upload): {local_dir}", flush=True
    )
    return 0

  from google.cloud import storage

  bucket_name, _, prefix = gcs_uri[len("gs://") :].partition("/")
  client = storage.Client()
  bucket = client.bucket(bucket_name)

  n = 0
  total = 0
  for root, _, files in os.walk(local_dir):
    for fn in files:
      lp = os.path.join(root, fn)
      rel = os.path.relpath(lp, local_dir)
      blob_name = f"{prefix.rstrip('/')}/{rel}" if prefix else rel
      bucket.blob(blob_name).upload_from_filename(lp)
      n += 1
      total += os.path.getsize(lp)
  print(
      f"[upload_dir] uploaded {n} files ({total/1e6:.1f} MB) "
      f"{local_dir} -> {gcs_uri}",
      flush=True,
  )
  return 0


if __name__ == "__main__":
  if len(sys.argv) != 3:
    print(
        "usage: python upload_dir.py <local_dir> gs://<bucket>/<prefix>",
        flush=True,
    )
    raise SystemExit(2)
  raise SystemExit(main(sys.argv[1], sys.argv[2]))
