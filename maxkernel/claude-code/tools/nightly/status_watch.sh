#!/usr/bin/env bash
# Refresh workspace/_nightly/STATUS.md every MK_STATUS_INTERVAL seconds.
# Detached companion to run_batch.sh; exits once the sweep is gone and one
# final page has been written.
set -uo pipefail
MK="${MK:-/home/cathygao_google_com/maxkernel}"
PY="${MK_PY:-/home/cathygao_google_com/maxkernel_venv/bin/python}"
INTERVAL="${MK_STATUS_INTERVAL:-60}"
cd "$MK" || exit 1
while true; do
  "$PY" "$MK/tools/nightly/status.py" >/dev/null 2>&1
  pgrep -f "run_batch.sh KernelBench" >/dev/null || { sleep "$INTERVAL"; \
    "$PY" "$MK/tools/nightly/status.py" >/dev/null 2>&1; exit 0; }
  sleep "$INTERVAL"
done
