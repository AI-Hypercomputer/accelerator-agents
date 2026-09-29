#!/usr/bin/env bash
# Nightly MaxKernel sweep: pick the N least-recently-run problems and run each
# one in its own headless `claude -p` conversation.
#
#   tools/nightly/run_batch.sh                  # rotation: next MK_BATCH problems
#   tools/nightly/run_batch.sh 63p_GDN rpav3    # explicit list, ignores rotation
#   MK_BATCH=6 MK_JOBS=3 tools/nightly/run_batch.sh
#
# Knobs (env):
#   MK_BATCH     how many problems this invocation attempts   (default 4)
#   MK_JOBS      how many run concurrently                    (default 2)
#   MK_TIMEOUT   per-problem wall-clock cap                   (default 10h)
#   MK_DEADLINE  cap for the whole batch                      (default 20h)
#   MK_MODEL     model alias passed to claude                 (default opus)
#   MK_BUDGET_USD  optional --max-budget-usd per problem
set -uo pipefail

MK="${MK:-/home/cathygao_google_com/maxkernel}"
NIGHTLY="$MK/workspace/_nightly"
LASTRUN="$NIGHTLY/last_run.tsv"
mkdir -p "$NIGHTLY"

BATCH="${MK_BATCH:-4}"
JOBS="${MK_JOBS:-2}"
DEADLINE="${MK_DEADLINE:-20h}"

# Never stack two sweeps. A single run can last 24h; cron will fire again long
# before that. If the previous sweep is still going, this one exits quietly.
exec 9>"$NIGHTLY/.lock"
if ! flock -n 9; then
  echo "[$(date -Is)] previous sweep still running; skipping this trigger"
  exit 0
fi

# Problems the guard or duplication makes unrunnable. Edit freely.
#   tokamax/*  -> blocked by the Read(//**/tokamax/**) deny rule in settings
#   DSV4/*     -> case-variant duplicate of DSv4/*, and has no kernel_task.yaml
EXCLUDE_RE='^(tokamax/|DSV4/)'

discover() {
  find "$MK/problems" -maxdepth 3 \
       \( -name baseline.py -o -name reference.py -o -name kernel.py \) \
       -printf '%h\n' \
  | sed "s#^$MK/problems/##" \
  | sort -u \
  | grep -Ev "$EXCLUDE_RE"
}

if [ "$#" -gt 0 ]; then
  SELECTED=("$@")
else
  # Rotation: order by last attempt time ascending (never-run = 0), take BATCH.
  touch "$LASTRUN"
  mapfile -t SELECTED < <(
    discover | while read -r p; do
      t=$(awk -F'\t' -v p="$p" '$1 == p {print $2}' "$LASTRUN")
      printf '%s\t%s\n' "${t:-0}" "$p"
    done | sort -n -k1,1 | head -n "$BATCH" | cut -f2
  )
fi

[ "${#SELECTED[@]}" -gt 0 ] || { echo "nothing to run"; exit 0; }

echo "[$(date -Is)] sweep start: ${#SELECTED[@]} problem(s), ${JOBS} at a time"
printf '  %s\n' "${SELECTED[@]}"

printf '%s\n' "${SELECTED[@]}" \
  | timeout --signal=INT --kill-after=10m "$DEADLINE" \
      xargs -r -n1 -P "$JOBS" -I{} "$MK/tools/nightly/run_one.sh" {}

echo "[$(date -Is)] sweep done"
echo "results: $NIGHTLY/results.csv"
