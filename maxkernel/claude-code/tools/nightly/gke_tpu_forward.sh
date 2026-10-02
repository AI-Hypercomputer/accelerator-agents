#!/usr/bin/env bash
# Bind 127.0.0.1:<port> to the MaxKernel TPU service running in the GKE cluster,
# and keep it bound.
#
#   tools/nightly/gke_tpu_forward.sh ensure      # bring it up if it is down
#   tools/nightly/gke_tpu_forward.sh supervise   # ensure, then watch forever
#   tools/nightly/gke_tpu_forward.sh status
#   tools/nightly/gke_tpu_forward.sh stop
#
# Why a port-forward and not a new tpu_client mode: tools/tpu_client.py already
# short-circuits in start_server_for_tpu() the moment 127.0.0.1:<port>/health
# answers {"status":"healthy"}, and the cluster's openresty gateway answers
# exactly that. So forwarding the service onto the port the client already uses
# makes every submit/profile/autotune land on cluster TPUs, with no local server
# booted and no change to tpu_client.py.
#
# The forward is the single point of failure for a multi-day sweep -- kubectl
# drops it on pod restarts, token expiry and network blips -- so `supervise`
# re-checks health on an interval and rebuilds it, re-running get-credentials
# when the failure looks like auth.
set -uo pipefail

MK="${MK:-/home/cathygao_google_com/maxkernel}"
export PATH="$HOME/.local/bin:$PATH"

CLUSTER="${MK_GKE_CLUSTER:-maxkernel-v6e-cluster}"
ZONE="${MK_GKE_ZONE:-us-east5-b}"
PROJECT="${MK_GKE_PROJECT:-tpu-prod-env-multipod}"
SERVICE="${MK_GKE_SERVICE:-svc/maxkernel-tpu-service}"
PORT="${MK_TPU_PORT:-8000}"
REMOTE_PORT="${MK_GKE_REMOTE_PORT:-8000}"
INTERVAL="${MK_FORWARD_INTERVAL:-15}"

NIGHTLY="$MK/_nightly"
mkdir -p "$NIGHTLY"
PIDFILE="$NIGHTLY/gke_forward.$PORT.pid"
LOGFILE="$NIGHTLY/gke_forward.$PORT.log"

log() { echo "[$(date -Is)] $*" | tee -a "$LOGFILE" >&2; }

healthy() {
  local body
  body="$(curl -fsS --max-time 5 "http://127.0.0.1:$PORT/health" 2>/dev/null)" || return 1
  [[ "$body" == *'"status":"healthy"'* ]]
}

have_creds() {
  kubectl --request-timeout=20s get svc "${SERVICE#svc/}" >/dev/null 2>&1
}

refresh_creds() {
  log "refreshing credentials for $CLUSTER"
  gcloud container clusters get-credentials "$CLUSTER" \
    --zone="$ZONE" --project="$PROJECT" >>"$LOGFILE" 2>&1
}

kill_forward() {
  if [ -f "$PIDFILE" ]; then
    local pid
    pid="$(cat "$PIDFILE" 2>/dev/null)"
    if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null
      sleep 1
      kill -9 "$pid" 2>/dev/null
    fi
    rm -f "$PIDFILE"
  fi
  # Anything else squatting on our port-forward for this service/port.
  pkill -f "kubectl.*port-forward.*${SERVICE}.*${PORT}:${REMOTE_PORT}" 2>/dev/null
}

start_forward() {
  kill_forward
  if ! have_creds; then
    refresh_creds || { log "get-credentials failed"; return 1; }
  fi
  log "starting port-forward $SERVICE ${PORT}:${REMOTE_PORT}"
  setsid kubectl port-forward "$SERVICE" "${PORT}:${REMOTE_PORT}" \
    >>"$LOGFILE" 2>&1 </dev/null &
  echo $! >"$PIDFILE"
  for _ in $(seq 1 20); do
    sleep 1
    healthy && { log "forward healthy on 127.0.0.1:$PORT"; return 0; }
  done
  log "forward did not become healthy on 127.0.0.1:$PORT"
  return 1
}

cmd_ensure() {
  if healthy; then
    return 0
  fi
  # Every concurrent op calls `ensure` before it starts, so without this lock a
  # simultaneous outage has N of them running kill_forward/start_forward against
  # each other and tearing down the forward they just built.
  exec 8>"$NIGHTLY/.gke_forward.$PORT.lock"
  flock 8
  healthy && return 0
  start_forward
}

cmd_supervise() {
  log "supervising 127.0.0.1:$PORT -> $CLUSTER/$SERVICE every ${INTERVAL}s"
  local fails=0
  while true; do
    if healthy; then
      fails=0
    else
      fails=$((fails + 1))
      log "health check failed (${fails})"
      # Two strikes before touching credentials: a single miss is usually the
      # gateway being busy, not the token being dead.
      if [ "$fails" -ge 2 ]; then
        have_creds || refresh_creds
        start_forward && fails=0
      fi
    fi
    sleep "$INTERVAL"
  done
}

# The cluster image ships jax, numpy and pydantic but not every import the
# benchmark sources use -- 64p_Linear_Softmax_Cross_Entropy needs jaxtyping, and
# without it the run dies in `exec(base.py)` with ModuleNotFoundError before a
# single kernel compiles. Jobs land on whichever pod pulls them, so all of them
# need the package. Installs are lost when a pod restarts, hence a command that
# is cheap to re-run rather than a one-off.
EXTRA_PIP="${MK_GKE_EXTRA_PIP:-jaxtyping}"

cmd_deps() {
  local pods pod rc=0
  pods="$(kubectl get pods -l app=maxkernel-tpu-server \
            -o jsonpath='{.items[*].metadata.name}' 2>/dev/null)"
  [ -n "$pods" ] || { log "no maxkernel-tpu-server pods found"; return 1; }

  for pod in $pods; do
    for pkg in $EXTRA_PIP; do
      if kubectl exec "$pod" -- python -c "import $pkg" >/dev/null 2>&1; then
        continue
      fi
      log "installing $pkg in $pod"
      if ! kubectl exec "$pod" -- pip install --no-cache-dir -q "$pkg" \
             >>"$LOGFILE" 2>&1; then
        log "FAILED to install $pkg in $pod"
        rc=1
        continue
      fi
      kubectl exec "$pod" -- python -c "import $pkg" >/dev/null 2>&1 \
        || { log "$pkg still not importable in $pod"; rc=1; }
    done
  done
  return $rc
}

cmd_status() {
  if healthy; then
    echo "UP    127.0.0.1:$PORT -> $CLUSTER $SERVICE"
    curl -fsS --max-time 5 "http://127.0.0.1:$PORT/health"; echo
    curl -fsS --max-time 5 "http://127.0.0.1:$PORT/queue"; echo
    return 0
  fi
  echo "DOWN  127.0.0.1:$PORT"
  return 1
}

case "${1:-ensure}" in
  ensure)    cmd_ensure ;;
  supervise) cmd_supervise ;;
  deps)      cmd_deps ;;
  status)    cmd_status ;;
  stop)      kill_forward; log "stopped"; ;;
  *) echo "usage: $0 {ensure|supervise|deps|status|stop}" >&2; exit 64 ;;
esac
