---
title: "XProf docs: Docker deployment"
type: source
tags: [docs, profiler, deployment, docker, gcs]
created: 2026-04-22
updated: 2026-04-22
---

Minimal recipe for building and running an XProf Docker image off the public PyPI release. This is operational background for experiment-runners — not a performance surface — but it documents the expected port map (`8791` UI, `50051` worker gRPC) and the two supported logdir mount patterns (local directory vs. GCS with mounted `gcloud` creds). The doc is short and example-driven; everything below is a compression, not an expansion.

## Overview

An XProf container is just `python:3.12-slim + pip install xprof==<ver>`, with `ENTRYPOINT ["xprof"]` and a default `CMD` that serves `/app/logs` on port `8791`. Three important parts for a runner:

1. **Ports exposed**: `8791` (XProf UI / HTTP), `50051` (worker gRPC — relevant when running as an aggregator/worker; see the Kubernetes deployment doc).
2. **Logdir mounts**: either `-v /local/logs:/app/logs` for a local logdir, or mount host gcloud creds and pass `--logdir=gs://bucket/path` for a GCS logdir.
3. **Version pinning**: `ARG XPROF_VERSION=2.21.3` is the doc's example; overridable via `--build-arg`.

## Key claims

- Base image is `python:3.12-slim`; install is `pip install --no-cache-dir xprof==${XPROF_VERSION}`.
- The Dockerfile is explicitly labeled a **basic configuration** — intended as a starting point, not a hardened image.
- Build must set `--platform=linux/amd64` (relevant on arm64 dev machines).
- Local-logs pattern: `docker run -p 8791:8791 -v /tmp/xprof_logs:/app/logs xprof:<ver>`.
- GCS-logs pattern: mount `~/.config/gcloud` into `/root/.config/gcloud` and pass `--logdir=gs://...` so XProf can authenticate as the host user's gcloud identity.

## Key data points

### Dockerfile (doc verbatim)

```dockerfile
FROM python:3.12-slim

ARG XPROF_VERSION=2.21.3

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir xprof==${XPROF_VERSION}

EXPOSE 8791 50051

ENTRYPOINT ["xprof"]

CMD ["--logdir=/app/logs", "--port=8791"]
```

### Commands

| Step | Command |
|---|---|
| Build | `docker build --platform=linux/amd64 -t xprof:2.21.3 .` |
| Override version | `docker build --build-arg XPROF_VERSION=<ver> -t xprof:<ver> .` |
| Run, local logs | `docker run -p 8791:8791 -v /tmp/xprof_logs:/app/logs xprof:2.21.3` |
| Run, GCS logs | `docker run -p 8791:8791 -v ~/.config/gcloud:/root/.config/gcloud xprof:2.21.3 --logdir=gs://your-bucket/xprof_logs --port=8791` |

### Ports

| Container port | Purpose |
|---|---|
| 8791 | XProf UI / HTTP |
| 50051 | Worker gRPC (aggregator/worker mode) |

## Techniques referenced

- **Pinned-version pip install** into a slim Python base image.
- **Port mapping** for the XProf UI and worker gRPC port.
- **Host gcloud credential mount** for GCS-backed logdirs (`-v ~/.config/gcloud:/root/.config/gcloud`).
- **Build-arg version override** pattern.

## Gaps & caveats

- Dockerfile is intentionally minimal — no healthcheck, no non-root user, no TLS termination. Production hardening is out of scope.
- `--platform=linux/amd64` is explicit because default on Apple Silicon builds an arm64 image that may not match the deployment target.
- The `50051` expose is only meaningful if running in aggregator/worker mode (`--grpc_port=50051` or similar); the default `CMD` here runs a standalone server and does not use it.
- GCS auth via mounted `~/.config/gcloud` uses the host's *user* credentials — for production, prefer a service account and ADC with a key mount or Workload Identity. The doc uses user creds for simplicity.
- Pinned `XPROF_VERSION=2.21.3` is the doc's example as of writing; newer XProf releases may require updated protos (tool views can change across versions). An experiment's `raw/profiles/` capture should ideally be viewed with a version ≥ the one that produced it.
- No log-rotation or disk-pressure guidance for the local-logs volume.

## Connections

- `xprof-deployment` — how the profiler UI is hosted.
- `xprof-logdir-gcs` — GCS-backed logdir pattern with mounted gcloud creds.
- `xprof-ports` — 8791 (UI) / 50051 (worker gRPC).
- `cloud-diagnostics-xprof` — recommended GCP wrapper (referenced in capturing-profiles doc; subsumes this manual Docker recipe).

## See also

- [xprof](../codebases/xprof.md)
- [XProf docs: Kubernetes deployment](2026-xprof-kubernetes-deployment.md)
- [XProf docs: Capturing profiles](2026-xprof-capturing-profiles.md)

## Sources

- `raw/code/xprof/docs/docker_deployment.md`
