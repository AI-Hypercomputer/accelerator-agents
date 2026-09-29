---
title: "XProf docs: Kubernetes distributed deployment"
type: source
tags: [docs, profiler, deployment, kubernetes, aggregator-worker]
created: 2026-04-22
updated: 2026-04-22
---

How to run XProf distributed on Kubernetes using an **aggregator + worker** pattern — a single-replica aggregator accepts UI traffic and dispatches profiling tasks round-robin to N worker replicas behind a headless Service. Operational background for experiment-runners; the important facts are the port layout, the round-robin gRPC discovery mechanism (headless Service + `GRPC_LB_POLICY=round_robin`), and the explicit incompatibility with Horizontal Pod Autoscaling.

## Overview

Two Deployments + two Services:

- **Aggregator** (1 replica): exposes the UI HTTP port (example `10000`), runs `xprof` with `--worker_service_address=dns:///xprof-worker-service.default.svc.cluster.local:8891` and `-gp=50051` and `--hide_capture_profile_button`. Fronted by a NodePort Service.
- **Worker** (N replicas): each runs `xprof --port=9999 -gp=8891 --hide_capture_profile_button` and exposes `containerPort: 8891`. Fronted by a **headless** ClusterIP Service (`clusterIP: None`) so gRPC DNS returns all pod IPs for `round_robin` client-side load balancing.

The aggregator dispatches requests to workers via `dns:///` URIs; gRPC's native DNS resolver (`GRPC_DNS_RESOLVER=native`) + round-robin policy (`GRPC_LB_POLICY=round_robin`) does the fan-out.

## Key claims

- **HPA for worker pods is explicitly incompatible** with this setup — gRPC's DNS resolution may not pick up replica-count changes reliably, so scaling out workers can silently fail to rebalance.
- The aggregator and workers both run with `--hide_capture_profile_button` — this deployment is for **viewing/analysis**, not for triggering on-demand capture from the UI.
- Worker discovery uses a **headless Service** (`clusterIP: None`) so that the DNS A record for `xprof-worker-service` returns every pod IP, not a single virtual IP. This is required for gRPC `round_robin` to have more than one endpoint to balance across.
- Port numbers in the YAML are examples and can be customized; the aggregator-to-worker dispatch uses worker port `8891` (`-gp=8891`).
- Build a local image via the Docker-deployment doc, then reference it with `imagePullPolicy: Never` in the manifests (assumes minikube or a locally-loaded image).
- NodePort `30001` is used to expose the aggregator externally in the example; local access via `minikube service xprof-agg-service --url`.

## Key data points

### Architecture summary

| Role | Replicas | Container port | Service | Flags |
|---|---|---|---|---|
| Aggregator | 1 | 10000 (HTTP UI) | `xprof-agg-service` (NodePort, nodePort 30001) | `--port=10000 --worker_service_address=dns:///xprof-worker-service.default.svc.cluster.local:8891 -gp=50051 --hide_capture_profile_button` |
| Worker | N (example 4) | 8891 (gRPC) | `xprof-worker-service` (headless, `clusterIP: None`) | `--port=9999 -gp=8891 --hide_capture_profile_button` |

### Required env vars on the aggregator container

| Env var | Value | Purpose |
|---|---|---|
| `GRPC_LB_POLICY` | `round_robin` | gRPC client-side load-balancing policy |
| `GRPC_DNS_RESOLVER` | `native` | Use gRPC's native DNS resolver (returns all A records) |

### Deploy commands

```sh
kubectl apply -f worker.yaml
kubectl apply -f agg.yaml
kubectl get services
minikube service xprof-agg-service --url
```

### Expected services

```
NAME                   TYPE        CLUSTER-IP     PORT(S)
xprof-agg-service      NodePort    10.96.13.172   8080:30001/TCP
xprof-worker-service   ClusterIP   None           80/TCP
```

### Flag glossary (relevant to this deployment)

| Flag | Meaning |
|---|---|
| `--port=<p>` | HTTP port for the XProf UI / aggregator frontend |
| `-gp=<p>` / `--grpc_port=<p>` | gRPC port the worker listens on (aggregator targets this) |
| `--worker_service_address=<uri>` | DNS URI the aggregator dispatches to (on workers) |
| `--hide_capture_profile_button` / `-hcpb` | Disable the UI capture button (viewing-only deployment) |

## Techniques referenced

- **Aggregator/worker XProf topology** for large-profile analysis.
- **Headless Kubernetes Service** (`clusterIP: None`) for gRPC endpoint discovery.
- **gRPC round-robin LB** via `GRPC_LB_POLICY=round_robin` + `GRPC_DNS_RESOLVER=native`.
- **`--hide_capture_profile_button`** flag to disable on-demand capture in shared/viewing deployments.
- **NodePort** service to expose the aggregator UI in minikube.

## Gaps & caveats

- **Incompatible with HPA** on worker pods — scaling up/down may not reach the aggregator's LB. If more capacity is needed, scale via explicit `replicas` and restart the aggregator.
- This deployment is **viewing-only** (capture button hidden) — on-demand capture from the UI is disabled. Programmatic capture from the workload side still works.
- Ports `10000` / `8891` / `9999` / `30001` are examples; nothing in the code requires these numbers. Consistency across aggregator args and worker Service `targetPort` is the user's responsibility.
- `imagePullPolicy: Never` in the examples assumes a locally loaded image (minikube). For real clusters, replace with a registry pull policy.
- No TLS / mTLS, no auth, no NetworkPolicy in the manifests — this is a minimal example, not a production template.
- No guidance on **persistent storage for profiles** — workers need access to the same logdir (typically a GCS bucket); how that is wired (volumes, service account) is left to the user.
- Architecture diagram referenced (`images/kubernetes/xprof_aggregator_worker_architecture_for_kubernetes.png`) is under `raw/code/xprof/docs/images/kubernetes/` but not reproduced here.

## Connections

- `xprof-deployment` — how the profiler UI is hosted.
- `xprof-aggregator-worker` — distributed viewing topology (aggregator fan-out to workers).
- `grpc-round-robin-lb` — client-side LB pattern used here.
- `headless-service-grpc-discovery` — Kubernetes pattern for gRPC endpoint enumeration.
- `xprof-ports` — UI vs gRPC port roles.

## See also

- [xprof](../codebases/xprof.md)
- [XProf docs: Docker deployment](2026-xprof-docker-deployment.md)
- [XProf docs: Capturing profiles](2026-xprof-capturing-profiles.md)

## Sources

- `raw/code/xprof/docs/kubernetes_deployment.md`
