---
title: "Deploying XProf"
type: topic
provenance: docs
updated: 2026-07-03
status: fresh
---
# Deploying XProf

XProf can run standalone, as a TensorBoard plugin, in Docker, or on Kubernetes for distributed (multi-host) profiling.

## Deployment options
- **Standalone / TensorBoard** — install and run per the [README](../sources/readme.md) ([installation](src:README.md#installation), [distributed-profiling](src:README.md#distributed-profiling)).
- **Docker** — build/run an XProf image with local or GCS logs ([building-an-xprof-docker-image](src:docs/docker_deployment.md#building-an-xprof-docker-image), [run-with-gcs-logs](src:docs/docker_deployment.md#run-with-gcs-logs)); see [Docker deployment](../sources/docker_deployment.md).
- **Kubernetes** — worker + aggregator components for distributed profiling ([kubernetes-configuration](src:docs/kubernetes_deployment.md#kubernetes-configuration), [aggregator](src:docs/kubernetes_deployment.md#aggregator)); see [Kubernetes deployment](../sources/kubernetes_deployment.md).

## See also
- [Capturing profiles](capturing-profiles.md), [JAX GPU profiling](../sources/jax_gpu_profiling.md) (containerized capture)

