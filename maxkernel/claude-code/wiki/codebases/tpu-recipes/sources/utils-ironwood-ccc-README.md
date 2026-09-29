---
title: "Source: utils/ironwood/ccc/README.md"
type: source-page
doc: utils/ironwood/ccc/README.md
updated: 2026-07-03
---
# utils/ironwood/ccc/README.md

Detail on the CCC (ComputeClass) provisioning pattern: unlike a standard node pool, a ComputeClass
does **not** create nodes immediately — GKE uses it as a template and provisions the matching node
pool only when a workload targets it via `nodeSelector`
([what-is-computeclass-ccc](src:utils/ironwood/ccc/README.md#what-is-computeclass-ccc)). The
`clustersetup_gke_ccc_tpuv7.sh` script's three sections cover infrastructure deployment (VPC,
GKE cluster with multi-networking/DPv2, JobSet CRD), interactive CCC-template creation supporting
either DWS Flex Start or specific reservations (falling back to Flex Start if the reservation-based
node pool creation fails), and CCC template deletion
([scripts-overview](src:utils/ironwood/ccc/README.md#scripts-overview)). Workloads target a
provisioned template via `cloud.google.com/compute-class: tpuv7-<TOPOLOGY>-class`
([usage-in-workloads](src:utils/ironwood/ccc/README.md#usage-in-workloads)).

## Feeds
- [Ironwood TPU7x training recipe matrix](../topics/ironwood-training-recipes.md) — the CCC/
  autoscaling provisioning alternative to the static node-pool script for that topic's k8s recipes.
