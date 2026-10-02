---
title: "Source: utils/ironwood/static/README.md"
type: source-page
doc: utils/ironwood/static/README.md
updated: 2026-07-03
---
# utils/ironwood/static/README.md

The standard (non-CCC) counterpart to the CCC utility scripts: `clustersetup_gke_tpuv7.sh`
provisions infrastructure and TPU v7 node pools directly (interactive create/delete, supporting DWS
Flex Start or specific reservations), and `clustercleanup_gke_tpuv7.sh` tears the whole environment
back down (cluster, workload policies, NAT/router, firewall, subnet/VPC) in one call
([scripts-overview](src:utils/ironwood/static/README.md#scripts-overview)). Workloads target a
provisioned node pool via `nodeSelector: {gke-tpu-topology: <TOPOLOGY>, gke-tpu-accelerator: tpu7x}`
([usage-in-workloads](src:utils/ironwood/static/README.md#usage-in-workloads)).

## Feeds
- [Ironwood TPU7x training recipe matrix](../topics/ironwood-training-recipes.md) — the static
  node-pool provisioning pattern behind that topic's k8s-launched recipes.
