---
title: "Source: utils/ironwood/README.md"
type: source-page
doc: utils/ironwood/README.md
updated: 2026-07-03
---
# utils/ironwood/README.md

The hub for Ironwood (TPU v7) GKE cluster-lifecycle utility scripts, describing two alternative
implementation patterns for provisioning TPU v7 node pools: a **standard** node-pool-based approach
(interactive create/delete scripts) and a **CCC** (Cloud Compute Class) approach that defines
reusable TPU-topology templates GKE auto-provisions from on workload submission
([directory-structure](src:utils/ironwood/README.md#directory-structure)). Both share a common
cleanup script that tears down the full infrastructure (network, cluster, NAT, firewall) in one call
([core-scripts](src:utils/ironwood/README.md#core-scripts)).

## Feeds
- [Ironwood TPU7x training recipe matrix](../topics/ironwood-training-recipes.md) — the k8s-launched
  recipes in that topic use one of these two cluster-provisioning patterns.
