---
title: "Source: training/XPK_README.md"
type: source-page
doc: training/XPK_README.md
updated: 2026-07-03
---
# training/XPK_README.md

The shared cluster-bootstrap doc every non-Ironwood training and microbenchmark recipe in this repo
points to first. It pins the exact XPK version to use (`v0.14.3`, cloned into `~/xpk` since recipes
look for it there) and Python 3.12 as the primary supported interpreter for MaxText workloads
([initialization](src:training/XPK_README.md#initialization)). Cluster creation/deletion itself is
delegated to the public GCP "Create an XPK cluster with multi-NIC support" documentation — the note
is explicit that recipe users only need that one section, not the full XPK framework-setup docs
([gke-cluster-creation](src:training/XPK_README.md#gke-cluster-creation)).

## Feeds
- [MaxText training recipe pattern](../topics/maxtext-training-recipes.md) — every MaxText leaf
  recipe's "XPK setup" step is "follow this doc."
- [Ironwood TPU7x training recipe matrix](../topics/ironwood-training-recipes.md) — the XPK-launched
  half of the Ironwood recipes reuses this same cluster-creation step.
