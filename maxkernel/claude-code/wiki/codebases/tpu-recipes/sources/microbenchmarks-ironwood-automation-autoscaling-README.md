---
title: "Source: microbenchmarks/ironwood/automation/autoscaling/README.md"
type: source-page
doc: microbenchmarks/ironwood/automation/autoscaling/README.md
updated: 2026-07-03
---
# microbenchmarks/ironwood/automation/autoscaling/README.md

The autoscaling sibling of the static automation harness: instead of requiring node pools to
pre-exist, it uses GKE **CustomComputeClass (CCC)** to create and delete the required node pools
automatically as benchmark workloads are submitted
([ironwood-benchmark-automation-with-ccc-for-nodepool-creation](src:microbenchmarks/ironwood/automation/autoscaling/README.md#ironwood-benchmark-automation-with-ccc-for-nodepool-creation)).
The launch → monitor/retry → aggregate pipeline and Kueue-based job queuing are otherwise identical
to the static variant
([overview](src:microbenchmarks/ironwood/automation/autoscaling/README.md#overview)); the only added
prerequisite is verifying/creating the CCC templates for each TPU topology before the first run
([3-verify-requirments-for-ccc](src:microbenchmarks/ironwood/automation/autoscaling/README.md#3-verify-requirments-for-ccc)).

## Feeds
- [TPU microbenchmarks](../topics/tpu-microbenchmarks.md) — the autoscaling alternative to the
  static automation harness this topic covers.
