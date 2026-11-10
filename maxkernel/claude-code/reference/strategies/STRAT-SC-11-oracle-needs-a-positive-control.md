---
id: STRAT-SC-11
title: A correctness oracle that has never failed is not known to be able to fail
applies_to: any
lever: method
generality: general
phase: [implement, profile, test]
verify: "manual: inject the fault the oracle guards against and confirm it fails, per cell class"
---

# STRAT-SC-11 — Ship the oracle's positive control, not just its PASS

**Rule.** A green oracle is evidence only if you have shown that oracle can go red. **Inject the
fault you are guarding against and report which cells catch it**, per cell class. A gate that is
blind on the cell class matching production is not a gate, however many cells it passes — and the
blindness is invisible from the PASS. Before quoting any probe, also read where it imports its
reference from.

**Mechanism.** Three things make an oracle blind and none shows up in its output. A transform that
is an **involution** turns a double-application fault into the *original bits* rather than garbage.
A distribution can place every element in the region where the wrong computation and the right one
happen to agree — for an order-preserving float-to-int key trick, a strictly-positive input is that
region, and production distributions are the most likely to sit in it. And an oracle can have **no
independent reference**: if the reference is imported from the variant's own package, every
mutation moves both sides together and the check stays green non-vacuously.

**Scope.** General. The mandatory cell set it argues for: short rows, an in-range sentinel rather
than an out-of-range one, a cell class that straddles the discriminating boundary on *every* row
(check on device that it really straddles — a nominal range can miss on many rows), and one injected
fault per invariant the kernel claims. For float accuracy, probe `xla_reference_vs_f64_oracle`
checks whether the reference itself is trustworthy.

**Does not apply when** the oracle is an exact bit-comparison against an independent implementation
on the same inputs — but confirm the reference is genuinely independent, since a shared helper
reintroduces exactly this failure.

**How to check on your kernel.** For each invariant the kernel claims, write a mutant that breaks
it (double conversion, wrong sentinel, off-by-one). Run the oracle on every cell class and tabulate
caught/missed per class; any class that misses every mutant is blind and needs a new cell design.
Confirm the reference is imported from an independent module, not the variant under test.
