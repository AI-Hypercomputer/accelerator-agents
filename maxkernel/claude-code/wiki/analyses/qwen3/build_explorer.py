#!/usr/bin/env python3
"""Build the interactive Qwen3 MFU/TPS experiment explorer (4 agent lanes).

    python wiki/analyses/qwen3/build_explorer.py        # re-extract + rebuild
    python wiki/analyses/qwen3/build_explorer.py --from-json   # rebuild HTML
    only

Outputs (next to this script): mfu_data.json + mfu-explorer.html, plus the four
per-lane data docs in wiki/analyses/. Shared UI/doc code lives in
../explorer_lib.py.
"""

import glob, json, os, re, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))  # wiki/analyses/
import explorer_lib as lib

LANES = ["cc", "ag", "cx", "cc5"]
LANE_NAME = {
    "cc": "cc (Claude)",
    "ag": "ag (Antigravity)",
    "cx": "cx (Codex)",
    "cc5": "cc5 (Fable5)",
}
COLORS = {"cc": "#1f77b4", "ag": "#ff7f0e", "cx": "#2ca02c", "cc5": "#d62728"}
# MaxText MFU causal-adjusted (the tpu-recipes-v0.1.4 figures 38.0/45.3% are non-causal;
# current MaxText applies the causal ÷2 — same finding as llama3). TPS is convention-free.
MAXTEXT = {
    "mfu": {"2048": 36.6, "8192": 39.8},
    "tps": {"2048": 7116, "8192": 6942},
}
CX_ADJ = (
    {}
)  # cx reports causal MFU directly; the VERIFIED clamp drops its non-causal figure
VERIFIED = {
    "cc": {2048: 35.8, 8192: 34.8},
    "ag": {2048: 33.0, 8192: 30.6},
    "cx": {2048: 47.3, 8192: 43.2},
    "cc5": {2048: 40.5, 8192: 39.9},
}
MT_CONSTS = {45.3, 45.0, 44.9, 44.6, 38.0, 37.7, 36.8}
MT_TPS = {6942, 7116, 6953, 6883, 6601}

mfu_pats = [
    re.compile(r"approx MFU\s*[:=]\s*([0-9]{1,2}\.[0-9])", re.I),
    re.compile(r"([0-9]{1,2}\.[0-9])\s*%\s*MFU", re.I),
    re.compile(r"MFU[:\s*]*\**\s*([0-9]{1,2}\.[0-9])\s*%", re.I),
    re.compile(r"MFU[^0-9%]{0,14}([0-9]{1,2}\.[0-9])\s*%", re.I),
]
tps_pat = re.compile(r"([0-9],?[0-9]{3})\s*tok/s/chip", re.I)
verdict_pat = re.compile(r"^verdict:\s*([^\n]+)", re.I | re.M)
status_pat = re.compile(r"^status:\s*([^\n]+)", re.I | re.M)


def norm_verdict(txt):
  m = verdict_pat.search(txt) or status_pat.search(txt)
  return lib.norm_verdict_from(m.group(1)) if m else ""


def vkey(fn):
  m = re.search(r"v(\d{1,3})", fn)
  n = int(m.group(1)) if m else 999
  s = re.search(r"v\d{1,3}([a-z]?)", fn)
  return (n, s.group(1) if s else "", fn)


def seqlen(fn, txt):
  f = fn.lower()
  if any(
      k in f for k in ["8192", "s8k", "seq8k", "-8k", "8k-", "8kbase", "8kbs"]
  ):
    return 8192
  if any(
      k in f
      for k in [
          "2048",
          "s2k",
          "seq2k",
          "-2k",
          "2k-",
          "2kbase",
          "2kbs",
          "2kctl",
          "2kunroll",
          "2kscan",
          "2kio",
          "2kce",
          "2kbatch",
          "batch1",
          "batch2",
      ]
  ):
    return 2048
  if "2k" in f:
    return 2048
  if "8k" in f:
    return 8192
  c8, c2 = txt.count("8192"), txt.count("2048")
  return 8192 if (c8 > c2 and c8 > 0) else (2048 if c2 > 0 else None)


def extract():
  records = []
  for L in LANES:
    d = f"wiki/experiments/qwen3_{L}_autoresearch_optimization/jax/experiments"
    files = sorted(
        glob.glob(os.path.join(d, "*.md")),
        key=lambda f: vkey(os.path.basename(f)),
    )
    order = 0
    for f in files:
      txt = open(f, encoding="utf-8", errors="ignore").read()
      bn = os.path.basename(f)[:-3]
      order += 1
      sl = seqlen(bn, txt)
      cands = [
          float(m)
          for p in mfu_pats
          for m in p.findall(txt)
          if 5.0 <= float(m) <= 52.0 and float(m) not in MT_CONSTS
      ]
      mfu = max(cands) if (L == "cx" and cands) else lib.modal(cands)
      adj = mfu
      if mfu is not None and L == "cx" and sl in CX_ADJ:
        adj = round(mfu / CX_ADJ[sl], 1)
      if adj is not None and sl in VERIFIED.get(L, {}):
        adj = min(adj, VERIFIED[L][sl])
      tcands = [
          int(m.replace(",", ""))
          for m in tps_pat.findall(txt)
          if 1000 <= int(m.replace(",", "")) <= 15000
          and int(m.replace(",", "")) not in MT_TPS
      ]
      tps = max(tcands) if tcands else None
      vd = norm_verdict(txt)
      records.append({
          "lane": L,
          "slug": bn,
          "vnum": vkey(bn)[0],
          "order": order,
          "seq": sl,
          "mfu": adj,
          "tps": tps,
          "verdict": vd,
          "success": vd == "supported",
          "maxtext": ("maxtext" in bn.lower()) or ("maxtext" in txt.lower()),
          "has_metric": (adj is not None) or (tps is not None),
          "link": (
              f"../../experiments/qwen3_{L}_autoresearch_optimization/jax/experiments/{bn}.md"
          ),
      })
  return records


CFG = {
    "title": "Qwen3-8B v6e-8",
    "subnote": (
        "Pick MFU or TPS · all MFU on the causal basis (÷2 attention) · line ="
        " running-best frontier · ○ 8k / △ 2k · dashed = MaxText (MFU"
        " causal-adjusted: 8k 39.8% / 2k 36.6%; the tpu-recipes-v0.1.4 figures"
        " 45.3/38.0% are non-causal) · dotted = MaxText onset · click a point"
        " to open its experiment page · drag to zoom"
    ),
    "lanes": LANES,
    "lane_name": LANE_NAME,
    "colors": COLORS,
    "maxtext": MAXTEXT,
    "mtnote": {"mfu": " (causal-adj.)", "tps": ""},
    "seqs": [
        {"key": "2048", "label": "2k", "sym": "triangle-up"},
        {"key": "8192", "label": "8k", "sym": "circle"},
    ],
    "mfu_range": [8, 52],
    "tps_range": [1500, 10000],
    "doc": {
        "fname": lambda L: f"2026-06-15-qwen3-{L}-jax-mfu-progression.md",
        "title": lambda L: f"Qwen3-{L} jax — per-experiment MFU/TPS data",
        "tags": lambda L: f"qwen3-{L}, jax, mfu, tps, data",
        "note": lambda L: (
            "**cx** reports MFU on the causal basis directly — **no ÷"
            ' adjustment**. Its inflated "MaxText-style" (non-causal) figure'
            " (e.g. 49.2% @8k) is discarded in favour of the causal value"
            " (verified: 2k 47.3% / 8k 43.2%)."
            if L == "cx"
            else "MFU is on the causal (MaxText-comparable) basis."
        ),
        "intro": (
            "Frontiers are clamped to the verified model-page best. MaxText MFU"
            " targets **causal-adjusted**: 2k = 36.6%, 8k = 39.8% (the"
            " tpu-recipes-v0.1.4 figures 38.0/45.3% are non-causal; the lanes"
            " count attention causally). TPS is convention-free."
        ),
        "regen": "python wiki/analyses/qwen3/build_explorer.py",
        "seealso": lambda L: [
            (
                "- [Interactive explorer](qwen3/mfu-explorer.html) ·"
                " [progression plot](qwen3/mfu-progression.png)"
            ),
            f"- [Qwen3 {L} — jax model page](../models/qwen3-{L}-jax.md)",
        ],
        "created": "2026-06-15",
        "updated": "2026-06-16",
    },
}


def main():
  os.chdir(REPO)
  if "--from-json" in sys.argv:
    records = json.load(open(os.path.join(SCRIPT_DIR, "mfu_data.json")))
  else:
    records = extract()
  lib.build_explorer(SCRIPT_DIR, records, CFG)
  lib.write_docs(REPO, records, CFG)
  lib.summarize(records, LANES)
  print("wrote mfu_data.json + mfu-explorer.html + 4 data docs")


if __name__ == "__main__":
  main()
