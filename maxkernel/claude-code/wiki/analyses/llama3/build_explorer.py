#!/usr/bin/env python3
"""Build the interactive Llama3-8B MFU/TPS experiment explorer (jax + torchax lanes).

    python wiki/analyses/llama3/build_explorer.py        # re-extract + rebuild
    python wiki/analyses/llama3/build_explorer.py --from-json

Outputs (next to this script): mfu_data.json + mfu-explorer.html, plus per-lane
data docs in wiki/analyses/. Shared UI/doc code lives in ../explorer_lib.py.

Llama3 vs Qwen3 differences handled here: lanes are jax/torchax (not agent
forks),
experiments are numbered expNN (not vNN), four context lengths (1k/2k/4k/8k),
MaxText reference only at seq8192, and verdicts come from
verdict:/accepted-rejected.
"""

import glob, json, os, re, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))  # wiki/analyses/
import explorer_lib as lib

LANES = ["jax", "torchax"]
LANE_NAME = {"jax": "JAX (Flax NNX)", "torchax": "torchax"}
COLORS = {"jax": "#1f77b4", "torchax": "#ff7f0e"}
# MaxText llama3-8b reference exists only at seq8192. MFU is causal-adjusted to 39.6%
# (the tpu-recipes-v0.1.4 figure 44.6% is non-causal; current MaxText applies the causal ÷2).
MAXTEXT = {"mfu": {"8192": 39.6}, "tps": {"8192": 7069}}
VERIFIED = {
    "jax": {8192: 43.6},
    "torchax": {8192: 36.8},
}  # clamp 8k only (others trust extraction)
MT_CONSTS = {44.6, 44.5, 49.0}  # MaxText ref + its alt-accounting figure
MT_TPS = {7069}

mfu_pats = [
    re.compile(r"approx MFU\s*[:=]\s*([0-9]{1,2}\.[0-9])", re.I),
    re.compile(r"([0-9]{1,2}\.[0-9])\s*%\s*MFU", re.I),
    re.compile(r"MFU[:\s*]*\**\s*([0-9]{1,2}\.[0-9])\s*%", re.I),
    re.compile(r"MFU[^0-9%]{0,16}([0-9]{1,2}\.[0-9])\s*%", re.I),
]
tps_pat = re.compile(r"([0-9],?[0-9]{3})\s*tok/s/chip", re.I)
verdict_pat = re.compile(r"^verdict:\s*([^\n]+)", re.I | re.M)
status_pat = re.compile(r"^status:\s*([^\n]+)", re.I | re.M)


def norm_verdict(txt, bn):
  m = verdict_pat.search(txt) or status_pat.search(txt)
  if m:
    return lib.norm_verdict_from(m.group(1))
  return lib.norm_verdict_from(
      bn
  )  # fall back to the slug (…-accepted / …-rejected)


def vkey(fn):
  # order by date, then experiment number (baseline / non-exp sorts first within a date)
  date = fn[:10] if re.match(r"\d{4}-\d{2}-\d{2}", fn) else "9999-99-99"
  m = re.search(r"exp(\d{1,3})", fn)
  n = int(m.group(1)) if m else 0
  s = re.search(r"exp\d{1,3}([a-z]?)", fn)
  suf = s.group(1) if s else ""
  return (date, n, suf, fn)


def seqlen(fn, txt):
  f = fn.lower()
  for tok, val in [("8k", 8192), ("4k", 4096), ("2k", 2048), ("1k", 1024)]:
    if tok in f:
      return val
  for val in (8192, 4096, 2048, 1024):
    if str(val) in f:
      return val
  m = re.search(r"seq(?:len|_len)?\s*[=:]\s*(\d{3,4})", txt, re.I)
  if m and int(m.group(1)) in (1024, 2048, 4096, 8192):
    return int(m.group(1))
  cnt = {v: txt.count(str(v)) for v in (8192, 4096, 2048, 1024)}
  best = max(cnt, key=cnt.get)
  return best if cnt[best] > 0 else None


def extract():
  records = []
  for L in LANES:
    d = f"wiki/experiments/llama3_8B_autoresearch_optimization/{L}/experiments"
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
          if 5.0 <= float(m) <= 50.0 and float(m) not in MT_CONSTS
      ]
      mfu = lib.modal(cands)
      if mfu is not None and sl in VERIFIED.get(L, {}):
        mfu = min(mfu, VERIFIED[L][sl])
      tcands = [
          int(m.replace(",", ""))
          for m in tps_pat.findall(txt)
          if 1000 <= int(m.replace(",", "")) <= 12000
          and int(m.replace(",", "")) not in MT_TPS
      ]
      tps = max(tcands) if tcands else None
      vd = norm_verdict(txt, bn)
      records.append({
          "lane": L,
          "slug": bn,
          "vnum": vkey(bn)[1],
          "order": order,
          "seq": sl,
          "mfu": mfu,
          "tps": tps,
          "verdict": vd,
          "success": vd == "supported",
          "maxtext": ("maxtext" in bn.lower()) or ("maxtext" in txt.lower()),
          "has_metric": (mfu is not None) or (tps is not None),
          "link": (
              f"../../experiments/llama3_8B_autoresearch_optimization/{L}/experiments/{bn}.md"
          ),
      })
  return records


CFG = {
    "title": "Llama3-8B v6e-8",
    "subnote": (
        "Pick MFU or TPS · lane MFU is causal (÷2 attention) · line ="
        " running-best frontier · ○ 8k / △ 2k / ◇ 4k / □ 1k · dashed = MaxText"
        " 8k (MFU causal-adjusted to 39.6%; the tpu-recipes-v0.1.4 figure 44.6%"
        " is non-causal) · dotted = MaxText onset · click a point to open its"
        " experiment page · drag to zoom"
    ),
    "lanes": LANES,
    "lane_name": LANE_NAME,
    "colors": COLORS,
    "maxtext": MAXTEXT,
    "mtnote": {"mfu": " (causal-adj.)", "tps": ""},
    "seqs": [
        {"key": "1024", "label": "1k", "sym": "square"},
        {"key": "2048", "label": "2k", "sym": "triangle-up"},
        {"key": "4096", "label": "4k", "sym": "diamond"},
        {"key": "8192", "label": "8k", "sym": "circle"},
    ],
    "mfu_range": [8, 50],
    "tps_range": [1500, 9000],
    "doc": {
        "fname": lambda L: f"2026-06-16-llama3-8b-{L}-mfu-progression.md",
        "title": lambda L: f"Llama3-8B {L} — per-experiment MFU/TPS data",
        "tags": lambda L: f"llama3-8b, {L}, mfu, tps, data",
        "note": lambda L: (
            "MFU as reported (MaxText formula); the lane and MaxText use the"
            " same FLOP accounting."
        ),
        "intro": (
            "Frontiers (8k) clamped to the verified best. MaxText reference:"
            " **39.6% MFU (causal-adjusted; the tpu-recipes-v0.1.4 figure 44.6%"
            " is non-causal — current MaxText applies the causal ÷2)** / 7,069"
            " tok/s/chip @ seq8192 (no ref for 1k/2k/4k)."
        ),
        "regen": "python wiki/analyses/llama3/build_explorer.py",
        "seealso": lambda L: [
            "- [Interactive explorer](llama3/mfu-explorer.html)",
            (
                "- [Llama3-8B"
                " program](../experiments/llama3_8B_autoresearch_optimization/README.md)"
            ),
        ],
        "created": "2026-06-16",
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
  print("wrote mfu_data.json + mfu-explorer.html + 2 data docs")


if __name__ == "__main__":
  main()
