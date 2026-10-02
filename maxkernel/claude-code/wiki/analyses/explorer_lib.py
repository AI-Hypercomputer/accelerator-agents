#!/usr/bin/env python3
"""Shared library for the interactive MFU/TPS experiment explorers.

Per-model wrappers (wiki/analyses/<model>/build_explorer.py) provide an
extract()
that returns a list of per-experiment records plus a cfg dict, then call
build_explorer() + write_docs() here. The HTML/JS template is seqlen-driven, so
models with 2 seqlens (qwen3: 2k/8k) or 4 (llama3: 1k/2k/4k/8k) share one
viewer.

Record schema (per experiment):
  lane, slug, vnum, order, seq (int|None), mfu (float|None), tps (int|None),
  verdict (str), success (bool), maxtext (bool), has_metric (bool), link (str)

cfg keys: title, subnote, lanes, lane_name, colors, seqs (list of
{key,label,sym}),
  maxtext ({mfu:{seqkey:val}, tps:{seqkey:val}} — partial ok), mfu_range,
  tps_range,
  doc (dict of doc-generation callables/strings — see write_docs).
"""

from collections import Counter
import json, os


def modal(cands):
  if not cands:
    return None
  cnt = Counter(cands)
  mx = cnt.most_common(1)[0][1]
  return max(m for m in cnt if cnt[m] == mx)


def norm_verdict_from(m):
  """Normalize a raw verdict/status string to a canonical verdict."""
  if not m:
    return ""
  s = m.strip().lower()
  if "supported" in s or "confirmed" in s or "accepted" in s:
    return "supported"
  if "invalid" in s or "failed" in s:
    return "invalid"  # invalid wins over refuted
  if "refuted" in s or "rejected" in s or "tie" in s:
    return "refuted"
  if "inconclusive" in s:
    return "inconclusive"
  if "potential" in s:
    return "potential"
  if "baseline" in s:
    return "baseline"
  if any(k in s for k in ("completed", "complete", "filed")):
    return "completed"
  if any(k in s for k in ("running", "pending", "open", "tbd")):
    return "pending"
  return ""


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__TITLE__ — experiment explorer</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root{--bg:#fafafa;--panel:#fff;--text:#222;--muted:#666;--label:#888;--border:#e5e5e5;--subt:#999}
  body.dark{--bg:#1b1b1b;--panel:#262626;--text:#e8e8e8;--muted:#9aa0a6;--label:#9aa0a6;--border:#3a3a3a;--subt:#888}
  body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;background:var(--bg);color:var(--text)}
  header{padding:14px 20px 6px}
  h1{font-size:18px;margin:0 0 2px} .sub{color:var(--muted);font-size:12px}
  #controls{display:flex;flex-wrap:wrap;gap:18px;padding:10px 20px;background:var(--panel);border-bottom:1px solid var(--border);align-items:center}
  .group{display:flex;gap:10px;align-items:center;font-size:13px}
  .group b{font-size:11px;color:var(--label);text-transform:uppercase;letter-spacing:.04em;margin-right:2px}
  label{display:inline-flex;gap:4px;align-items:center;cursor:pointer;user-select:none}
  .sw{width:11px;height:11px;border-radius:2px;display:inline-block}
  #chart{width:100%;height:74vh}
  .hint{font-size:11px;color:var(--subt);padding:4px 20px}
  #theme{margin-left:auto;cursor:pointer;background:var(--panel);color:var(--text);border:1px solid var(--border);border-radius:5px;padding:3px 11px;font-size:12px}
</style>
</head>
<body>
<header>
  <h1>__TITLE__ — MFU &amp; throughput per experiment</h1>
  <div class="sub">__SUBNOTE__</div>
</header>
<div id="controls"></div>
<div class="hint" id="count"></div>
<div id="chart"></div>
<script>
const DATA = __DATA__;
const COLORS = __COLORS__;
const MAXTEXT = __MAXTEXT__;
const LANE_NAME = __LANENAMES__;
const LANES = __LANES__;
const SEQS = __SEQS__;                                   // [{key,label,sym}]
const SEQKEYS = SEQS.map(s=>s.key);
const SYM = {"null":"square"}; SEQS.forEach(s=>SYM[s.key]=s.sym);
const SEQLABEL = {}; SEQS.forEach(s=>SEQLABEL[s.key]=s.label);
const MFURANGE = __MFURANGE__, TPSRANGE = __TPSRANGE__;
const MTNOTE = __MTNOTE__;   // per-metric suffix on the MaxText line label, e.g. {mfu:" (causal-adj.)"}
const MTMARK = {};   // lane -> order of the first experiment that engaged MaxText
for(const r of DATA){ if(r.maxtext && (!(r.lane in MTMARK) || r.order<MTMARK[r.lane])) MTMARK[r.lane]=r.order; }

const state = {
  lane: Object.fromEntries(LANES.map(L=>[L,true])),
  seq:  Object.fromEntries(SEQKEYS.map(k=>[k,true])),
  metric: "mfu",
  showUnsuccessful: true,
  showCrashed: false,
  showMtMark: true,
  dark: false,
};

// ---- controls ----
const C = document.getElementById("controls");
function chk(id, on, label, swatch){
  const l=document.createElement("label");
  const b=document.createElement("input"); b.type="checkbox"; b.checked=on; b.dataset.id=id;
  l.appendChild(b);
  if(swatch){const s=document.createElement("span"); s.className="sw"; s.style.background=swatch; l.appendChild(s);}
  l.appendChild(document.createTextNode(label));
  return {l,b};
}
function group(title){const g=document.createElement("div"); g.className="group"; const t=document.createElement("b"); t.textContent=title; g.appendChild(t); return g;}

const gLane=group("Variation");
LANES.forEach(L=>{const {l,b}=chk("lane:"+L,state.lane[L],LANE_NAME[L],COLORS[L]); b.onchange=()=>{state.lane[L]=b.checked;render();}; gLane.appendChild(l);});
C.appendChild(gLane);

const gSeq=group("Context");
SEQS.forEach(s=>{const {l,b}=chk("seq:"+s.key,state.seq[s.key],s.label); b.onchange=()=>{state.seq[s.key]=b.checked;render();}; gSeq.appendChild(l);});
C.appendChild(gSeq);

const gMetric=group("Metric");
[["mfu","MFU"],["tps","TPS"]].forEach(([k,lab])=>{
  const l=document.createElement("label");
  const b=document.createElement("input"); b.type="radio"; b.name="metric"; b.value=k; b.checked=(state.metric===k);
  b.onchange=()=>{ if(b.checked){ state.metric=k; render(); } };
  l.appendChild(b); l.appendChild(document.createTextNode(lab));
  gMetric.appendChild(l);
});
C.appendChild(gMetric);

const gShow=group("Show");
{const {l,b}=chk("unsucc",state.showUnsuccessful,"unsuccessful (refuted/invalid/inconcl.)"); b.onchange=()=>{state.showUnsuccessful=b.checked;render();}; gShow.appendChild(l);}
{const {l,b}=chk("crashed",state.showCrashed,"crashed / no-metric"); b.onchange=()=>{state.showCrashed=b.checked;render();}; gShow.appendChild(l);}
{const {l,b}=chk("mtmark",state.showMtMark,"MaxText onset (dotted)"); b.onchange=()=>{state.showMtMark=b.checked;render();}; gShow.appendChild(l);}
C.appendChild(gShow);

const themeBtn=document.createElement("button");
themeBtn.id="theme"; themeBtn.textContent="🌙 Dark";
themeBtn.onclick=()=>{ state.dark=!state.dark; document.body.classList.toggle("dark",state.dark);
  themeBtn.textContent=state.dark?"☀ Light":"🌙 Dark"; render(); };
C.appendChild(themeBtn);

// ---- filtering + trace building ----
const METRIC = {
  mfu: {field:"mfu", title:"MFU (%)", range:MFURANGE, floor:MFURANGE[0]+1,
        fmt:v=>v+"%", ulabel:"", mt:MAXTEXT.mfu, unit:"MFU"},
  tps: {field:"tps", title:"tok/s/chip", range:TPSRANGE, floor:TPSRANGE[0]+200,
        fmt:v=>Number(v).toLocaleString(), ulabel:" tok/s/chip", mt:MAXTEXT.tps, unit:"TPS"},
};
const FAIL = {refuted:1, invalid:1, inconclusive:1};   // the "unsuccessful" verdicts
function visible(d){
  if(!state.lane[d.lane]) return false;
  const sk = d.seq===null ? null : String(d.seq);
  if(sk!==null && !state.seq[sk]) return false;
  if(sk===null && !SEQKEYS.some(k=>state.seq[k])) return false;
  if(!d.has_metric) return state.showCrashed;
  if(FAIL[d.verdict] && !state.showUnsuccessful) return false;
  return true;
}
function hover(d, label){
  const seq = d.seq===null?"?":(SEQLABEL[String(d.seq)]||String(d.seq));
  const v = d.verdict||"—";
  return `<b>${d.slug}</b><br>${LANE_NAME[d.lane]} · seq ${seq} · ${v}`
       + `<br>${label}<br><i>click → open page</i>`;
}
function frontierLine(L, s, F){
  const rows = DATA.filter(d=>d.lane===L && d.seq===s && d[F]!==null && d.verdict!=="invalid")
                   .sort((a,b)=>a.order-b.order);
  let m=-1, xs=[], ys=[];
  for(const d of rows){ if(d[F]>m) m=d[F]; xs.push(d.order); ys.push(m); }
  return {xs,ys};
}
function buildTraces(){
  const M=METRIC[state.metric], F=M.field, traces=[];
  for(const L of LANES){
    if(!state.lane[L]) continue;
    const rows = DATA.filter(d=>d.lane===L && visible(d));
    const pts = rows.filter(d=>d[F]!==null);
    if(pts.length) traces.push({
      x:pts.map(d=>d.order), y:pts.map(d=>d[F]), customdata:pts.map(d=>d.link),
      text:pts.map(d=>hover(d, M.unit+": "+M.fmt(d[F])+M.ulabel)), hovertemplate:"%{text}<extra></extra>",
      mode:"markers", name:LANE_NAME[L], legendgroup:L, yaxis:"y",
      marker:{color:COLORS[L],size:7,symbol:pts.map(d=>SYM[String(d.seq)]),line:{width:0}},
    });
    for(const sq of SEQS){
      if(!state.seq[sq.key]) continue;
      const fr=frontierLine(L, Number(sq.key), F);
      if(fr.xs.length>1) traces.push({
        x:fr.xs, y:fr.ys, mode:"lines", yaxis:"y", legendgroup:L, showlegend:false,
        line:{color:COLORS[L],width:2.4,shape:"hv"}, hoverinfo:"skip",
      });
    }
    if(state.showCrashed){
      const cpts = rows.filter(d=>!d.has_metric);
      if(cpts.length) traces.push({
        x:cpts.map(d=>d.order), y:cpts.map(()=>M.floor), customdata:cpts.map(d=>d.link),
        text:cpts.map(d=>hover(d,"crashed / no metric")), hovertemplate:"%{text}<extra></extra>",
        mode:"markers", name:LANE_NAME[L]+" · crashed", legendgroup:L, yaxis:"y",
        marker:{color:COLORS[L],size:7,symbol:"x",opacity:0.5}, showlegend:false,
      });
    }
  }
  return traces;
}
function theme(){ return state.dark
  ? {paper:"#262626",plot:"#202020",font:"#e8e8e8",grid:"#333",mt:"#aaa",line:"#888"}
  : {paper:"#fff",plot:"#fff",font:"#222",grid:"#eee",mt:"#555",line:"#555"}; }
function shapes(){
  const M=METRIC[state.metric], T=theme(), s=[];
  for(const sq of SEQS){
    if(!state.seq[sq.key]) continue;
    const y=M.mt[sq.key]; if(y==null) continue;
    s.push({type:"line",xref:"paper",x0:0,x1:1,yref:"y",y0:y,y1:y,line:{color:T.line,width:1,dash:"dash"}});
  }
  if(state.showMtMark) for(const L of LANES){
    if(!state.lane[L] || !(L in MTMARK)) continue;
    s.push({type:"line",xref:"x",x0:MTMARK[L],x1:MTMARK[L],yref:"paper",y0:0,y1:1,
            line:{color:COLORS[L],width:1.5,dash:"dot"},opacity:0.6});
  }
  return s;
}
function annotations(){
  const M=METRIC[state.metric], T=theme(), a=[];
  for(const sq of SEQS){
    if(!state.seq[sq.key]) continue;
    const y=M.mt[sq.key]; if(y==null) continue;
    a.push({xref:"paper",x:0.005,yref:"y",y:y,yanchor:"bottom",showarrow:false,
            text:`MaxText ${sq.label} ${M.fmt(y)}${M.ulabel}${MTNOTE[state.metric]||""}`,font:{size:10,color:T.mt}});
  }
  return a;
}
function xMax(){
  let mx=10;
  for(const L of LANES){
    if(!state.lane[L]) continue;
    for(const d of DATA) if(d.lane===L && d.order>mx) mx=d.order;
  }
  return mx;
}
function render(){
  const M=METRIC[state.metric], T=theme();
  const traces=buildTraces();
  const layout={
    margin:{l:62,r:20,t:10,b:45}, hovermode:"closest", showlegend:true, font:{color:T.font},
    legend:{orientation:"h",y:-0.13,font:{size:11,color:T.font}},
    xaxis:{title:"experiment (in order, per lane)",range:[0,xMax()+3],gridcolor:T.grid,zerolinecolor:T.grid},
    yaxis:{title:M.title,range:M.range,gridcolor:T.grid,zerolinecolor:T.grid},
    shapes:shapes(), annotations:annotations(), plot_bgcolor:T.plot, paper_bgcolor:T.paper,
  };
  Plotly.react("chart",traces,layout,{responsive:true,displaylogo:false});
  const shown=DATA.filter(visible).length;
  document.getElementById("count").textContent=`${shown} of ${DATA.length} experiments shown`;
}
// When served from GitHub Pages (*.github.io), .md links render as raw text.
// Rewrite to github.com/<user>/<repo>/blob/main/... which renders the markdown.
// Off github.io (local file://, Obsidian) keep the resolved relative link.
function ghBlob(rel){
  const u=new URL(rel,location.href);
  if(!u.hostname.endsWith("github.io")) return u.href;
  const parts=u.pathname.replace(/^\//,"").split("/");
  const repo=parts.shift();
  const user=u.hostname.split(".")[0];
  return `https://github.com/${user}/${repo}/blob/main/${parts.join("/")}`;
}
document.getElementById("chart").on = null;
Plotly.newPlot("chart",[],{}).then(()=>{
  render();
  document.getElementById("chart").on("plotly_click",ev=>{
    const p=ev.points&&ev.points[0];
    if(p&&p.customdata) window.open(ghBlob(p.customdata),"_blank");
  });
});
</script>
</body>
</html>
"""


def build_explorer(script_dir, records, cfg):
  """Write mfu_data.json + mfu-explorer.html into script_dir."""
  html = (
      HTML.replace("__TITLE__", cfg["title"])
      .replace("__SUBNOTE__", cfg["subnote"])
      .replace("__DATA__", json.dumps(records))
      .replace("__COLORS__", json.dumps(cfg["colors"]))
      .replace("__MAXTEXT__", json.dumps(cfg["maxtext"]))
      .replace("__LANENAMES__", json.dumps(cfg["lane_name"]))
      .replace("__LANES__", json.dumps(cfg["lanes"]))
      .replace("__SEQS__", json.dumps(cfg["seqs"]))
      .replace("__MFURANGE__", json.dumps(cfg["mfu_range"]))
      .replace("__TPSRANGE__", json.dumps(cfg["tps_range"]))
      .replace("__MTNOTE__", json.dumps(cfg.get("mtnote", {})))
  )
  json.dump(
      records, open(os.path.join(script_dir, "mfu_data.json"), "w"), indent=0
  )
  open(os.path.join(script_dir, "mfu-explorer.html"), "w").write(html)


def write_docs(repo, records, cfg):
  """Write one per-lane data doc under wiki/analyses/, driven by cfg['doc']."""
  dd = cfg["doc"]
  seqs = cfg["seqs"]
  fnum = lambda v: "—" if v is None else f"{v:,}"
  fpct = lambda v: "—" if v is None else f"{v:.1f}%"
  for L in cfg["lanes"]:
    rows = [r for r in records if r["lane"] == L]
    fr = []
    for s in seqs:
      k = int(s["key"])
      bm = max(
          [r["mfu"] for r in rows if r["seq"] == k and r["mfu"] is not None],
          default=None,
      )
      bt = max(
          [r["tps"] for r in rows if r["seq"] == k and r["tps"] is not None],
          default=None,
      )
      if bm is not None or bt is not None:
        fr.append(f"{s['label']} = {fpct(bm)} MFU / {fnum(bt)} tok/s/chip")
    note = dd["note"](L) if callable(dd["note"]) else dd["note"]
    o = [
        "---",
        f'title: "{dd["title"](L)}"',
        "type: analysis",
        f"created: {dd['created']}",
        f"updated: {dd['updated']}",
        f"tags: [{dd['tags'](L)}]",
        "---",
        "",
        f"# {dd['title'](L)}",
        "",
        (
            f"Best-effort per-experiment extraction for the **{L}** lane, in"
            f" experiment order. {note} {dd['intro']} `—` = not recoverable"
            f" (crash / no metric / unparsed). Regenerate: `{dd['regen']}`."
        ),
        "",
        f"**Frontier:** " + " · ".join(fr) + ".",
        "",
        "| # | Experiment | seq | verdict | MFU | tok/s/chip |",
        "|---|---|---|---|---|---|",
    ]
    seqlab = {int(s["key"]): s["label"] for s in seqs}
    for i, r in enumerate(rows, 1):
      o.append(
          f"| {i} | {r['slug']} | {seqlab.get(r['seq'], '?')} |"
          f" {r['verdict'] or '—'} | {fpct(r['mfu'])} | {fnum(r['tps'])} |"
      )
    o += ["", "## See also", ""] + dd["seealso"](L) + [""]
    open(os.path.join(repo, "wiki", "analyses", dd["fname"](L)), "w").write(
        "\n".join(o)
    )


def summarize(records, lanes):
  print(
      f"extracted {len(records)} experiments"
      f" ({sum(1 for r in records if r['has_metric'])} with a metric)"
  )
  for L in lanes:
    rs = [r for r in records if r["lane"] == L]
    print(
        f"  {L}: {len(rs)} exp, {sum(1 for r in rs if r['mfu'] is not None)}"
        f" MFU, {sum(1 for r in rs if r['tps'] is not None)} TPS,"
        f" {sum(1 for r in rs if not r['has_metric'])} crashed"
    )
