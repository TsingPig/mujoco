"""Local web GUI for seeds 2.0 — categorized & collapsible.

Same backend pattern as `viz_gui.py` (stdlib http.server, spawns
`visualize_seed.py` / `visualize_mutator.py` subprocesses), but the seed
table is grouped by the 12 bug-taxonomy categories defined in
`seeds2/categories.yaml`. Every category renders inside a `<details>` block
so the page is short by default. A toolbar lets you expand/collapse all.

Usage:
    python tools/viz_gui2.py            # serves http://127.0.0.1:9100/
    python tools/viz_gui2.py --port 9100 --no-open
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mutations.registry import MUTATORS, MUTATOR_IDS  # noqa: E402

SEEDS_DIR = ROOT / "seeds" / "curated"
SEEDS2_DIR = ROOT / "seeds2"
TOOLS_DIR = ROOT / "tools"
PYTHON = sys.executable

_CACHE: dict | None = None


def _load() -> dict:
    """Load (and cache) the seeds2 manifest + categories.yaml + per-cat memberships."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    cats_path = SEEDS2_DIR / "categories.yaml"
    manifest_path = SEEDS2_DIR / "MANIFEST.json"
    if not cats_path.is_file() or not manifest_path.is_file():
        print(f"[viz_gui2] missing seeds2 files; run "
              f"`python tools/build_seeds2.py` first", file=sys.stderr)
        cats: list = []
        manifest: list = []
    else:
        cats = yaml.safe_load(cats_path.read_text(encoding="utf-8")).get("categories", [])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_name = {m["name"]: m for m in manifest}
    grouped: dict[str, list[dict]] = {c["id"]: [] for c in cats}
    for m in manifest:
        for cid in m.get("categories", []):
            if cid in grouped:
                grouped[cid].append(m)
    # Sort each category by complexity heuristic, descending.
    def _complexity(m: dict) -> int:
        return ((m.get("nq") or 0) + (m.get("nv") or 0)
                + 2 * (m.get("nbody") or 0) + 2 * (m.get("ngeom") or 0)
                + 5 * (m.get("nu") or 0))
    for cid in grouped:
        grouped[cid].sort(key=lambda m: -_complexity(m))
    _CACHE = {"cats": cats, "manifest": manifest, "grouped": grouped,
              "by_name": by_name, "uncategorized": [m for m in manifest
                                                    if not m.get("categories")]}
    return _CACHE


def mutator_catalog() -> list[dict]:
    out = []
    for mid in MUTATOR_IDS:
        m = MUTATORS[mid]
        legal = [x for x in m.intensity_modes if x not in m.invalid_parseable_modes]
        invalid = list(m.invalid_parseable_modes)
        out.append({
            "id": mid, "runtime_only": bool(getattr(m, "runtime_only", False)),
            "legal": legal, "invalid": invalid,
        })
    return out


def spawn(cmd: list[str]) -> int:
    print(f"[spawn] {' '.join(cmd)}")
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_CONSOLE  # type: ignore[attr-defined]
    proc = subprocess.Popen(cmd, cwd=str(ROOT),
                            creationflags=creationflags,
                            stdout=None, stderr=None)
    return proc.pid


INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<title>seeds 2.0 · 按 bug 分类的可视化入口</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: ui-sans-serif, system-ui, "Segoe UI", "Microsoft YaHei", sans-serif;
         margin: 0; padding: 18px 24px; max-width: 1500px; }
  h1 { margin: 0 0 4px 0; }
  .sub { color: #888; margin-bottom: 14px; font-size: 13px; }
  .card { border: 1px solid #8884; border-radius: 10px; padding: 12px 16px;
          margin-bottom: 14px; background: #ffffff08; }
  details { border: 1px solid #8883; border-radius: 8px; padding: 6px 12px;
            margin-bottom: 8px; background: #ffffff04; }
  details > summary { cursor: pointer; font-weight: 600; font-size: 15px;
                      list-style: none; padding: 6px 0; user-select: none; }
  details > summary::before { content: '▸'; display: inline-block; width: 1.2em;
                              transition: transform .12s; }
  details[open] > summary::before { content: '▾'; }
  details > summary:hover { color: #2563eb; }
  .cat-id    { display: inline-block; min-width: 36px; padding: 1px 7px;
               border-radius: 6px; background: #2563eb; color: white;
               font-family: ui-monospace, Consolas, monospace; font-size: 12px;
               text-align: center; margin-right: 8px; }
  .cat-meta  { color: #888; font-weight: 400; font-size: 12px; margin-left: 8px; }
  .cat-desc  { color: #888; font-size: 12px; margin: 4px 0 8px 44px; }
  .ops       { font-size: 11px; color: #666; margin: 4px 0 8px 44px;
               font-family: ui-monospace, Consolas, monospace; }
  .ops b     { color: inherit; font-family: ui-sans-serif; }
  table.seeds { width: 100%; border-collapse: collapse; font-size: 12px;
                margin-top: 4px; }
  table.seeds th { text-align: left; font-weight: 500; color: #888;
                   border-bottom: 1px solid #8883; padding: 4px 6px;
                   cursor: pointer; user-select: none; }
  table.seeds td { padding: 4px 6px; border-bottom: 1px solid #8881;
                   vertical-align: middle; }
  table.seeds tr:hover td { background: #8881; }
  .seed-name { font-family: ui-monospace, Consolas, monospace; font-size: 11px; }
  .num { text-align: right; font-variant-numeric: tabular-nums;
         font-family: ui-monospace, Consolas, monospace; }
  button, select, input[type=text] {
    font-size: 12px; padding: 4px 8px; border-radius: 6px;
    border: 1px solid #8884; background: #ffffff10; color: inherit;
  }
  button { cursor: pointer; }
  button.primary { background: #2563eb; color: white; border-color: #2563eb; }
  button.primary:hover { background: #1d4ed8; }
  button.ghost { background: transparent; }
  .iconbtn { padding: 1px 6px; font-size: 11px; }
  .pill { display: inline-block; font-size: 10px; padding: 1px 6px;
          border-radius: 999px; margin-left: 4px; line-height: 1.5; }
  .pill.src    { background: #2563eb22; color: #2563eb; }
  .pill.tier   { background: #16a34a22; color: #16a34a; }
  .pill.cnt    { background: #f59e0b22; color: #b45309; font-variant-numeric: tabular-nums; }
  .toolbar { display: flex; gap: 8px; align-items: center; flex-wrap: wrap;
             margin-bottom: 8px; }
  .toolbar input[type=text] { min-width: 220px; }
  #log { font-family: ui-monospace, Consolas, monospace; font-size: 12px;
         background: #0001; padding: 8px; border-radius: 6px; max-height: 160px;
         overflow: auto; white-space: pre-wrap; }
  mark { background: #fde047; color: inherit; padding: 0 1px; border-radius: 2px; }
</style>
</head>
<body>

<h1>seeds 2.0 · 按 MuJoCo bug 分类的可视化入口</h1>
<div class="sub">
  分类规则定义于
  <code>seeds2/categories.yaml</code>；映射详见
  <code>_docs/mujoco_bug_taxonomy_seed_operator_oracle.md</code>。
  种子文件实际仍存放于 <code>seeds/curated/&lt;name&gt;/model.xml</code>，本页只做索引。
  默认全部分类折叠。
</div>

<div class="card">
  <div class="toolbar">
    <button id="expand-all" class="ghost">▾ 全部展开</button>
    <button id="collapse-all" class="ghost">▸ 全部折叠</button>
    <input type="text" id="filter" placeholder="过滤种子名 / 来源..."/>
    <span style="color:#888;font-size:12px">共 <b id="total-cnt">0</b> 类 · <b id="total-seeds">0</b> 种子</span>
    <button class="ghost" id="refresh">⟳ 刷新</button>
  </div>
</div>

<div id="cats"></div>

<div class="card">
  <details>
    <summary>未归类种子（uncategorized）<span class="pill cnt" id="uncat-cnt">0</span></summary>
    <div id="uncat-body" style="margin-top:8px"></div>
  </details>
</div>

<div class="card">
  <details>
    <summary>② Mutator 前后对比（与 viz_gui v1 一致）</summary>
    <div class="toolbar" style="margin-top:8px">
      <label>种子：</label>
      <input type="text" id="mut-seed" list="mut-seed-list" placeholder="输入名字或留空" style="min-width:340px"/>
      <datalist id="mut-seed-list"></datalist>
      <button class="ghost" id="mut-seed-clear">✕</button>
      <label><input type="checkbox" id="mut-no-before"/> 跳过 BEFORE</label>
      <input type="text" id="mut-filter" placeholder="过滤 mutator..." style="min-width:180px"/>
    </div>
    <div id="muts" style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px"></div>
  </details>
</div>

<div class="card">
  <h3 style="margin:0 0 6px 0;font-size:13px;color:#888">日志</h3>
  <div id="log">(等待操作)</div>
</div>

<script>
const STATE = __STATE__;          // {cats, grouped, manifest, uncategorized}
const MUTS = __MUTS__;
const $log = document.getElementById("log");
const $catsRoot = document.getElementById("cats");
const $filter = document.getElementById("filter");
const LS_KEY = "mjfuzz_viz2_state_v1";
const UI = Object.assign({open: {}, filter: ""}, (() => {
  try { return JSON.parse(localStorage.getItem(LS_KEY) || "{}"); }
  catch { return {}; }
})());
$filter.value = UI.filter || "";

function saveUI() {
  try { localStorage.setItem(LS_KEY, JSON.stringify(UI)); } catch {}
}
function log(m) {
  const t = new Date().toLocaleTimeString();
  $log.textContent += `\\n[${t}] ${m}`;
  $log.scrollTop = $log.scrollHeight;
}
async function launch(payload) {
  log("launching " + JSON.stringify(payload));
  try {
    const r = await fetch("/launch", {method:"POST",
      headers:{"content-type":"application/json"}, body: JSON.stringify(payload)});
    const j = await r.json();
    if (j.ok) log("→ pid=" + j.pid + "  " + j.cmd.join(" "));
    else log("✗ " + j.error);
  } catch (e) { log("✗ network: " + e); }
}
function escHtml(s) {
  return String(s).replace(/[&<>"']/g, c =>
    ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[c]);
}
function highlight(text, q) {
  const t = escHtml(text);
  if (!q) return t;
  const i = t.toLowerCase().indexOf(q.toLowerCase());
  if (i < 0) return t;
  return t.slice(0,i) + "<mark>" + t.slice(i,i+q.length) + "</mark>" + t.slice(i+q.length);
}

function buildSeedTable(seeds, q) {
  const filtered = q
    ? seeds.filter(s => s.name.toLowerCase().includes(q.toLowerCase())
                     || (s.source||"").toLowerCase().includes(q.toLowerCase()))
    : seeds;
  if (!filtered.length) {
    return '<div style="color:#888;font-size:12px;padding:6px 0">(过滤后无匹配)</div>';
  }
  const rows = filtered.map(s => {
    const xmlPath = s.xml_path || `seeds/curated/${s.name}/model.xml`;
    const tier = s.tier === "asset" ? '<span class="pill tier">asset</span>' : '';
    return `
      <tr>
        <td><span class="seed-name">${highlight(s.name, q)}</span> ${tier}</td>
        <td><span class="pill src">${escHtml(s.source||"?")}</span></td>
        <td class="num">${s.nq ?? "?"}</td>
        <td class="num">${s.nv ?? "?"}</td>
        <td class="num">${s.nbody ?? "?"}</td>
        <td class="num">${s.ngeom ?? "?"}</td>
        <td class="num">${s.nu ?? "?"}</td>
        <td>
          <button class="primary iconbtn" data-act="run" data-name="${escHtml(s.name)}" title="实时跑">▶</button>
          <button class="ghost iconbtn"   data-act="static" data-name="${escHtml(s.name)}" title="静态查看">◻</button>
          <button class="ghost iconbtn"   data-act="copy" data-path="${escHtml(xmlPath)}" title="复制 XML 路径">⧉</button>
          <button class="ghost iconbtn"   data-act="mut"  data-name="${escHtml(s.name)}" title="送入 mutator 区">M</button>
        </td>
      </tr>`;
  }).join("");
  return `<table class="seeds">
    <thead><tr>
      <th>name</th><th>source</th>
      <th class="num">nq</th><th class="num">nv</th>
      <th class="num">nbody</th><th class="num">ngeom</th><th class="num">nu</th>
      <th>动作</th>
    </tr></thead>
    <tbody>${rows}</tbody></table>`;
}

function bindRowActions(scope) {
  scope.querySelectorAll('button[data-act="run"]').forEach(b => {
    b.onclick = () => launch({tool:"seed", seed:b.dataset.name, static:false});
  });
  scope.querySelectorAll('button[data-act="static"]').forEach(b => {
    b.onclick = () => launch({tool:"seed", seed:b.dataset.name, static:true});
  });
  scope.querySelectorAll('button[data-act="copy"]').forEach(b => {
    b.onclick = async () => {
      try { await navigator.clipboard.writeText(b.dataset.path); log("已复制: " + b.dataset.path); }
      catch { log("复制失败: " + b.dataset.path); }
    };
  });
  scope.querySelectorAll('button[data-act="mut"]').forEach(b => {
    b.onclick = () => {
      document.getElementById("mut-seed").value = b.dataset.name;
      log("mutator 种子 → " + b.dataset.name);
    };
  });
}

function renderCats() {
  const q = $filter.value.trim();
  $catsRoot.innerHTML = "";
  let totalSeeds = 0;
  STATE.cats.forEach(c => {
    const seeds = (STATE.grouped[c.id] || []);
    totalSeeds += seeds.length;
    const isOpen = !!UI.open[c.id];
    const det = document.createElement("details");
    if (isOpen) det.setAttribute("open", "");
    det.dataset.cid = c.id;
    const ops = (c.operators || []).slice(0, 8).map(o => `<code>${escHtml(o)}</code>`).join(" · ");
    const oracles = (c.oracles || []).slice(0, 6).map(o => `<code>${escHtml(o)}</code>`).join(" · ");
    det.innerHTML = `
      <summary>
        <span class="cat-id">${c.id}</span>${escHtml(c.name_zh || "")}
        <span class="cat-meta">· ${seeds.length} seeds</span>
      </summary>
      <div class="cat-desc">${escHtml(c.description || "")}</div>
      <div class="ops"><b>operators:</b> ${ops || "(none)"}</div>
      <div class="ops"><b>oracles:</b> ${oracles || "(none)"}</div>
      <div class="seed-table"></div>
    `;
    det.querySelector(".seed-table").innerHTML = buildSeedTable(seeds, q);
    bindRowActions(det);
    det.addEventListener("toggle", () => {
      UI.open[c.id] = det.open; saveUI();
    });
    $catsRoot.appendChild(det);
  });
  document.getElementById("total-cnt").textContent = STATE.cats.length;
  document.getElementById("total-seeds").textContent = STATE.manifest.length;

  // Uncategorized.
  const $uncatBody = document.getElementById("uncat-body");
  $uncatBody.innerHTML = buildSeedTable(STATE.uncategorized, q);
  document.getElementById("uncat-cnt").textContent = STATE.uncategorized.length;
  bindRowActions($uncatBody);
}

document.getElementById("expand-all").onclick = () => {
  document.querySelectorAll("#cats details").forEach(d => {
    d.setAttribute("open", ""); UI.open[d.dataset.cid] = true;
  });
  saveUI();
};
document.getElementById("collapse-all").onclick = () => {
  document.querySelectorAll("#cats details").forEach(d => {
    d.removeAttribute("open"); UI.open[d.dataset.cid] = false;
  });
  saveUI();
};
$filter.oninput = () => { UI.filter = $filter.value; saveUI(); renderCats(); };
document.getElementById("refresh").onclick = async () => {
  const r = await fetch("/api/state?refresh=1"); const j = await r.json();
  STATE.cats = j.state.cats; STATE.grouped = j.state.grouped;
  STATE.manifest = j.state.manifest; STATE.uncategorized = j.state.uncategorized;
  renderCats(); rebuildMutDatalist(); log("reloaded.");
};

// ---- mutator section ----
const $muts = document.getElementById("muts");
const $mutFilter = document.getElementById("mut-filter");
const $mutSeed = document.getElementById("mut-seed");
function rebuildMutDatalist() {
  const list = document.getElementById("mut-seed-list");
  list.innerHTML = "";
  STATE.manifest.slice().sort((a,b) => a.name.localeCompare(b.name)).forEach(s => {
    const o = document.createElement("option");
    o.value = s.name; o.label = `[${s.source}] nq=${s.nq ?? '?'} nbody=${s.nbody ?? '?'}`;
    list.appendChild(o);
  });
}
function renderMuts() {
  const filt = $mutFilter.value.trim().toLowerCase();
  $muts.innerHTML = "";
  MUTS.forEach(m => {
    if (filt && !m.id.toLowerCase().includes(filt)) return;
    const card = document.createElement("div");
    card.style = "border:1px solid #8883;border-radius:6px;padding:6px 10px";
    const tag = m.runtime_only ? '<span class="pill cnt">runtime-only</span>' : '';
    const opts = ['<option value="">(全部合法强度)</option>',
                  ...m.legal.map(x => `<option value="${x}">${x}</option>`)].join("");
    card.innerHTML = `
      <div style="font-family:ui-monospace,Consolas,monospace;font-size:12px;font-weight:600">
        ${m.id} ${tag}
      </div>
      <div style="font-size:10px;color:#888;margin:2px 0 4px 0">
        legal: ${m.legal.join(", ") || "(none)"}
      </div>
      <div class="toolbar" style="margin:0">
        <select>${opts}</select>
        <button class="primary iconbtn">▶ Before/After</button>
      </div>
    `;
    const sel = card.querySelector("select");
    card.querySelector("button").onclick = () => launch({
      tool: "mutator", mutator: m.id,
      seed: $mutSeed.value || null,
      intensity: sel.value || null,
      no_before: document.getElementById("mut-no-before").checked,
    });
    $muts.appendChild(card);
  });
}
$mutFilter.oninput = renderMuts;
document.getElementById("mut-seed-clear").onclick = () => { $mutSeed.value = ""; };

renderCats();
rebuildMutDatalist();
renderMuts();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quiet
        return

    def _json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _state_payload(self, refresh: bool) -> dict:
        global _CACHE
        if refresh:
            _CACHE = None
        s = _load()
        return {
            "cats": [{k: v for k, v in c.items() if k in
                      ("id", "name_zh", "description", "doc_anchor",
                       "operators", "oracles", "sources")}
                     for c in s["cats"]],
            "grouped": s["grouped"],
            "manifest": s["manifest"],
            "uncategorized": s["uncategorized"],
        }

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)
        if parsed.path in ("/", "/index.html"):
            state = self._state_payload(refresh=False)
            html = (INDEX_HTML
                    .replace("__STATE__", json.dumps(state, ensure_ascii=False))
                    .replace("__MUTS__", json.dumps(mutator_catalog())))
            data = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if parsed.path == "/api/state":
            return self._json(200, {"state": self._state_payload(refresh="refresh" in qs)})
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        if urllib.parse.urlparse(self.path).path != "/launch":
            self.send_error(404); return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            return self._json(400, {"ok": False, "error": f"bad json: {e}"})
        tool = payload.get("tool")
        try:
            if tool == "seed":
                seed = payload.get("seed")
                if not seed:
                    raise ValueError("seed required")
                cmd = [PYTHON, str(TOOLS_DIR / "visualize_seed.py"), seed]
                if payload.get("static"):
                    cmd.append("--static")
            elif tool == "mutator":
                mid = payload.get("mutator")
                if not mid:
                    raise ValueError("mutator required")
                cmd = [PYTHON, str(TOOLS_DIR / "visualize_mutator.py"), mid]
                if payload.get("seed"):
                    cmd += ["--seed", payload["seed"]]
                if payload.get("intensity"):
                    cmd += ["--intensity", payload["intensity"]]
                if payload.get("no_before"):
                    cmd.append("--no-before")
            else:
                raise ValueError(f"unknown tool: {tool!r}")
            return self._json(200, {"ok": True, "pid": spawn(cmd), "cmd": cmd})
        except Exception as e:
            return self._json(400, {"ok": False, "error": str(e)})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=9100)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"[viz_gui2] serving {url}  (Ctrl+C to stop)")
    if not args.no_open:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[viz_gui2] bye.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
