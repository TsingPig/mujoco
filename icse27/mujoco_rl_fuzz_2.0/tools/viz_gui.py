"""Local web GUI for launching the visualization CLIs.

Usage:
    python tools/viz_gui.py            # serves http://127.0.0.1:8765/
    python tools/viz_gui.py --port 9000 --no-open

Pure stdlib (http.server) — no Flask / pip needed. Each click POSTs to the
server, which spawns the matching `visualize_seed.py` / `visualize_mutator.py`
subprocess detached so the MuJoCo viewer window pops up locally.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mutations.registry import MUTATORS, MUTATOR_IDS  # noqa: E402

SEEDS_DIR = ROOT / "seeds" / "curated"
TOOLS_DIR = ROOT / "tools"
PYTHON = sys.executable


def list_seeds() -> list[str]:
    if not SEEDS_DIR.is_dir():
        return []
    return sorted(p.parent.name for p in SEEDS_DIR.glob("*/model.xml"))


def mutator_catalog() -> list[dict]:
    out = []
    for mid in MUTATOR_IDS:
        m = MUTATORS[mid]
        legal = [x for x in m.intensity_modes if x not in m.invalid_parseable_modes]
        invalid = list(m.invalid_parseable_modes)
        out.append({
            "id": mid,
            "runtime_only": bool(getattr(m, "runtime_only", False)),
            "legal": legal,
            "invalid": invalid,
        })
    return out


def spawn(cmd: list[str]) -> int:
    """Spawn a detached subprocess so the GUI request returns immediately."""
    print(f"[spawn] {' '.join(cmd)}")
    creationflags = 0
    if os.name == "nt":
        # CREATE_NEW_CONSOLE so the viewer window + its stdout get their own console.
        creationflags = subprocess.CREATE_NEW_CONSOLE  # type: ignore[attr-defined]
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        creationflags=creationflags,
        stdout=None, stderr=None,
    )
    return proc.pid


INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<title>mujoco_rl_fuzz_2.0 · 可视化入口</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: ui-sans-serif, system-ui, "Segoe UI", "Microsoft YaHei", sans-serif;
         margin: 0; padding: 24px; max-width: 1200px; }
  h1 { margin: 0 0 4px 0; }
  .sub { color: #888; margin-bottom: 24px; }
  .card { border: 1px solid #8884; border-radius: 10px; padding: 16px 18px;
          margin-bottom: 18px; background: #ffffff08; }
  .row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin: 6px 0; }
  select, button { font-size: 14px; padding: 6px 10px; border-radius: 6px;
                   border: 1px solid #8884; background: #ffffff10; color: inherit; }
  button { cursor: pointer; }
  button.primary { background: #2563eb; color: white; border-color: #2563eb; }
  button.primary:hover { background: #1d4ed8; }
  button.ghost { background: transparent; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
  .mut { border: 1px solid #8883; border-radius: 8px; padding: 10px 12px; }
  .mut h3 { margin: 0 0 6px 0; font-size: 14px; font-family: ui-monospace, Consolas, monospace; }
  .mut .tag { font-size: 11px; color: #888; margin-left: 6px; }
  .mut .modes { font-size: 12px; color: #888; margin-bottom: 6px; word-break: break-all; }
  #log { font-family: ui-monospace, Consolas, monospace; font-size: 12px;
         background: #0001; padding: 10px; border-radius: 8px; max-height: 220px;
         overflow: auto; white-space: pre-wrap; }
  .pill { display: inline-block; font-size: 11px; padding: 1px 6px; border-radius: 999px;
          background: #2563eb22; color: #2563eb; margin-left: 6px; }
  .pill.warn { background: #f59e0b22; color: #b45309; }
  .seed-item { display: flex; gap: 8px; align-items: center; padding: 6px 8px;
               border-radius: 6px; }
  .seed-item:hover { background: #8881; }
  .seed-name { font-family: ui-monospace, Consolas, monospace; font-size: 13px; flex: 1; }
</style>
</head>
<body>

<h1>mujoco_rl_fuzz_2.0 · 可视化入口</h1>
<div class="sub">点击按钮即在本机启动 <code>mujoco.viewer</code> 窗口。</div>

<div class="card">
  <h2 style="margin-top:0">① 种子（seed）</h2>
  <div class="sub" style="margin-bottom:8px">直接用 viewer 跑一个 curated 种子。</div>
  <div id="seeds"></div>
</div>

<div class="card">
  <h2 style="margin-top:0">② Mutator 前后对比</h2>
  <div class="row">
    <label>种子：</label>
    <select id="mut-seed"><option value="">(synthetic 合成种子)</option></select>
    <label style="margin-left:12px">
      <input type="checkbox" id="mut-no-before"/> 跳过 BEFORE
    </label>
  </div>
  <div class="grid" id="muts"></div>
</div>

<div class="card">
  <h2 style="margin-top:0">日志</h2>
  <div id="log">(等待操作)</div>
</div>

<script>
const SEEDS = __SEEDS__;
const MUTS  = __MUTS__;

const $log = document.getElementById("log");
function log(msg) {
  const t = new Date().toLocaleTimeString();
  $log.textContent += `\\n[${t}] ${msg}`;
  $log.scrollTop = $log.scrollHeight;
}

async function launch(payload) {
  log("launching " + JSON.stringify(payload));
  try {
    const r = await fetch("/launch", {
      method: "POST",
      headers: {"content-type": "application/json"},
      body: JSON.stringify(payload),
    });
    const j = await r.json();
    if (j.ok) log("→ pid=" + j.pid + "  cmd: " + j.cmd.join(" "));
    else log("✗ error: " + j.error);
  } catch (e) {
    log("✗ network error: " + e);
  }
}

// ------- seeds -------
const $seeds = document.getElementById("seeds");
const $mutSeed = document.getElementById("mut-seed");
SEEDS.forEach(name => {
  const row = document.createElement("div");
  row.className = "seed-item";
  row.innerHTML = `
    <span class="seed-name">${name}</span>
    <button class="primary">▶ 跑起来</button>
    <button class="ghost">◻ 静态</button>
  `;
  row.children[1].onclick = () => launch({tool:"seed", seed:name, static:false});
  row.children[2].onclick = () => launch({tool:"seed", seed:name, static:true});
  $seeds.appendChild(row);

  const opt = document.createElement("option");
  opt.value = name; opt.textContent = name;
  $mutSeed.appendChild(opt);
});
if (!SEEDS.length) $seeds.textContent = "(暂无 curated 种子；先跑 tools/fetch_seeds.py)";

// ------- mutators -------
const $muts = document.getElementById("muts");
MUTS.forEach(m => {
  const card = document.createElement("div");
  card.className = "mut";
  const tag = m.runtime_only ? '<span class="pill warn">runtime-only</span>' : '';
  const allModes = ["__all__", ...m.legal];
  const opts = allModes.map(x =>
    x === "__all__" ? `<option value="">(全部合法强度)</option>`
                    : `<option value="${x}">${x}</option>`).join("");
  card.innerHTML = `
    <h3>${m.id} ${tag}</h3>
    <div class="modes">legal: ${m.legal.join(", ") || "(none)"}<br/>
         ${m.invalid.length ? "invalid_parseable: " + m.invalid.join(", ") : ""}</div>
    <div class="row">
      <select>${opts}</select>
      <button class="primary">▶ Before / After</button>
    </div>
  `;
  const sel = card.querySelector("select");
  card.querySelector("button").onclick = () => launch({
    tool: "mutator",
    mutator: m.id,
    seed: $mutSeed.value || null,
    intensity: sel.value || null,
    no_before: document.getElementById("mut-no-before").checked,
  });
  $muts.appendChild(card);
});
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quiet default access log
        return

    def _send_json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urllib.parse.urlparse(self.path).path
        if path in ("/", "/index.html"):
            html = (INDEX_HTML
                    .replace("__SEEDS__", json.dumps(list_seeds()))
                    .replace("__MUTS__", json.dumps(mutator_catalog())))
            data = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if path == "/api/state":
            return self._send_json(200, {
                "seeds": list_seeds(),
                "mutators": mutator_catalog(),
            })
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        path = urllib.parse.urlparse(self.path).path
        if path != "/launch":
            self.send_error(404); return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            return self._send_json(400, {"ok": False, "error": f"bad json: {e}"})

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
            pid = spawn(cmd)
            return self._send_json(200, {"ok": True, "pid": pid, "cmd": cmd})
        except Exception as e:
            return self._send_json(400, {"ok": False, "error": str(e)})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-open", action="store_true", help="do not open the browser automatically")
    args = ap.parse_args()

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"serving on {url}  (Ctrl+C to stop)")
    if not args.no_open:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
