"""A local control panel: one web page with buttons for the commands you would otherwise type.

It adds no new powers. Every button runs the same `python -m engine.service ...` command (or the same pytest
check) you would run by hand, as a subprocess, and shows its output. Safety:

* listens on 127.0.0.1 only; requests must carry the right Host header (blocks DNS-rebinding) and a per-launch
  token that only the served page knows, sent as JSON (blocks cross-site form posts)
* secrets stay in the environment of the process that launched the panel; the browser never sees them
* arguments are validated against allowlists before anything runs; LIVE mode cannot be switched from here
* the page renders model-written text with textContent only, never as HTML
"""

from __future__ import annotations

import json
import os
import re
import secrets as token_source
import signal
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from collections.abc import Mapping
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from engine.clock import SystemClock
from engine.service.config import ServiceConfig, load_config

ROOT = Path(__file__).resolve().parents[3]  # the trading-engine folder
SYMBOL = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,14}$")
NUMBER = re.compile(r"^\d{1,12}(\.\d{1,8})?$")
IDENT = re.compile(r"^[A-Za-z0-9._:\-]{1,100}$")
COMMAND_TIMEOUT_S = 180

LIVE_CHECKS = {
    "check_alpaca": "live_paper",
    "check_news": "live_news",
    "check_claude": "live_anthropic",
    "check_grok": "live_xai",
}


class BadRequest(ValueError):
    pass


def build_argv(action: str, p: Mapping[str, Any], cfg_path: str, cfg: ServiceConfig) -> list[str]:
    """Turn a button press into a command line, or raise BadRequest. The only place arguments are accepted."""
    accounts = {a.id: a for a in cfg.accounts}
    base = [sys.executable, "-m", "engine.service"]

    def account() -> str:
        name = str(p.get("account", ""))
        if name not in accounts:
            raise BadRequest(f"unknown account {name!r}")
        return name

    def need(pattern: re.Pattern[str], key: str) -> str:
        value = str(p.get(key, "")).strip()
        if not pattern.match(value):
            raise BadRequest(f"invalid {key}")
        return value

    simple = {"migrate": ["migrate"], "status": ["status"], "proposals": ["proposals"]}
    if action in simple:
        return [*base, *simple[action], *(["--config", cfg_path] if action != "migrate" else [])]
    if action == "quotes":
        return [*base, "quotes", "--config", cfg_path, "--account", account()]
    if action == "kill":
        flag = ["--off"] if p.get("off") else []
        return [
            *base,
            "kill",
            "--config",
            cfg_path,
            "--account",
            account(),
            *flag,
            "--reason",
            "control panel",
        ]
    if action == "order":
        acct = accounts[account()]
        side = str(p.get("side", ""))
        if side not in ("BUY", "SELL"):
            raise BadRequest("side must be BUY or SELL")
        argv = [*base, "submit-intent", "--config", cfg_path, "--account", acct.id,
                "--symbol", need(SYMBOL, "symbol"), "--side", side, "--quantity", need(NUMBER, "quantity"),
                "--asset-class", "STOCK" if acct.broker == "alpaca" else "CRYPTO"]  # fmt: skip
        if side == "BUY":
            argv += ["--stop-loss", need(NUMBER, "stop_loss")]  # every entry needs a stop
        return argv
    if action == "show":
        return [
            *base,
            "show",
            "--config",
            cfg_path,
            "--account",
            account(),
            "--intent-id",
            need(IDENT, "intent_id"),
        ]
    if action in ("approve", "reject"):
        return [*base, action, "--config", cfg_path, "--id", need(IDENT, "id")]
    if action == "report":
        days, horizon = str(p.get("days", "30")), str(p.get("horizon", "24"))
        if not (days.isdigit() and horizon.isdigit() and 1 <= int(days) <= 365 and 1 <= int(horizon) <= 500):
            raise BadRequest("days must be 1-365 and horizon 1-500")
        return [*base, "llm-report", "--config", cfg_path, "--days", days, "--horizon-bars", horizon]
    if action in LIVE_CHECKS:
        return [sys.executable, "-m", "pytest", "-m", LIVE_CHECKS[action], "-o", "addopts=", "-q", "-rs"]
    raise BadRequest(f"unknown action {action!r}")


class ServiceProcess:
    """Starts and stops the execution service as a child process, logging to runs/service.log."""

    def __init__(self, cfg_path: str, root: Path = ROOT) -> None:
        self._cfg_path, self._root = cfg_path, root
        self._proc: subprocess.Popen[bytes] | None = None
        self.log_path = root / "runs" / "service.log"
        self._lock = threading.Lock()

    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> str:
        with self._lock:
            if self.running():
                return "already running"
            self.log_path.parent.mkdir(exist_ok=True)
            log = self.log_path.open("wb")  # each start begins a fresh log
            self._proc = subprocess.Popen(  # noqa: S603  (fixed argv, no user input)
                [sys.executable, "-m", "engine.service", "run", "--config", self._cfg_path],
                cwd=self._root, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )  # fmt: skip
            return f"started (pid {self._proc.pid}); output goes to {self.log_path}"

    def stop(self) -> str:
        with self._lock:
            if not self.running():
                return "not running (from this panel)"
            assert self._proc is not None
            self._proc.send_signal(signal.SIGTERM)  # the service stops cleanly and releases its leader lock
            try:
                self._proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            return "stopped"

    def tail(self, lines: int = 120) -> str:
        if not self.log_path.exists():
            return "(no log yet)"
        with self.log_path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 60_000))
            return "\n".join(f.read().decode(errors="replace").splitlines()[-lines:])


def _redact(text: str) -> str:
    for name, value in os.environ.items():
        if (
            value
            and len(value) >= 8
            and any(w in name for w in ("KEY", "SECRET", "TOKEN", "WEBHOOK", "DATABASE_URL"))
        ):
            text = text.replace(value, "***")
    return text


def run_command(argv: list[str], extra_env: Mapping[str, str] | None = None) -> dict[str, Any]:
    try:
        done = subprocess.run(  # noqa: S603  (argv comes from build_argv's allowlist, never a shell string)
            argv, cwd=ROOT, capture_output=True, text=True, timeout=COMMAND_TIMEOUT_S,
            env={**os.environ, **(extra_env or {})},
        )  # fmt: skip
    except subprocess.TimeoutExpired:
        return {"ok": False, "code": None, "output": f"timed out after {COMMAND_TIMEOUT_S}s"}
    out = (done.stdout + ("\n" + done.stderr if done.stderr.strip() else "")).strip()
    return {"ok": done.returncode == 0, "code": done.returncode, "output": _redact(out) or "(no output)"}


def health(port: int | None) -> dict[str, Any] | None:
    if not port:
        return None
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1) as r:  # noqa: S310
            return json.loads(r.read())  # type: ignore[no-any-return]
    except urllib.error.HTTPError as exc:  # 503 still carries the body
        try:
            return json.loads(exc.read())  # type: ignore[no-any-return]
        except ValueError:
            return None
    except (OSError, ValueError):
        return None


def pending_proposals() -> list[dict[str, Any]] | str:
    from engine.service.config import Secrets
    from engine.strategy.approvals import PostgresProposalStore

    try:
        store = PostgresProposalStore(Secrets.from_env(os.environ).database_url.get_secret_value())
    except Exception as exc:
        return f"database unavailable: {type(exc).__name__}"
    try:
        now = SystemClock().now()
        return [
            {
                "id": p.proposal_id, "account": p.account_id, "strategy": p.strategy_id,
                "side": str(p.intent.side), "quantity": str(p.intent.quantity), "symbol": p.intent.symbol,
                "stop": str(p.intent.stop_loss_price), "thesis": p.thesis, "news": list(p.cited),
                "minutes_left": max(0, int((p.expires_at - now) / timedelta(minutes=1))),
            }
            for p in store.pending(now)
        ]  # fmt: skip
    except Exception as exc:
        return f"could not read proposals: {type(exc).__name__}"
    finally:
        store.close()


class Panel:
    def __init__(self, cfg_path: str, port: int = 8765) -> None:
        self.cfg_path = str(Path(cfg_path).resolve())
        self.cfg = load_config(self.cfg_path)
        self.token = token_source.token_urlsafe(24)
        self.service = ServiceProcess(self.cfg_path)
        panel = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:
                return

            def _host_ok(self) -> bool:
                return self.headers.get("Host", "") in {f"127.0.0.1:{panel.port}", f"localhost:{panel.port}"}

            def _send(self, status: int, body: bytes, ctype: str = "application/json") -> None:
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Content-Security-Policy", "default-src 'self' 'unsafe-inline'")
                self.end_headers()
                self.wfile.write(body)

            def _json(self, status: int, payload: Any) -> None:
                self._send(status, json.dumps(payload).encode())

            def do_GET(self) -> None:
                if not self._host_ok():
                    return self._json(403, {"error": "bad host"})
                if self.path == "/":
                    return self._send(200, panel.page().encode(), "text/html; charset=utf-8")
                if self.headers.get("X-Panel-Token") != panel.token:
                    return self._json(403, {"error": "bad token"})
                if self.path == "/api/state":
                    return self._json(200, panel.state())
                if self.path == "/api/log":
                    return self._json(200, {"log": _redact(panel.service.tail())})
                self._json(404, {"error": "not found"})

            def do_POST(self) -> None:
                if not self._host_ok() or self.headers.get("X-Panel-Token") != panel.token:
                    return self._json(403, {"error": "forbidden"})
                if not self.headers.get("Content-Type", "").startswith("application/json"):
                    return self._json(415, {"error": "json only"})
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    body = json.loads(self.rfile.read(min(size, 10_000)) or b"{}")
                    result = panel.handle(self.path, body)
                except BadRequest as exc:
                    return self._json(400, {"ok": False, "output": f"refused: {exc}"})
                except (ValueError, TypeError):
                    return self._json(400, {"ok": False, "output": "malformed request"})
                self._json(200, result)

        self._server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.port = int(self._server.server_address[1])

    # ---- behaviour ----------------------------------------------------------------------------
    def handle(self, path: str, body: Mapping[str, Any]) -> dict[str, Any]:
        if path == "/api/service":
            op = body.get("op")
            if op == "start":
                if health(self.cfg.health_port) is not None and not self.service.running():
                    return {"ok": False, "output": "a service is already running outside this panel"}
                return {"ok": True, "output": self.service.start()}
            if op == "stop":
                return {"ok": True, "output": self.service.stop()}
            raise BadRequest("op must be start or stop")
        if path == "/api/run":
            action = str(body.get("action", ""))
            argv = build_argv(action, body.get("params") or {}, self.cfg_path, self.cfg)
            return run_command(argv, {"RUN_LIVE_TESTS": "1"} if action in LIVE_CHECKS else None)
        raise BadRequest("unknown endpoint")

    def state(self) -> dict[str, Any]:
        return {
            "service": {"running": self.service.running(), "health": health(self.cfg.health_port)},
            "accounts": [
                {"id": a.id, "broker": a.broker, "mode": str(a.mode), "enabled": a.enabled}
                for a in self.cfg.accounts
            ],  # fmt: skip
            "proposals": pending_proposals(),
            "llm": self.cfg.llm is not None,
        }

    def page(self) -> str:
        return PAGE.replace("__TOKEN__", self.token)

    def serve(self) -> None:
        print(f"Control panel: http://127.0.0.1:{self.port}   (Ctrl+C to quit)")
        try:
            self._server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            self._server.server_close()
            if self.service.running():
                print("Leaving the trading service running. Stop it from the panel or with `kill`.")


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Trading Engine Panel</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#16181d;--mute:#667085;--line:#e3e6eb;--accent:#2457d6;--ok:#16794c;--warn:#b54708;--bad:#b42318}
@media (prefers-color-scheme:dark){:root{--bg:#0f1115;--card:#171a21;--ink:#e8eaee;--mute:#9aa3b2;--line:#272c36;--accent:#7aa2ff;--ok:#5bd69a;--warn:#f5a04a;--bad:#ff7a70}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,sans-serif}
main{max-width:1000px;margin:0 auto;padding:20px 16px 60px}h1{font-size:20px;margin:0 0 4px}.sub{color:var(--mute);margin-bottom:18px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}
.card h2{font-size:14px;margin:0 0 10px;text-transform:uppercase;letter-spacing:.04em;color:var(--mute)}
.wide{grid-column:1/-1}
button{font:inherit;border:1px solid var(--line);background:var(--card);color:var(--ink);padding:8px 13px;border-radius:8px;cursor:pointer;margin:0 6px 6px 0}
button:hover{border-color:var(--accent)}button.p{background:var(--accent);color:#fff;border-color:var(--accent)}
button.d{color:var(--bad);border-color:var(--bad)}button:disabled{opacity:.5;cursor:wait}
input,select{font:inherit;padding:7px 9px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--ink);margin:0 6px 6px 0;max-width:100%}
label{display:block;font-size:12px;color:var(--mute);margin:0 0 3px}.row{display:flex;flex-wrap:wrap;align-items:flex-end}.row>div{margin-right:6px}
.pill{display:inline-block;padding:2px 10px;border-radius:99px;font-size:13px;border:1px solid var(--line)}
.ok{color:var(--ok);border-color:var(--ok)}.warn{color:var(--warn);border-color:var(--warn)}.bad{color:var(--bad);border-color:var(--bad)}
pre{margin:0;white-space:pre-wrap;word-break:break-word;font:13px/1.45 ui-monospace,Menlo,monospace;max-height:340px;overflow:auto;background:var(--bg);padding:12px;border-radius:8px;border:1px solid var(--line)}
.prop{border:1px solid var(--line);border-radius:10px;padding:12px;margin-bottom:10px}.prop b{font-size:16px}.muted{color:var(--mute);font-size:13px}
</style></head><body><main>
<h1>Trading Engine Panel</h1><div class="sub">Local only. Every button runs the same command you would type.</div>
<div class="grid">
<section class="card"><h2>Service</h2><p><span id="svc" class="pill">checking...</span> <span id="acct" class="muted"></span></p>
<button class="p" onclick="svc('start')">Start</button><button class="d" onclick="svc('stop')">Stop</button>
<button onclick="toggleLog()">Show log</button></section>

<section class="card"><h2>Look</h2>
<div class="row"><div><label>Account</label><select id="acc"></select></div></div>
<button onclick="run('status')">Status</button><button onclick="run('quotes',{account:v('acc')})">Quotes</button>
<button onclick="run('migrate')">Migrate DB</button></section>

<section class="card"><h2>Emergency</h2><p class="muted">Blocks new entries immediately. Exits and stops stay live.</p>
<button class="d" onclick="kill(false)">Kill switch ON</button><button onclick="kill(true)">Kill switch OFF</button></section>

<section class="card"><h2>Checks (real APIs, no real money)</h2>
<button onclick="run('check_alpaca')">Alpaca paper</button><button onclick="run('check_news')">News feed</button>
<button onclick="run('check_claude')">Claude</button><button onclick="run('check_grok')">Grok</button></section>

<section class="card wide"><h2>Manual order</h2><div class="row">
<div><label>Side</label><select id="side"><option>BUY</option><option>SELL</option></select></div>
<div><label>Symbol</label><input id="sym" value="AAPL" size="8"></div>
<div><label>Quantity</label><input id="qty" value="1" size="6"></div>
<div><label>Stop price (BUY)</label><input id="stop" placeholder="below market" size="10"></div>
<div><button class="p" onclick="order()">Queue order</button></div></div>
<div class="muted">Uses the account selected above. The service decides it within a couple of seconds; the risk engine can still refuse it.</div></section>

<section class="card wide"><h2>Model proposals <span id="pc" class="muted"></span></h2><div id="props" class="muted">loading...</div></section>

<section class="card wide"><h2>Reports</h2><div class="row">
<div><label>Days</label><input id="days" value="30" size="4"></div><div><label>Hold (bars)</label><input id="hz" value="24" size="4"></div>
<div><button onclick="run('report',{days:v('days'),horizon:v('hz')})">Model report</button></div>
<div><label>Intent id</label><input id="iid" size="22"></div>
<div><button onclick="run('show',{account:v('acc'),intent_id:v('iid')})">Show intent</button></div></div></section>

<section class="card wide" id="logcard" style="display:none"><h2>Service log</h2><pre id="log"></pre></section>
<section class="card wide"><h2>Output</h2><pre id="out">Nothing run yet.</pre></section>
</div></main>
<script>
const T="__TOKEN__",H={"Content-Type":"application/json","X-Panel-Token":T};
const $=id=>document.getElementById(id),v=id=>$(id).value.trim();let logOn=false;
async function post(path,body){const r=await fetch(path,{method:"POST",headers:H,body:JSON.stringify(body)});return r.json()}
function show(res){$("out").textContent=(res.ok===false?"FAILED\n":"")+(res.output||JSON.stringify(res));refresh()}
async function run(action,params){$("out").textContent="running "+action+"...";document.querySelectorAll("button").forEach(b=>b.disabled=true);
 try{show(await post("/api/run",{action,params:params||{}}))}catch(e){$("out").textContent="request failed: "+e}
 document.querySelectorAll("button").forEach(b=>b.disabled=false)}
async function svc(op){if(op==="stop"&&!confirm("Stop the trading service?"))return;show(await post("/api/service",{op}))}
function kill(off){const a=v("acc");if(!off&&!confirm("Turn the kill switch ON for "+a+"?"))return;run("kill",{account:a,off})}
function order(){const p={account:v("acc"),side:v("side"),symbol:v("sym").toUpperCase(),quantity:v("qty"),stop_loss:v("stop")};
 if(confirm(p.side+" "+p.quantity+" "+p.symbol+" on "+p.account+(p.stop_loss?" with stop "+p.stop_loss:"")+"?"))run("order",p)}
function decide(a,id){if(a==="reject"||confirm("Approve this trade? It still faces the risk engine."))run(a,{id})}
function toggleLog(){logOn=!logOn;$("logcard").style.display=logOn?"block":"none";refresh()}
function el(t,txt,cls){const e=document.createElement(t);if(txt!==undefined)e.textContent=txt;if(cls)e.className=cls;return e}
async function refresh(){try{
 const s=await (await fetch("/api/state",{headers:H})).json();const sv=s.service,h=sv.health;
 const pill=$("svc");let t="stopped",c="bad";
 if(h){t=h.ok?"running, healthy":"running, needs attention";c=h.ok?"ok":"warn"}else if(sv.running){t="starting...";c="warn"}
 pill.textContent=t;pill.className="pill "+c;
 const sel=$("acc"),cur=sel.value;if(!sel.options.length||sel.dataset.n!=s.accounts.length){sel.innerHTML="";
  s.accounts.forEach(a=>{const o=el("option",a.id+" ("+a.mode+(a.enabled?"":", disabled")+")");o.value=a.id;sel.appendChild(o)});sel.dataset.n=s.accounts.length;if(cur)sel.value=cur}
 const box=$("props");box.textContent="";
 if(typeof s.proposals==="string"){box.textContent=s.proposals;$("pc").textContent=""}
 else{$("pc").textContent="("+s.proposals.length+" waiting)";
  if(!s.proposals.length)box.textContent=s.llm?"None right now.":"No [llm] section in the config.";
  s.proposals.forEach(p=>{const d=el("div",undefined,"prop");d.appendChild(el("b",p.side+" "+p.quantity+" "+p.symbol+"  stop "+p.stop));
   d.appendChild(el("div",p.thesis));d.appendChild(el("div","news "+p.news.join(", ")+" · "+p.minutes_left+" min left · "+p.id,"muted"));
   const a=el("button","Approve","p");a.onclick=()=>decide("approve",p.id);const r=el("button","Reject");r.onclick=()=>decide("reject",p.id);
   d.appendChild(a);d.appendChild(r);box.appendChild(d)})}
 if(logOn){const l=await (await fetch("/api/log",{headers:H})).json();$("log").textContent=l.log}
}catch(e){$("svc").textContent="panel lost";$("svc").className="pill bad"}}
refresh();setInterval(refresh,5000);
</script></body></html>
"""


def run_panel(config: str, port: int = 8765) -> int:
    Panel(config, port).serve()
    return 0
