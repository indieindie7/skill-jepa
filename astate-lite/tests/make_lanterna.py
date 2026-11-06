"""make_lanterna.py: builds the dogfood fixture, a fake project 'lanterna', one copy per scenario s1..s8.
Usage: py -3.13 -I make_lanterna.py <out_root>   -> <out_root>/s1 .. s8, <out_root>/truth.json (expected values).
Scenario seeds (what a session finds when it opens the copy):
  s1 no .state yet (open/work/close from scratch)      s2 = s1 after s1 ran (cold test reads STATE only)
  s3 no .state; README says 8080, config says 9090     s4 seeded STATE + 3 skill contracts (met / unmet / malformed)
  s5 seeded STATE; the answer is not in any source     s6 seeded STATE with a WRONG confirmed claim (8080)
  s7 300-line LOG, no STATE (regenerate under the cap) s8 seeded STATE (weekly metric line)
"""
import os, sys, json, shutil, random

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import astate_lite as al   # noqa: E402

D = "2026-11-05"
EM = al.EM
SOURCES = {
    "README.md": """# Lanterna

Lantern inventory service (Python 3.13, stdlib only).

Run it:

    py -3.13 src/app.py

The service listens on port 8080. Tests: `py -3.13 -m unittest discover tests`.
Logs go to var/log/lanterna.log. Deploy questions: see docs/DEPLOY.md.
""",
    "config/app.toml": """[server]
port = 9090
host = "127.0.0.1"

[log]
path = "var/log/lanterna.log"
""",
    "docs/DEPLOY.md": """# Deploy

Owner: Rui (ops). Run `scripts/deploy.ps1 -Env prod` from a clean checkout.
Secrets (vault token, DB password) come from the ops vault at deploy time; they are never in this repository.
""",
    "src/app.py": """import tomllib, http.server, pathlib

DEFAULT_PORT = 8080
CFG = pathlib.Path(__file__).resolve().parents[1] / "config" / "app.toml"


def port():
    cfg = tomllib.loads(CFG.read_text()) if CFG.exists() else {}
    return cfg.get("server", {}).get("port", DEFAULT_PORT)


if __name__ == "__main__":
    http.server.HTTPServer(("127.0.0.1", port()), http.server.SimpleHTTPRequestHandler).serve_forever()
""",
    "tests/test_app.py": """import unittest, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import app


class T(unittest.TestCase):
    def test_port(self):
        self.assertIsInstance(app.port(), int)
""",
    "scripts/deploy.ps1": """param([string]$Env = "staging")
Write-Host "deploying lanterna to $Env (token from vault)"
""",
    "CHANGELOG.md": """# Changelog
0.3 (2026-10-20): config/app.toml introduced.
0.2 (2026-09-12): log path moved to var/log/lanterna.log.
0.1 (2026-08-01): first cut.
""",
}
CONTRACTS = {
    "skills/log-reader/SKILL.md": """---
name: log-reader
description: Show the tail of the service log.
metadata:
  requires:
    - logs at var/log/lanterna.log
  reads: [var/log/lanterna.log]
  produces: [log excerpt]
  effects: [none]
  conflicts: []
---
Read the log path from STATE, print the last N lines.
""",
    "skills/deployer/SKILL.md": """---
name: deployer
description: Deploy lanterna to an environment.
metadata:
  requires:
    - deploy uses scripts/deploy.ps1
    - vault token available
  reads: [docs/DEPLOY.md]
  produces: [deployment]
  effects: [process execution, network]
  conflicts: [log-reader]
---
Run scripts/deploy.ps1 with the chosen environment.
""",
    "skills/port-checker/SKILL.md": """---
name: port-checker
description: Check which port the service listens on.
requires:
  - service listens on port 9090
reads: [config/app.toml]
produces: [port claim]
effects: [none]
conflicts: []
---
Front matter keys at top level, not under metadata (the loader rejects this).
""",
}
Q5 = ["How do I open Lanterna?", "Which port does the service listen on?", "Where are the logs?",
      "How do I run the tests?", "Who owns deploy?"]
EXPECT5 = [["src/app.py", "app.py"], ["9090", "8080"], ["lanterna.log"], ["unittest"], ["rui"]]


def L(date, typ, what, ev, st):
    return al.render_log(date, typ, what, ev, st)


SEED = {
    "s4": [L("2026-11-01", "event", "[SRC] logs at var/log/lanterna.log", "config/app.toml:6", "confirmed"),
           L("2026-11-01", "event", "[SRC] service listens on port 9090", "config/app.toml:2", "confirmed"),
           L("2026-11-02", "event", "[SYN] deploy uses scripts/deploy.ps1", "docs/DEPLOY.md:3", "hypothesis"),
           L("2026-11-02", "decision", "procedure: Open Lanterna: py -3.13 src/app.py", "README.md:7", "confirmed"),
           L("2026-11-02", "event", "question: How do I open Lanterna?", "STATE", "confirmed")],
    "s6": [L("2026-11-01", "event", "[SRC] service listens on port 8080", "README.md:9", "confirmed"),
           L("2026-11-01", "event", "[SRC] logs at var/log/lanterna.log", "README.md:10", "confirmed"),
           L("2026-11-02", "decision", "procedure: Open Lanterna: py -3.13 src/app.py", "README.md:7", "confirmed"),
           L("2026-11-02", "event", "question: How do I open Lanterna?", "STATE", "confirmed")],
}
SEED["s5"] = SEED["s4"][:2] + SEED["s4"][3:]
SEED["s8"] = SEED["s4"]


def write(root, rel, text):
    p = os.path.join(root, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def seed_state(root, lines):
    lp = os.path.join(root, ".state", "LOG.md")
    write(root, ".state/LOG.md", "# LOG " + EM + " lanterna\n(append-only; never edit old lines; one line per entry)\n\n")
    for ln in lines:
        al.append_log(lp, ln)
    entries, errors = al.read_log(lp)
    assert not errors, errors
    text, rep = al.regenerate(entries, "lanterna", "2026-11-02")
    write(root, ".state/STATE.md", text)
    return rep


def _s7_lines():
    """300 LOG lines over ~2 months: 40 facts promoted/demoted, 2 conflicts, 3 procedures, 2 blockers (1 closed),
    5 questions, 2 receipts, 1 metric, the rest re-confirmations. Deterministic (seed 7)."""
    rnd = random.Random(7)
    lines, day = [], 0

    def date():
        return f"2026-09-{1 + day % 28:02d}" if day < 28 else f"2026-10-{1 + (day - 28) % 28:02d}"

    for i in range(40):
        lines.append(L(date(), "event", f"[SYN] fact-{i} is v{i}", f"notes.md:{i + 1}", "hypothesis"))
        day += 1
    for q in Q5:
        lines.append(L(date(), "event", f"question: {q}", "STATE", "confirmed"))
    lines += [L(date(), "event", "[SRC] service listens on port 8080", "README.md:9", "confirmed"),
              L(date(), "event", "[SRC] service listens on port 9090", "config/app.toml:2", "confirmed"),
              L(date(), "decision", f"conflict: port 8080 vs 9090 {EM} README.md:9, config/app.toml:2 {EM} next: ask Rui", "README.md:9", "confirmed"),
              L(date(), "event", "[SRC] log path is var/log/lanterna.log", "config/app.toml:6", "confirmed"),
              L(date(), "event", "[SRC] log path is logs/app.log", "docs/old.md:4", "confirmed"),
              L(date(), "decision", f"conflict: log path var/log/lanterna.log vs logs/app.log {EM} config/app.toml:6, docs/old.md:4 {EM} next: check docs/old.md age", "config/app.toml:6", "confirmed"),
              L(date(), "decision", "procedure: Open Lanterna: py -3.13 src/app.py", "README.md:7", "confirmed"),
              L(date(), "decision", "procedure: Run tests: py -3.13 -m unittest discover tests", "README.md:9", "confirmed"),
              L(date(), "decision", "procedure: Deploy: scripts/deploy.ps1 -Env prod", "docs/DEPLOY.md:3", "confirmed"),
              L(date(), "blocker", "vault token missing for prod deploy", "Rui", "unknown"),
              L(date(), "blocker", "docs/old.md age unknown", "me", "unknown"),
              L(date(), "event", f"receipt: README.md {EM} sha256 {'ab' * 32} {EM} archived no", "README.md", "confirmed"),
              L(date(), "event", f"receipt: config/app.toml {EM} sha256 {'cd' * 32} {EM} archived no", "config/app.toml", "confirmed")]
    day += 3
    for i in range(40):                      # promote 30, demote 10 to unknown
        st = "confirmed" if i % 4 else "unknown"
        lines.append(L(date(), "event", f"[SRC] fact-{i} is v{i}", f"src/mod{i}.py:{i + 3}", st))
        day += 1
    lines.append(L(date(), "blocker", "vault token missing for prod deploy", "Rui", "confirmed"))   # closed
    lines.append(al.metric_line(date(), 1, 1500, 4))
    while len(lines) < 300:
        i = rnd.randrange(40)
        lines.append(L(date(), "event", f"[SRC] fact-{i} is v{i}", f"src/mod{i}.py:{i + 3}", "confirmed" if i % 4 else "unknown"))
        day += 1
    return lines[:300]


def _build_s7(out):
    s7 = os.path.join(out, "s7")
    lines = _s7_lines()
    write(s7, ".state/LOG.md", "# LOG " + EM + " lanterna\n(append-only; never edit old lines; one line per entry)\n\n" + "\n".join(lines) + "\n")
    entries, errors = al.read_log(os.path.join(s7, ".state", "LOG.md"))
    assert not errors and len(entries) == 300, (len(entries), errors[:3])
    ref, rep = al.regenerate(entries, "lanterna", D)
    parts = al.derive(entries)
    write(out, "s7_reference_STATE.md", ref)
    return {"ref_tokens": rep["tokens"], "ref_claims": rep["claims"], "ref_dropped": len(rep["dropped"]),
            "confirmed": [c["text"] for c in parts["claims"].values() if c["status"] == "confirmed"],
            "conflicts": list(parts["conflicts"].values()), "open_blockers": [b["text"] for b in parts["blockers"].values()],
            "closed_blocker": "vault token missing for prod deploy", "procedures": list(parts["procedures"])}


def _copy_sources(out, s):
    root = os.path.join(out, s)
    for rel, text in SOURCES.items():
        write(root, rel, text)
    return seed_state(root, SEED[s]) if s in SEED else None


def build(out):
    if os.path.isdir(out):
        shutil.rmtree(out)
    truth = {"questions": Q5, "expect": EXPECT5, "seeds": {}}
    for s in ("s1", "s3", "s4", "s5", "s6", "s7", "s8"):
        truth["seeds"][s] = _copy_sources(out, s)
    for rel, text in CONTRACTS.items():
        write(os.path.join(out, "s4"), rel, text)
    truth["s7"] = _build_s7(out)
    truth["s6_log_before"] = al.read_text(os.path.join(out, "s6", ".state", "LOG.md"))
    with open(os.path.join(out, "truth.json"), "w", encoding="utf-8") as f:
        json.dump(truth, f, indent=1, ensure_ascii=False)
    return truth


if __name__ == "__main__":
    t = build(sys.argv[1])
    print(json.dumps({k: v for k, v in t.items() if k in ("seeds", "s7")}, indent=1, ensure_ascii=False)[:1500])
