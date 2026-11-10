"""tripwire.py (2026-11-10): Tripwire notes, rollout step 1 (single agent, scripts only).
Source: the user's print "ASTATE-lite + Tripwire Notes" (9 November 2026), text in skills/astate-lite/tree/tripwire-print.md.

A tripwire note = a steer the user gave mid-task, stored with triggers (path, symbol, keyword).
  pre   (PreToolUse, edit tools): a note whose trigger matches the edit is injected once, before the edit.
  post  (PostToolUse, edit tools): the touched file is appended to a log; that log decides "already touched".
  close (Stop): blocks "done" while any note is neither applied nor waived (second stop always passes: no loop).
Inert for a session with no notes file: every mode exits 0 without output.

CLI (session = --session <id>, else env CLAUDE_CODE_SESSION_ID):
  py -3.13 -I tripwire.py add "<note>" [--path p ...] [--symbol s ...] [--keyword k ...] [--block]
      Live 11-10: the injected note reaches the model WITH the edit's result (the edit as composed has already
      landed), so a plain note means a fix-up edit. --block refuses that first matching edit instead (must-not notes).
  py -3.13 -I tripwire.py applied <id> [--fact] [--pointer <file:line>] [--state <dir with LOG.md>]
  py -3.13 -I tripwire.py waive <id> "<reason>"
  py -3.13 -I tripwire.py list | stats
Store: ~/.claude/compact-captures/<session>/tripwire.json, tripwire_touched.log, tripwire.log (JSON lines).
`applied` appends a v2.2 LOG line to the project's .state/LOG.md when one exists (--state, else found upward from
the cwd): a `decision` line, or with --fact a `finding ... | hypothesis` line (a durable fact becomes a STATE
hypothesis claim at the next checkpoint)."""
import datetime as dt
import fnmatch
import json
import os
import re
import sys
from pathlib import Path

CAPTURES = Path.home() / ".claude" / "compact-captures"
EDIT_TOOLS = ("Edit", "Write", "NotebookEdit", "MultiEdit")
TEXT_KEYS = ("old_string", "new_string", "content", "new_source")


def now():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M")


def sdir(session):
    return CAPTURES / re.sub(r"[^\w-]", "", str(session or "unknown"))


def load(session):
    p = sdir(session) / "tripwire.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def save(session, data):
    d = sdir(session)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f"tripwire.json.{os.getpid()}.tmp"          # own tmp per process: parallel hooks never share one
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, d / "tripwire.json")


def log(session, event, **kw):
    with open(sdir(session) / "tripwire.log", "a", encoding="utf-8") as f:
        f.write(json.dumps({"t": now(), "event": event, **kw}, ensure_ascii=False) + "\n")


def norm_path(p):
    return str(p or "").replace("\\", "/").lower()


def touched(session):
    p = sdir(session) / "tripwire_touched.log"
    if not p.is_file():
        return []
    return list(dict.fromkeys(ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()))


# ---------------------------------------------------------------- matching
def path_hit(trigger, path):
    """A path trigger matches a file when it is a glob for it, its tail, or a folder/fragment inside it."""
    t, p = norm_path(trigger), norm_path(path)
    if not t or not p:
        return False
    if any(ch in t for ch in "*?["):
        return fnmatch.fnmatch(p, t) or fnmatch.fnmatch(p, "*/" + t)
    return p == t or p.endswith("/" + t) or ("/" + t.strip("/") + "/") in ("/" + p)


def symbol_hit(symbol, text):
    return bool(symbol) and bool(re.search(r"(?<![\w])" + re.escape(symbol) + r"(?![\w])", text))


def edit_of(hook):
    """(file path, edited text) of an edit-tool call; (None, '') for anything else."""
    if hook.get("tool_name") not in EDIT_TOOLS:
        return None, ""
    inp = hook.get("tool_input") or {}
    parts = [str(inp.get(k) or "") for k in TEXT_KEYS]
    parts += [str(e.get("old_string", "")) + "\n" + str(e.get("new_string", "")) for e in inp.get("edits") or [] if isinstance(e, dict)]
    return inp.get("file_path") or inp.get("notebook_path"), "\n".join(parts)


def why_match(note, path, text):
    """The trigger that fires for this edit, or None. Path triggers look at the file, symbols at the edited text
    (whole word, case kept), keywords at file + text (substring, case ignored)."""
    for t in note.get("paths", []):
        if path_hit(t, path):
            return f"path {t}"
    for s in note.get("symbols", []):
        if symbol_hit(s, text):
            return f"symbol {s}"
    hay = (norm_path(path) + "\n" + text).lower()
    for k in note.get("keywords", []):
        if k and k.lower() in hay:
            return f"keyword {k}"
    return None


def open_notes(data):
    return [n for n in data["notes"] if n["status"] == "open"]


def fresh_hits(data, path, text):
    """[(note, trigger)] for open notes not yet injected that match this edit; marks them injected."""
    hits = []
    for n in open_notes(data):
        why = None if n.get("injected") else why_match(n, path, text)
        if why:
            n["injected"] = now()
            hits.append((n, why))
    return hits


# ---------------------------------------------------------------- hook modes
def mode_pre(hook):
    session = hook.get("session_id")
    data = load(session)
    path, text = edit_of(hook)
    if not data or not path:
        return
    hits = fresh_hits(data, path, text)
    if not hits:
        return
    save(session, data)
    for n, why in hits:                                  # logged only after the save: a failed save counts nothing
        log(session, "inject", id=n["id"], why=why, file=norm_path(path))
    me = Path(__file__).as_posix()
    lines = ["[tripwire] The user's note(s) for what you are about to edit. Apply them in this edit, then mark each:"]
    lines += [f"- note {n['id']} ({why}): {n['text']}" for n, why in hits]
    lines.append(f'Mark: py -3.13 -I "{me}" applied <id>   |   does not apply here: py -3.13 -I "{me}" waive <id> "<reason>"')
    out = {"hookEventName": "PreToolUse", "additionalContext": "\n".join(lines)}
    if any(n.get("block") for n, _ in hits):             # --block note: refuse this one edit so it is redone with the note
        out.update(permissionDecision="deny", permissionDecisionReason="\n".join(lines))
    print(json.dumps({"hookSpecificOutput": out}))


def mode_post(hook):
    session = hook.get("session_id")
    path, _ = edit_of(hook)
    if not path:
        return
    sdir(session).mkdir(parents=True, exist_ok=True)     # logged in every session: the dispatcher reads it as "mid-task"
    with open(sdir(session) / "tripwire_touched.log", "a", encoding="utf-8") as f:
        f.write(norm_path(path) + "\n")


def mode_close(hook):
    session = hook.get("session_id")
    data = load(session)
    todo = open_notes(data) if data else []
    if not todo:
        return
    ids = [n["id"] for n in todo]
    if hook.get("stop_hook_active"):                     # second stop: pass, but the miss is on record
        log(session, "missed", ids=ids)
        return
    log(session, "close_block", ids=ids)
    me = Path(__file__).as_posix()
    lines = [f"[tripwire] {len(todo)} note(s) from the user are neither applied nor waived. Not done yet:"]
    lines += [f"- note {n['id']}: {n['text']}" + ("" if n.get("injected") else "  (its trigger has not fired: no matching edit yet)") for n in todo]
    lines.append(f'Apply each and run: py -3.13 -I "{me}" applied <id>   |   or: py -3.13 -I "{me}" waive <id> "<reason>"')
    print(json.dumps({"decision": "block", "reason": "\n".join(lines)}))


# ---------------------------------------------------------------- CLI
def opt_all(argv, flag):
    return [argv[i + 1] for i, a in enumerate(argv[:-1]) if a == flag and not argv[i + 1].startswith("--")]


def opt(argv, flag, default=None):
    vals = opt_all(argv, flag)
    return vals[-1] if vals else default


def find_log(start):
    """<dir>/.state/LOG.md, looking upward from `start` (a folder that already holds LOG.md also counts)."""
    p = Path(start).resolve()
    if (p / "LOG.md").is_file():
        return p / "LOG.md"
    for d in (p, *p.parents):
        if (d / ".state" / "LOG.md").is_file():
            return d / ".state" / "LOG.md"
    return None


def cmd_add(session, argv):
    data = load(session) or {"notes": []}
    note = {"id": max([n["id"] for n in data["notes"]], default=0) + 1, "text": argv[0], "paths": opt_all(argv, "--path"),
            "symbols": opt_all(argv, "--symbol"), "keywords": opt_all(argv, "--keyword"), "status": "open", "added": now(), "block": "--block" in argv}
    already = [f for f in touched(session) if any(path_hit(t, f) for t in note["paths"])]
    if already:                                          # "already-touched steer": a revisit item, not a first-touch note
        note["revisit"] = already
    data["notes"].append(note)
    save(session, data)
    triggers = len(note["paths"]) + len(note["symbols"]) + len(note["keywords"])
    log(session, "add", id=note["id"], revisit=bool(already), triggers=triggers)
    return add_report(note, already, triggers)


def add_report(note, already, triggers):
    out = {"id": note["id"], "status": "open"}
    if already:
        out["already_touched"] = already
        out["do"] = "revisit these files now (put it on the TASK frontier), then run applied"
    elif not triggers:
        out["warning"] = "no trigger: it will never be injected, only the close gate will raise it"
    return out


def cmd_mark(session, argv, status):
    data = load(session)
    note = next((n for n in (data or {"notes": []})["notes"] if str(n["id"]) == argv[0]), None)
    if not note:
        return {"error": f"no note {argv[0]}"}
    if note["status"] != "open":
        return {"error": f"note {argv[0]} is already {note['status']}"}
    if status == "waived" and (len(argv) < 2 or argv[1].startswith("--")):
        return {"error": "waive needs a reason"}
    note["status"], note["closed"] = status, now()
    out = {"id": note["id"], "status": status}
    if status == "waived":
        note["reason"] = argv[1]
    else:
        out["log"] = log_applied(note, argv)
    save(session, data)
    log(session, status, id=note["id"], injected=bool(note.get("injected")), reason=note.get("reason"))
    return out


def log_applied(note, argv):
    lp = find_log(opt(argv, "--state") or os.getcwd())
    if not lp:
        return "no .state/LOG.md found: nothing appended"
    ptr = opt(argv, "--pointer") or "user message " + note["added"]
    text = " ".join(note["text"].replace("|", "/").split())
    line = (f"{now()} | finding | {text} | hypothesis | {ptr}" if "--fact" in argv
            else f"{now()} | decision | tripwire note applied: {text} | {ptr}")
    with open(lp, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    return f"appended to {lp}"


def cmd_stats(session):
    """The print's step-1 measures, from the logs: notes applied before done, missed, false injections."""
    p = sdir(session) / "tripwire.log"
    ev = [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines()] if p.is_file() else []
    count = lambda name: sum(1 for e in ev if e["event"] == name)     # noqa: E731
    return {"notes": count("add"), "applied": count("applied"), "waived": count("waived"), "injections": count("inject"),
            "false_injections": sum(1 for e in ev if e["event"] == "waived" and e.get("injected")),
            "close_blocks": count("close_block"), "missed_at_close": sum(len(e["ids"]) for e in ev if e["event"] == "missed"),
            "open": len(open_notes(load(session) or {"notes": []}))}


def main(argv):
    mode = argv[0] if argv else ""
    if mode in ("pre", "post", "close"):
        try:
            hook = json.loads(sys.stdin.read() or "{}")
        except ValueError:
            return
        {"pre": mode_pre, "post": mode_post, "close": mode_close}[mode](hook)
        return
    session = opt(argv, "--session") or os.environ.get("CLAUDE_CODE_SESSION_ID")
    out = run_cli(mode, session, argv[1:]) if session else {"error": "no session: pass --session <id>"}
    print(json.dumps(out, indent=1, ensure_ascii=False))


def run_cli(mode, session, rest):
    has_arg = bool(rest) and not rest[0].startswith("--")
    if mode == "add" and has_arg:
        return cmd_add(session, rest)
    if mode in ("applied", "waive") and has_arg:
        return cmd_mark(session, rest, "applied" if mode == "applied" else "waived")
    if mode == "list":
        return (load(session) or {"notes": []})["notes"]
    if mode == "stats":
        return cmd_stats(session)
    return {"error": "usage: tripwire.py add|applied|waive|list|stats (see the file header)"}


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        main(sys.argv[1:])
    except Exception as e:                                # a hook must never break the tool call it watches
        print(f"tripwire: {type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(0)
