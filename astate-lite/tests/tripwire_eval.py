"""tripwire_eval.py: the print's nine-case eval set for ~/.claude/hooks/tripwire.py (skill astate-lite tree/tripwire-print.md s2 "5. Measurement").
Run on every harness change, both directions: each hook must fire when it should and stay silent when it should not.
Runs the real script as a subprocess with hook JSON on stdin, against throwaway session ids (twtest-*), removed at the end.

  py -3.13 -I -X utf8 tripwire_eval.py      exit 0 = every case passes both ways (step-2 cases call hooks/tripwire_dispatch.py; labelled-set bench: tripwire_bench.py -v)
"""
import json
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

H = Path(__file__).resolve().parents[1] / "hooks"          # repo copy; installed: ~/.claude/hooks
H = H if (H / "tripwire.py").exists() else Path.home() / ".claude/hooks"
TW = str(H / "tripwire.py")
DISP = str(H / "tripwire_dispatch.py")
CAP = Path.home() / ".claude/compact-captures"
PY = [sys.executable, "-I", TW]
sessions, rows = [], []


def new_session():
    s = "twtest-" + uuid.uuid4().hex[:10]
    sessions.append(s)
    return s


def cli(session, *args):
    r = subprocess.run(PY + list(args) + ["--session", session], capture_output=True, text=True, encoding="utf-8", cwd=tempfile.gettempdir())
    return json.loads(r.stdout) if r.stdout.strip() else None


def hook(mode, session, path=None, new="x = 1", **extra):
    payload = {"session_id": session, **extra}
    if path:
        payload.update(tool_name="Edit", tool_input={"file_path": path, "old_string": "", "new_string": new})
    r = subprocess.run(PY + [mode], input=json.dumps(payload), capture_output=True, text=True, encoding="utf-8")
    return json.loads(r.stdout) if r.stdout.strip() else None


def ctx(out):
    return (out or {}).get("hookSpecificOutput", {}).get("additionalContext", "")


def blocked(out):
    return (out or {}).get("decision") == "block"


def case(name, fires, silent):
    rows.append((name, bool(fires), bool(silent)))


def steer_before_touch():
    s = new_session()
    cli(s, "add", "keep the port configurable", "--path", "src/server.py")
    first = hook("pre", s, "C:/proj/src/server.py")
    again = hook("pre", s, "C:/proj/src/server.py")
    gate = hook("close", s)
    cli(s, "applied", "1")
    case("steer before touch", "keep the port configurable" in ctx(first) and blocked(gate), again is None and hook("close", s) is None)


def steer_after_touch():
    s = new_session()
    cli(s, "add", "seed note so the session is live", "--keyword", "zzz-never")
    hook("post", s, "C:/proj/src/render.py")
    late = cli(s, "add", "use the shared palette", "--path", "src/render.py")
    fresh = cli(s, "add", "untouched file", "--path", "src/other.py")
    case("steer after touch", bool(late.get("already_touched")), not fresh.get("already_touched"))


def unrelated_file():
    s = new_session()
    cli(s, "add", "only for the shader", "--path", "shaders/skin.hlsl", "--symbol", "SkinLight")
    other = hook("pre", s, "C:/proj/src/audio.py", new="def SkinLighter(): pass")
    hit = hook("pre", s, "C:/proj/shaders/skin.hlsl")
    case("note on an unrelated file", "only for the shader" in ctx(hit), other is None)


def stale_note():
    s = new_session()
    cli(s, "add", "old steer, no longer wanted", "--path", "src/old.py")
    gate = hook("close", s)
    cli(s, "waive", "1", "superseded by the user")
    case("stale note", blocked(gate), hook("close", s) is None and hook("pre", s, "C:/proj/src/old.py") is None)


def say(session, text):
    """The dispatcher's line for a user message ('' = silent)."""
    r = subprocess.run([sys.executable, "-I", DISP, "hook"], input=json.dumps({"session_id": session, "prompt": text}),
                       capture_output=True, text=True, encoding="utf-8")
    return ctx(json.loads(r.stdout)) if r.stdout.strip() else ""


def mid_task(*files):
    s = new_session()
    for f in files:
        hook("post", s, f)
    return s


def notes(session):
    return len(cli(session, "list"))


def parallel_new_task():
    s = mid_task("C:/proj/src/render.py")
    line = say(s, "now build the audio mixer")
    idle = say(new_session(), "now build the audio mixer")           # no edits, no notes: sorts and logs, says nothing
    case("parallel new task", "separate worker" in line and notes(s) == 0, idle == "" and "CONFLICT" not in line)


def conflicting_write():
    s = mid_task("C:/proj/src/render.py")
    clash = say(s, "next, write a new exporter in render.py")
    free = say(s, "next, write a new exporter in export.py")
    case("conflicting write", "WRITE CONFLICT" in clash and "render.py" in clash, "CONFLICT" not in free and "separate worker" in free)


def status_question():
    s = mid_task("C:/proj/src/render.py")
    line = say(s, "what's the progress, did the tests pass")
    steer = say(s, "use the shared palette instead of hardcoded colours")
    case("status question", "status question" in line and notes(s) == 1, "status question" not in steer and "steer -> note 1" in steer)


def cancel():
    s = mid_task("C:/proj/src/render.py")
    line = say(s, "stop, cancel that and use the old layout")
    steer = say(s, "when you edit the loader keep the old format working")   # 11-12: a source prototype; the old sentence now escalates at margin 0.05
    case("cancel", "you judge" in line and "stop_word" in line, "stop_word" not in steer and "steer -> note" in steer and notes(s) == 1)


def ambiguous_remark():
    s = mid_task("C:/proj/src/render.py")
    line = say(s, "huh, funny")
    clear = say(s, "is it done yet")
    case("ambiguous remark", "you judge" in line and notes(s) == 0, "unclear" not in clear and "status question" in clear)


STEP_2 = (parallel_new_task, conflicting_write, status_question, cancel, ambiguous_remark)


def main():
    try:
        for f in (steer_before_touch, steer_after_touch, unrelated_file, stale_note) + STEP_2:
            f()
    finally:
        for s in sessions:
            shutil.rmtree(CAP / s, ignore_errors=True)
    sys.exit(0 if report() == len(rows) else 1)


def report():
    for name, fires, silent in rows:
        print(f"{'PASS' if fires and silent else 'FAIL'}  {name}: fires={fires} silent={silent}")
    ok = sum(1 for _, f, q in rows if f and q)
    print(f"{ok} of {len(rows)} cases pass both directions (the last {len(STEP_2)} need the 8084 embedder up)")
    return ok


if __name__ == "__main__":
    main()
