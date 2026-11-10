"""tripwire_dispatch.py: Tripwire step 2, the dispatcher (skill astate-lite tree/tripwire-print.md, print s6 "Second, the
dispatcher with small-model triage"). UserPromptSubmit hook: sorts the user's message into the print's five classes and tells
the main model, in one or two lines, what the class asks for. "scripts measure, small models sort, the main model judges."

  script      stop words (any hit -> escalate: the print's guard against "cancel read as steer"), bare link, files named in the
              message vs the touched-file log (write conflict / already-touched steer)
  small model EmbeddingGemma-300M on 127.0.0.1:8084 (the resident CPU helper): nearest prototype over steer / new_task /
              status / cancel / remark; cancel, remark, low score or low margin -> escalate
  main model  only what was escalated

Speaks only in a session that has edited files or holds notes (mid-task); elsewhere it only logs. Every decision is one
"dispatch" line in compact-captures/<session>/tripwire.log (escalation rate = the print's step-2 metric).

  hook:  py -3.13 -I tripwire_dispatch.py hook            (JSON on stdin: session_id, prompt)
  CLI:   py -3.13 -I tripwire_dispatch.py classify "<message>" [--session <id>]
"""
import hashlib
import json
import math
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tripwire as tw  # noqa: E402

EMB_URL = "http://127.0.0.1:8084/v1/embeddings"
CACHE = Path(__file__).resolve().parent / "tripwire_protos.json"
MIN_SCORE, MIN_MARGIN = 0.50, 0.03          # tuned 11-10 on tripwire_eval.py's labelled set; below either -> escalate
STOP = re.compile(r"\b(stop|cancel|abort|halt|hold on|never ?mind|forget (it|that)|scrap (it|that)|drop (it|that)|undo|revert|"
                  r"pare|para com|cancela|esquece)\b", re.I)
URL_ONLY = re.compile(r"^\s*https?://\S+\s*$")
FILE = re.compile(r"[\w./\\-]*\w\.[A-Za-z][A-Za-z0-9]{0,5}\b|[\w.-]+(?:[/\\][\w.-]+)+")
PROTOS = {
    "steer": ["use the shared palette instead of hardcoded colours", "keep the port configurable", "make it blue, not red",
              "also handle the empty case there", "no, name it foo_bar", "when you edit the loader keep the old format working",
              "use plain python for that, no numpy", "make sure that function stays short", "prefer the short flag names",
              "actually put it in the hooks folder", "keep the comments in english", "that should be a warning, not an error"],
    "new_task": ["now build the audio mixer", "next, write a script that exports the map", "can you also make a new tool for screenshots",
                 "start on the inventory screen", "add a new feature: save slots", "please research this and write a report",
                 "read the papers and download the code", "create a separate test for the parser", "item 3 please",
                 "watch this video and take notes", "let's do the next one", "implement the importer please"],
    "status": ["how is it going", "what's the status", "is it done yet", "what are you doing now", "did the tests pass",
               "how far along are you", "what's left to do", "any progress", "which files did you change", "what did you find",
               "are you still running", "how long will it take"],
    "cancel": ["that's enough for today", "leave it, let's do something else", "we are done with this", "change of plans",
               "go back to how it was before", "not now, tomorrow", "this approach is wrong, go back to the plan",
               "don't continue with this"],
    "remark": ["okay", "thank you", "cool", "haha", "good to know", "huh", "got it", "great", "that is odd", "makes sense", "oh",
               "funny, I was thinking about something else"],
}
DO = {
    "status": "status question: answer from the TASK capsule, LOG and notes (tripwire.py list); start no new work for it.",
    "cancel": "you judge (possible cancel/redirect). If it is one: checkpoint (TASK + LOG line), stop workers, re-plan. "
              "If it is a steer: tripwire.py add \"<steer>\" --path <file>.",
    "ambiguous": "class unclear, you judge: steer (tripwire.py add \"<steer>\" --path <file>), new task, status question or cancel.",
}


# ---------------------------------------------------------------- small model
def embed(texts):
    body = json.dumps({"input": ["task: classification | query: " + t for t in texts]}).encode()
    req = urllib.request.Request(EMB_URL, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=4) as r:
        data = json.loads(r.read())["data"]
    return [unit(e["embedding"]) for e in sorted(data, key=lambda e: e["index"])]


def unit(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def protos():
    """{class: [unit vectors]}; embedded once, cached by the hash of the prototype text."""
    key = hashlib.sha1(json.dumps(PROTOS, sort_keys=True).encode()).hexdigest()
    try:
        c = json.loads(CACHE.read_text(encoding="utf-8"))
        if c["key"] == key:
            return c["vecs"]
    except (OSError, ValueError, KeyError):
        pass
    vecs = {cls: [[round(x, 5) for x in v] for v in embed(texts)] for cls, texts in PROTOS.items()}
    tmp = CACHE.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps({"key": key, "vecs": vecs}), encoding="utf-8")
    os.replace(tmp, CACHE)
    return vecs


def nearest(text):
    """(class, score, margin) by the mean of each class's two closest prototypes."""
    q = embed([text])[0]
    scores = {}
    for cls, vecs in protos().items():
        top = sorted((sum(a * b for a, b in zip(q, v)) for v in vecs), reverse=True)[:2]
        scores[cls] = sum(top) / len(top)
    rank = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return rank[0][0], rank[0][1], rank[0][1] - rank[1][1]


# ---------------------------------------------------------------- script tier
def named_files(text):
    return [tw.norm_path(m).strip("./") for m in FILE.findall(re.sub(r"https?://\S+", " ", text))]


def overlap(files, touched):
    """Touched files a named file points at: same basename, and the named path is a suffix of the touched one."""
    hit = []
    for f in files:
        hit += [t for t in touched if t == f or t.endswith("/" + f)]
    return sorted(set(hit))


def triage(text):
    """{cls, escalated, reason, score, margin}; cls is one of steer new_task status cancel ambiguous."""
    stop = STOP.search(text)
    if stop:
        return {"cls": "cancel", "escalated": True, "reason": "stop_word:" + stop.group(0).lower()}
    if URL_ONLY.match(text):
        return {"cls": "new_task", "escalated": False, "reason": "bare_link"}
    try:
        cls, score, margin = nearest(text[:600])
    except (OSError, ValueError, KeyError) as e:
        return {"cls": "ambiguous", "escalated": True, "reason": "embedder_down:" + type(e).__name__}
    out = {"cls": cls, "escalated": cls == "cancel", "reason": "model", "score": round(score, 3), "margin": round(margin, 3)}
    if cls == "remark" or score < MIN_SCORE or margin < MIN_MARGIN:
        out.update(cls="ambiguous", escalated=True, reason=("remark" if cls == "remark" else f"unsure:{cls}"))
    return out


# ---------------------------------------------------------------- what the class asks for
def steer_line(session, text, hit, files):
    """A steer becomes a note at once (the close gate then holds 'done' until it is applied or waived)."""
    args = [text[:300]]
    for p in hit or files:
        args += ["--path", p]
    rep = tw.cmd_add(session, args)
    me = (Path(__file__).parent / "tripwire.py").as_posix()
    if hit:
        how = f"it names files already edited ({', '.join(hit)}): revisit them now"
    elif files:
        how = f"it fires at the first edit of {', '.join(files)}"
    else:
        how = "no file named, so only the close gate raises it: apply it where it belongs"
    return rep["id"], (f"steer -> note {rep['id']} recorded; {how}. Then: py -3.13 -I \"{me}\" applied {rep['id']}"
                       f"   |   not a steer: py -3.13 -I \"{me}\" waive {rep['id']} \"<reason>\"")


def task_line(hit):
    if hit:
        return f"new task, WRITE CONFLICT with files this task edited ({', '.join(hit)}): queue it behind the current task, one writer."
    return "new task, no write overlap with edited files: may run as a separate worker (subagent or side session); reads in parallel."


def decide(session, text):
    """(log record, line for the main model or None)."""
    rec = triage(text)
    touched = tw.touched(session)
    data = tw.load(session)
    if not touched and not (data and data["notes"]):          # not mid-task: sort and log, say nothing
        return rec, None
    files = named_files(text)
    hit = overlap(files, touched)
    rec["conflict"] = bool(hit)
    if rec["cls"] == "steer":
        rec["note"], line = steer_line(session, text, hit, files)
    elif rec["cls"] == "new_task":
        line = task_line(hit)
    else:
        line = DO[rec["cls"]] + (f" ({rec['reason']})" if rec["escalated"] else "")
    return rec, "[tripwire] " + line


def run(session, text):
    t0 = time.time()
    rec, line = decide(session, text)
    rec["ms"] = int((time.time() - t0) * 1000)
    rec["spoke"] = line is not None
    if tw.sdir(session).is_dir():
        tw.log(session, "dispatch", **rec)
    return rec, line


def main(argv):
    if argv[:1] == ["hook"]:
        hook = json.loads(sys.stdin.read() or "{}")
        text = str(hook.get("prompt") or "").strip()
        if text and hook.get("session_id"):
            _, line = run(hook["session_id"], text)
            if line:
                print(json.dumps({"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": line}}))
        return
    if argv[:1] == ["classify"] and len(argv) > 1:
        session = tw.opt(argv, "--session")
        rec, line = run(session, argv[1]) if session else (triage(argv[1]), None)
        print(json.dumps({**rec, "line": line}, ensure_ascii=False))
        return
    print(__doc__)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        main(sys.argv[1:])
    except Exception as e:                                # a hook must never break the prompt it watches
        print(f"tripwire_dispatch: {type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(0)
