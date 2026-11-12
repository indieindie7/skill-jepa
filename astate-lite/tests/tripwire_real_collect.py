"""Collect real user prompts since the dispatcher went live, run triage() on each, write numbered files. Prints counts only."""
import collections
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / ".claude/hooks"))
import tripwire_dispatch as td  # noqa: E402

OUT = Path(sys.argv[1])
SINCE = dt.datetime(2026, 11, 10, 17, 0).astimezone()
SKIP = ("<", "This session is being continued", "[", "Caveat:", "Stop hook", "The user hasn't heard")


def text_of(m):
    c = (m.get("message") or {}).get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        if any(b.get("type") in ("tool_result", "image") for b in c if isinstance(b, dict)):
            return ""
        return " ".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")
    return ""


rows, seen = [], set()
for p in (Path.home() / ".claude/projects").glob("*/*.jsonl"):
    if dt.datetime.fromtimestamp(p.stat().st_mtime).astimezone() < SINCE:
        continue
    with open(p, encoding="utf-8", errors="replace") as f:
        for ln in f:
            if '"type":"user"' not in ln:
                continue
            try:
                m = json.loads(ln)
            except ValueError:
                continue
            if m.get("type") != "user" or m.get("isMeta") or m.get("isSidechain"):
                continue
            try:
                t = dt.datetime.fromisoformat(m.get("timestamp", "").replace("Z", "+00:00"))
            except ValueError:
                continue
            if t < SINCE:
                continue
            s = " ".join(text_of(m).split())
            if not s or s.startswith(SKIP) or len(s) > 600 or s in seen:
                continue
            seen.add(s)
            rows.append((t.isoformat(), p.stem, s))
rows.sort()
preds = []
for i, (t, sess, s) in enumerate(rows, 1):
    r = td.triage(s)
    preds.append({"n": i, "t": t, "session": sess, "cls": r["cls"], "escalated": r["escalated"], "reason": r.get("reason", "")})
(OUT / "real_msgs.txt").write_text("\n".join(f"{i}. {s[:300]}" for i, (_, _, s) in enumerate(rows, 1)) + "\n", encoding="utf-8")
(OUT / "real_pred.json").write_text(json.dumps(preds, indent=0), encoding="utf-8")
n = len(preds)
print("messages", n, "sessions", len({p["session"] for p in preds}))
print("class", dict(collections.Counter(p["cls"] for p in preds)))
print("escalated", sum(p["escalated"] for p in preds))
print("reasons", dict(collections.Counter(p["reason"].split(":")[0] for p in preds if p["escalated"])))
print("unescalated by class", dict(collections.Counter(p["cls"] for p in preds if not p["escalated"])))
