import sys, json
from pathlib import Path
H = Path(__file__).resolve().parents[1] / "hooks"          # repo copy; installed: ~/.claude/hooks
sys.path.insert(0, str(H if (H / "tripwire.py").exists() else Path.home() / ".claude/hooks"))
import tripwire_dispatch as d
SET = {
 "steer": ["make the button bigger", "use tabs not spaces there", "keep the old api working too", "call it loader_v2", "the timeout should be 30 seconds", "write the notes in short lines", "no globals in that file please", "it should log to the same file", "also strip the whitespace", "prefer json over yaml for that config"],
 "new_task": ["now write the exporter", "Item 2 please", "please make a tool that renames the photos", "build the settings menu next", "Read them all and download the code please", "next add multiplayer lobby", "write tests for the tokenizer", "Shrink them please", "start the terrain generator", "can you set up a weekly report"],
 "status": ["is it finished", "hows it going", "what is the progress", "did it work", "are the tests green", "what are you working on", "how much is left", "what changed so far", "still running?", "did the build pass"],
 "cancel": ["stop", "cancel that", "never mind", "forget it, do the other thing", "abort the run", "hold on", "undo that change", "revert it", "lets call it a day", "this is wrong, go back"],
 "ambiguous": ["hmm", "interesting", "ok", "lol", "thanks", "I see", "wow", "that reminds me of the old days", "nice", "the weather is bad today"],
}
rows = []
for want, msgs in SET.items():
    for m in msgs:
        r = d.triage(m)
        rows.append((want, m, r))
def ok(want, r):
    if want in ("cancel", "ambiguous"):
        return r["escalated"]
    return r["cls"] == want and not r["escalated"]
if "-v" in sys.argv:
    for want, m, r in rows:
        print(("ok  " if ok(want, r) else "BAD ") + f"{want:9} {r['cls']:9} {r.get('score','')!s:6} {r.get('margin','')!s:6} {r['reason']:22} {m}")
n = len(rows)
clear = [x for x in rows if x[0] in ("steer", "new_task", "status")]
print("correct", sum(ok(w, r) for w, _, r in rows), "of", n)
print("clear classes sorted right without escalation", sum(ok(w, r) for w, _, r in clear), "of", len(clear))
print("clear classes WRONG class (not escalated)", sum((not r["escalated"]) and r["cls"] != w for w, _, r in clear))
print("cancel/ambiguous escalated", sum(r["escalated"] for w, _, r in rows if w in ("cancel", "ambiguous")), "of 20")
print("cancel sorted as steer/new_task/status (dangerous)", sum((not r["escalated"]) for w, _, r in rows if w == "cancel"))
print("escalation rate", sum(r["escalated"] for _, _, r in rows), "/", n)
