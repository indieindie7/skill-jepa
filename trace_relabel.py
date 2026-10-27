"""Relabel router-self-labelled trace rows (brief_only, miss) with local Sage. py -3.13 -I trace_relabel.py [--dry-run]
Writes data/pairs_trace_relabel.jsonl (same rows/order). Starts Sage (localread.start_own) if 8093 is down and stops it after."""
import json, os, re, sys, urllib.request
import numpy as np
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
LLM = r"C:\Users\john\Documents\Tools\llm"
SYS = ("You label which skill a user request needs. Candidates are numbered. Reply with ONLY the number of the single best skill, "
       "or 0 if none of them clearly fits the request. No words.")
IN, OUT = os.path.join(D, "pairs_trace.jsonl"), os.path.join(D, "pairs_trace_relabel.jsonl")


def prompt(r, names, skills, X, Y):
    cands = [r["skills"][0]] if r["skills"][0] in skills else []
    for j in np.argsort(-(Y @ X))[:12]:
        if names[j] not in cands and len(cands) < 8: cands.append(names[j])
    cands.sort(key=lambda c: -float(Y[names.index(c)] @ X))   # similarity order, label not always first
    lst = "\n".join(f"{i}. {c}: {skills[c]}" for i, c in enumerate(cands, 1))
    return cands, f"Request: {r['q'][:600]}\n\nCandidate skills:\n{lst}\n0. none of these\n\nNumber:"


def ask(u):
    body = {"messages": [{"role": "system", "content": SYS}, {"role": "user", "content": u}], "temperature": 0, "max_tokens": 8, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request("http://127.0.0.1:8093/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=300).read())["choices"][0]["message"]["content"] or ""


def main():
    dry = "--dry-run" in sys.argv
    rows = [json.loads(l) for l in open(IN, encoding="utf-8")]; X = np.load(os.path.join(D, "emb_trace.npz"))["X"]
    z = np.load(os.path.join(D, "emb.npz")); Y, names = z["Y"], [str(n) for n in z["names"]]; skills = json.load(open(os.path.join(D, "skills.json"), encoding="utf-8"))
    done = [json.loads(l) for l in open(OUT, encoding="utf-8")] if os.path.exists(OUT) and not dry else []
    todo = [i for i, r in enumerate(rows) if r["cls"] in ("brief_only", "miss")]
    if dry:
        for i in todo[:3]: print(prompt(rows[i], names, skills, X[i], Y)[1], "\n-----")
        return
    sys.path.insert(0, LLM); import localread as lr
    proc = None
    if not lr.healthy(lr.SAGE_PORT): proc = lr.start_own(need_mb=lr.SAGE_MB, model=lr.SAGE, extra=lr.SAGE_EXTRA, port=lr.SAGE_PORT); print("started Sage", flush=True)
    else: print("reused Sage", flush=True)
    try:
        out = list(done)
        for i in range(len(done), len(rows)):
            r = dict(rows[i])
            if r["cls"] in ("brief_only", "miss"):
                cands, u = prompt(r, names, skills, X[i], Y); raw = ask(u); m = re.search(r"\d+", raw); k = int(m.group()) if m else -1
                r["orig_label"], r["judge_raw"] = r["skills"][0], raw
                r["judge_label"] = "none" if k == 0 else cands[k - 1] if 1 <= k <= len(cands) else "PARSEFAIL"
                if r["judge_label"] != "PARSEFAIL": r["skills"] = r["skills_clean"] = [r["judge_label"]]
            out.append(r)
            with open(OUT, "w", encoding="utf-8") as f: f.writelines(json.dumps(x) + "\n" for x in out)
    finally:
        if proc: lr._Tree(proc).kill(); print("stopped Sage")
    j = [r for r in out if "judge_label" in r]; pf = [r for r in j if r["judge_label"] == "PARSEFAIL"]; ok = [r for r in j if r["judge_label"] != "PARSEFAIL"]
    ag = [r for r in ok if r["judge_label"] == r["orig_label"]]; nn = [r for r in ok if r["judge_label"] == "none"]; ch = [r for r in ok if r not in ag and r not in nn]
    print(f"judged {len(j)} agree {len(ag)} changed-skill {len(ch)} changed-none {len(nn)} parse-fail {len(pf)}")
    for r in (ch + nn)[:10]: print(f"  {r['q'][:80]!r}: {r['orig_label']} -> {r['judge_label']}")


main()
