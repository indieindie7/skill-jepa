"""a1_split.py (jepa-context chat, 10-27): A1 test from notes/context-nudger-PLAN.md.
Claim: fresh-judge vs router disagreement (.233 agree) = closed-loop bias. Wrong if the judge disagrees as much on
context-FREE prompts as on context-dependent ones (then the gap is missing context, not bias).
Steps: Sage up (judge_fresh.start_model, freest GPU) -> judge_fresh.py --n N --seed S (more judged rows, scratch file)
-> Sage tags every judged row ctx|free (prompt text only) -> Sage stopped -> agreement per group -> markdown appended to --out.
Writes only to --scratch and --out; data/ untouched."""
import os, sys, json, time, argparse, subprocess, collections
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import judge_fresh as jf

TAG = ("Message a user typed into a coding/ops assistant. Could a skill router pick the right tool from THIS TEXT ALONE, "
       "or does it need the earlier conversation (pronouns like 'it', 'that', bare 'ok', continuation of unseen work)? "
       "Answer one word: free (text alone is enough) or ctx (needs earlier conversation).\n\nMessage: ")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=60); ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--scratch", required=True); ap.add_argument("--out", default=os.path.join(HERE, "notes", "nudger-measure-2026-10-27.md")); a = ap.parse_args()
    os.makedirs(a.scratch, exist_ok=True)
    lr, proc = jf.start_model()
    if proc == "skip": sys.exit("no GPU free for Sage")
    more = os.path.join(a.scratch, "judge_more.jsonl")
    try:
        if a.n > 0:
            subprocess.run([sys.executable, "-I", os.path.join(HERE, "judge_fresh.py"), "--n", str(a.n), "--seed", str(a.seed), "--out", more], check=False)
        rows, seen = [], set()
        for p in (os.path.join(jf.D, "judge_disagree.jsonl"), more):
            if not os.path.exists(p): continue
            for r in jf.jl(p):
                k = (r["sid"], r["q"])
                if k in seen or " stub" in r.get("run", ""): continue
                seen.add(k); rows.append(r)
        for r in rows:
            x = lr.http(jf.PORT, "/v1/chat/completions", {"messages": [{"role": "user", "content": TAG + r["q"][:600]}], "temperature": 0.0,
                        "max_tokens": 5, "chat_template_kwargs": {"enable_thinking": False}})
            t = lr.clean(x["choices"][0]["message"].get("content") or "").lower()
            r["ctx"] = "ctx" if "ctx" in t else ("free" if "free" in t else "?")
    finally:
        if proc is not None:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True); print("a1_split: Sage stopped")
    with open(os.path.join(a.scratch, "a1_tagged.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    g = collections.defaultdict(list)
    for r in rows: g[r["ctx"]].append(r)
    out = ["", f"## A1 context-dependence split {time.strftime('%Y-%m-%d %H:%M')} (a1_split.py, Sage tags ctx|free from prompt text only; rows = judge_disagree.jsonl + {a.n} more via judge_fresh seed {a.seed})",
           "  | group | n | judge==router | judge==label | router==label | judge says none |", "  |---|---|---|---|---|---|"]
    for k in ("free", "ctx", "?"):
        rs = g.get(k, [])
        if not rs: continue
        n = len(rs); f1 = lambda fn: f"{sum(fn(r) for r in rs)/n:.3f}"
        out.append(f"  | {k} | {n} | {f1(lambda r: r['judge']==r['router_top1'])} | {f1(lambda r: r['judge']==r['label'])} | {f1(lambda r: r['router_top1']==r['label'])} | {f1(lambda r: r['judge']=='none')} |")
    fr, cx = g.get("free", []), g.get("ctx", [])
    if fr and cx:
        af = sum(r["judge"] == r["router_top1"] for r in fr) / len(fr); ac = sum(r["judge"] == r["router_top1"] for r in cx) / len(cx)
        out.append(f"- verdict: judge==router free {af:.3f} vs ctx {ac:.3f}; A1 (gap = loop bias) {'HOLDS' if af - ac < 0.15 else 'WEAKENED: gap is mostly missing context'} (threshold .15); router==label on both groups shows what the self-labels reward")
    open(a.out, "a", encoding="utf-8", newline="\n").write("\n".join(out) + "\n"); print("\n".join(out))

if __name__ == "__main__": main()
