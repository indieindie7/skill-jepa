"""judge_fresh.py: fresh-context judge (SELFIMPROVE_PLAN sec 9 step 4). Samples N labelled trace rows (data/pairs_trace.jsonl, label != none),
shows a local model ONLY the query + the skill names with one-line descriptions (data/skills.json; no plan, no notes, no router answer, no label),
asks for one skill or none. Writes data/judge_disagree.jsonl rows {sid,t,q,label,router_top1,judge,agree,run} (appended; agree = judge == router top-1
from data/trace_pairs.jsonl, joined on sid+query). Disagreement is logged, never resolved: nothing reads it back into training.
Model: Sage-A1.5 llama-server on 8093 (reused when already up, never stopped then; else started hidden on the GPU gpu.pick_gpu() picks and stopped at the end).
No GPU with >= 9.5 GB free (or a game on gpu 0 and gpu 1 full) -> skipped, exit 3.
  py -3.13 -I judge_fresh.py [--n 30] [--seed 1] [--stub] [--out PATH]   --stub: fake judge (no model), for tests"""
import os, sys, json, re, random, time, argparse, subprocess

HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "data"); LLM = os.path.join(HERE, "..", "llm")
sys.path.insert(0, LLM)
PORT = 8093


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")] if os.path.exists(p) else []


def sample_rows(n, seed):
    tp = {(r["sid"], r["query"]): r for r in jl(os.path.join(D, "trace_pairs.jsonl"))}
    rows = []
    for r in jl(os.path.join(D, "pairs_trace.jsonl")):
        lab = (r.get("skills_clean") or r.get("skills") or ["none"])[0]
        t = tp.get((r["sid"], r["q"]))
        if lab == "none" or not t or not t.get("route_top"): continue
        rows.append({"sid": r["sid"], "t": r["t"], "q": r["q"], "label": lab, "router_top1": t["route_top"][0]})
    random.Random(seed).shuffle(rows)
    return rows[:n]


def prompt(q, skills):
    cat = "\n".join(f"- {k}: {v}" for k, v in skills.items())
    return ("You pick which skill, if any, a request needs. Skills:\n" + cat +
            "\n\nReply with exactly one skill name from the list, or none if no skill fits. No other words.\n\nRequest: " + q[:400])


def parse(text, skills):
    t = (text or "").strip().lower().strip("`'\". \n")
    if t in skills: return t
    hits = [k for k in skills if k in t]
    return max(hits, key=len) if hits else "none"


def start_model():
    import localread as lr, gpu
    if lr.healthy(PORT): return lr, None
    g = gpu.pick_gpu(lr.SAGE_MB)
    if g < 0: return lr, "skip"
    log = open(os.path.join(LLM, "localread.log"), "w")
    proc = subprocess.Popen([lr.SERVER, "-m", lr.SAGE, "--port", str(PORT), "-ngl", "99", "-c", "8192", "-np", "1", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0",
                             "--jinja", "--device", f"CUDA{g}", *lr.SAGE_EXTRA], cwd=os.path.dirname(lr.SERVER), stdout=log, stderr=subprocess.STDOUT,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    for _ in range(240):
        if lr.healthy(PORT): print(f"judge_fresh: Sage up on gpu {g}"); return lr, proc
        if proc.poll() is not None: sys.exit("judge_fresh: Sage exited during start-up (see llm/localread.log)")
        time.sleep(1)
    subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True); sys.exit("judge_fresh: Sage not up in 4 min")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=30); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--stub", action="store_true"); ap.add_argument("--out", default=os.path.join(D, "judge_disagree.jsonl")); a = ap.parse_args()
    skills = json.load(open(os.path.join(D, "skills.json"), encoding="utf-8"))
    rows = sample_rows(a.n, a.seed)
    if not rows: print("no labelled rows with a router top-1"); return 1
    lr = proc = None
    if a.stub:
        random.seed(a.seed); ask = lambda r: random.choice([r["router_top1"], r["label"], "none"])   # fake judge: tests the plumbing only
    else:
        lr, proc = start_model()
        if proc == "skip": print("skipped: no GPU with >= 9.5 GB free (or game on gpu 0)"); return 3
        def ask(r):
            for temp in (0.0, 0.7):
                x = lr.http(PORT, "/v1/chat/completions", {"messages": [{"role": "user", "content": prompt(r["q"], skills)}], "temperature": temp,
                            "max_tokens": 30, "chat_template_kwargs": {"enable_thinking": False}})
                t = lr.clean(x["choices"][0]["message"].get("content") or "")
                if t: return parse(t, skills)
            return "none"
    run = time.strftime("%F %T"); out = []
    try:
        for r in rows:
            j = ask(r); out.append({**r, "judge": j, "agree": j == r["router_top1"], "run": run + (" stub" if a.stub else "")})
    finally:
        if proc is not None:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True); print("judge_fresh: Sage stopped")
    with open(a.out, "a", encoding="utf-8", newline="\n") as f:
        for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = len(out); ag = sum(r["agree"] for r in out); jl_ = sum(r["judge"] == r["label"] for r in out); rl = sum(r["router_top1"] == r["label"] for r in out)
    print(f"judge_fresh{' STUB' if a.stub else ''}: n {n} judge==router {ag / n:.3f} ({ag}/{n}) judge==label {jl_ / n:.3f} router==label {rl / n:.3f} -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
