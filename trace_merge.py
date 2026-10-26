"""trace_merge.py: nightly loop trace_pairs -> data/pairs_trace.jsonl (+ data/emb_trace.npz) -> AVB head retrain at 300 new rows (SELFIMPROVE_PLAN sec 1/6/8).
Rows in pairs.jsonl format; sids already in data/pairs.jsonl dropped (transcript-labelled = leakage, sec 8).
  py -3.13 -I trace_merge.py [--dry-run] [--retrain-at 300] [--force-retrain]   exit 2 = embedder 8084 down (nothing written)"""
import os, sys, re, json, time, argparse, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "data"); sys.path.insert(0, HERE)
NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)
TP, PT, EMB = (os.path.join(D, f) for f in ("trace_pairs.jsonl", "pairs_trace.jsonl", "emb_trace.npz"))
STATE, LOG = os.path.join(D, "trace_state.json"), os.path.join(D, "trace_merge.log")
TRAIN_PY = r"C:\llm\qlora\venv\Scripts\python.exe"


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")] if os.path.exists(p) else []


def iso(s):   # route.log local time -> pairs.jsonl style UTC
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(time.mktime(time.strptime(s, "%Y-%m-%d %H:%M:%S")))) + ".000Z"


def convert(rows, skills, old_sids, have):
    new, drop = [], {}
    for r in rows:
        sid, q, lab = r.get("sid", ""), r["query"], r.get("label", "")
        why = ("empty_sid" if not sid else "bad_sid" if not re.fullmatch(r"[0-9a-f]{8}", sid) else "in_pairs" if sid in old_sids
               else "short_q" if len(q) < 8 or q.startswith("<") else "no_label" if not lab and r["class"] != "none"
               else "unknown_skill" if lab and lab not in skills else None)
        if not why:
            t = iso(r["t"])
            if f"{sid}|{t}|{q}" in have: why = "dup"
        if why: drop[why] = drop.get(why, 0) + 1; continue
        sk = [lab] if lab and r["class"] != "none" else ["none"]
        new.append({"sid": sid, "t": t, "q": q, "skills": sk, "skills_clean": sk, "src": "trace", "cls": r["class"]})
        have.add(f"{sid}|{t}|{q}")
    return new, drop


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--retrain-at", type=int, default=300); ap.add_argument("--force-retrain", action="store_true"); a = ap.parse_args()
    if not a.dry_run: subprocess.run(["py", "-3.13", "-I", os.path.join(HERE, "trace_pairs.py")], cwd=HERE, creationflags=NOWIN, capture_output=True, check=True)
    skills = json.load(open(os.path.join(D, "skills.json"), encoding="utf-8")); old_sids = {p["sid"][:8] for p in jl(os.path.join(D, "pairs.jsonl"))}
    cur = jl(PT); have = {f"{p['sid']}|{p['t']}|{p['q']}" for p in cur}; tp = jl(TP)
    new, drop = convert(tp, skills, old_sids, have)
    X = np.load(EMB)["X"] if os.path.exists(EMB) else np.zeros((0, 768), np.float32); X = X[:len(cur)]
    todo = cur[len(X):] + new     # rows lacking an embedding (crash between npz and jsonl writes) are re-embedded
    msg = f"trace {len(tp)} kept {len(new)} dropped {sum(drop.values())} {drop} embed {len(todo)} total {len(cur) + len(new)}"
    if a.dry_run: print("dry-run " + msg); return 0
    if todo:
        import skilljepa as sj
        try: E = sj.embed([sj.Q_PRE + p["q"] for p in todo])
        except OSError as e:
            open(LOG, "a", encoding="utf-8").write(f"{time.strftime('%F %T')} ERR embedder down {e}; {msg}\n"); print("embedder down:", e); return 2
        X = np.vstack([X, E]); np.savez(EMB + ".tmp.npz", X=X); os.replace(EMB + ".tmp.npz", EMB)
        with open(PT, "a", encoding="utf-8", newline="\n") as f:
            for r in new: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    st = json.load(open(STATE)) if os.path.exists(STATE) else {"rows_since_retrain": 0, "last_retrain": None, "last_retrain_result": None}
    st["rows_total"] = len(cur) + len(new); st["rows_since_retrain"] += len(new); st["last_run"] = time.strftime("%F %T")
    if a.force_retrain or st["rows_since_retrain"] >= a.retrain_at:
        tag = "trace-" + time.strftime("%Y%m%d"); out = os.path.join(D, f"avb-{tag}.json")
        r = subprocess.run([TRAIN_PY, "avb_rerank.py", "--extra", PT, "--tag", tag], cwd=HERE, creationflags=NOWIN, capture_output=True, text=True, timeout=1500)
        try:
            j = json.load(open(out, encoding="utf-8")); h = j["avb2_head_thr"]; res = f"rc {r.returncode} extra {j['avb2_info'].get('extra_rows')} hit1 {h['hit1'][0]} acc_abstain {h['acc_with_abstain'][0]} (v2 .566/.751) -> {out}"
        except (OSError, KeyError, ValueError): res = f"rc {r.returncode} FAILED: {(r.stderr or r.stdout)[-300:]}"
        st.update(last_retrain=st["last_run"], last_retrain_result=res)
        if r.returncode == 0: st["rows_since_retrain"] = 0
        msg += " | retrain " + res
    json.dump(st, open(STATE, "w"), indent=1); open(LOG, "a", encoding="utf-8").write(f"{st['last_run']} {msg}\n"); print(msg); return 0


if __name__ == "__main__":
    sys.exit(main())
