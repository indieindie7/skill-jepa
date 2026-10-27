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
        why = ("suspect_miss" if r["class"] == "suspect_miss"   # 10-27 23:xx usual-suspect slot: miss = no label, suspect_hit is a labelled row (label = suspect skill), counted apart from explore
               else "explore_miss" if r["class"] == "explore_miss"   # 10-27 exploration slot: no skill read = no label; explore_hit is a labelled row (label = explore skill)
               else "empty_sid" if not sid else "bad_sid" if not re.fullmatch(r"[0-9a-f]{8}", sid) else "in_pairs" if sid in old_sids
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


LABELLED = ("brief_only", "full_read", "miss")


def surprise(rows):
    """Night's rows -> (agreement, agreement_read, explore_hit_rate, explore_hit, explore_miss, suspect_hit_rate, suspect_hit, suspect_miss). agreement = labelled rows whose label == router top-1
    (brief_only + full_read vs miss; brief_only agrees by construction, so agreement_read = full_read vs miss only is logged too).
    Rows carrying an explore slot count by class_main for agreement (their `class` is explore_*)."""
    cm = [(r.get("class_main") or r["class"], r) for r in rows]
    lab = [c for c, r in cm if c in LABELLED and r.get("label") and r.get("route_top")]
    ag = sum(1 for c in lab if c != "miss") / len(lab) if lab else None
    rd = [c for c in lab if c != "brief_only"]
    agr = sum(1 for c in rd if c == "full_read") / len(rd) if rd else None
    eh = sum(1 for r in rows if r["class"] == "explore_hit"); em = sum(1 for r in rows if r["class"] == "explore_miss")
    sh = sum(1 for r in rows if r["class"] == "suspect_hit"); sm = sum(1 for r in rows if r["class"] == "suspect_miss")   # never pooled with explore
    return ag, agr, (eh / (eh + em) if eh + em else None), eh, em, (sh / (sh + sm) if sh + sm else None), sh, sm


def falsifier(rows):
    """Night's rows -> (falsifier_score, n_refuted, n_with_prediction). SELFIMPROVE_PLAN sec 9: refuted / rows with a prediction
    (trace_pairs.py writes refuted true|false for rows whose route.log line has col 7 'refute=...'; None = no prediction).
    Score is None when there are 0 predictions."""
    wp = [r for r in rows if r.get("refuted") is not None]; nr = sum(1 for r in wp if r["refuted"])
    return (nr / len(wp) if wp else None), nr, len(wp)


def low_surprise(hist, retrain_hist, k=7):
    """Ashby 2nd-loop trigger (SELFIMPROVE_PLAN sec 9): last k runs all agree > 0.9 AND hit1 did not improve over the last k retrains
    (fewer than 2 retrains = nothing improved). No action yet, signal only."""
    last = hist[-k:]
    if len(last) < k or any(h.get("agreement") is None or h["agreement"] <= 0.9 for h in last): return False
    rh = retrain_hist[-k:]
    return not (len(rh) >= 2 and rh[-1] > rh[0] + 0.005)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--retrain-at", type=int, default=300); ap.add_argument("--force-retrain", action="store_true"); a = ap.parse_args()
    if not a.dry_run: subprocess.run(["py", "-3.13", "-I", os.path.join(HERE, "trace_pairs.py")], cwd=HERE, creationflags=NOWIN, capture_output=True, check=True)
    skills = json.load(open(os.path.join(D, "skills.json"), encoding="utf-8")); old_sids = {p["sid"][:8] for p in jl(os.path.join(D, "pairs.jsonl"))}
    cur = jl(PT); have = {f"{p['sid']}|{p['t']}|{p['q']}" for p in cur}; tp = jl(TP)
    new, drop = convert(tp, skills, old_sids, have)
    X = np.load(EMB)["X"] if os.path.exists(EMB) else np.zeros((0, 768), np.float32); X = X[:len(cur)]
    todo = cur[len(X):] + new     # rows lacking an embedding (crash between npz and jsonl writes) are re-embedded
    msg = f"trace {len(tp)} kept {len(new)} dropped {sum(drop.values())} {drop} embed {len(todo)} total {len(cur) + len(new)}"
    st = json.load(open(STATE)) if os.path.exists(STATE) else {"rows_since_retrain": 0, "last_retrain": None, "last_retrain_result": None}
    nrows = [r for r in tp if r["t"] > st.get("last_run", "")]   # the night's rows = trace rows newer than the last successful run
    ag, agr, ehr, eh, em, shr, sh, sm = surprise(nrows)
    fs, fr, fn = falsifier(nrows)
    if "retrain_hist" not in st:   # seed from the one retrain recorded before this field existed
        m = re.search(r"hit1 ([0-9.]+)", st.get("last_retrain_result") or ""); st["retrain_hist"] = [float(m.group(1))] if m else []
    if a.dry_run:
        print("dry-run " + msg); st = dict(st); st["rows_total"] = len(cur) + len(new); st["rows_since_retrain"] += len(new)
        h = list(st.get("history", []))[-29:] + [{"date": time.strftime("%F"), "n_new": len(new), "agreement": ag, "agreement_read": agr, "explore_hit_rate": ehr, "explore_hit": eh, "explore_miss": em, "suspect_hit_rate": shr, "suspect_hit": sh, "suspect_miss": sm, "falsifier_score": fs, "refuted": fr, "with_prediction": fn, "hit1": None}]
        st["history"] = h; st["low_surprise"] = low_surprise(h, st["retrain_hist"]); print(json.dumps(st, indent=1)); return 0
    if todo:
        import skilljepa as sj
        try: E = sj.embed([sj.Q_PRE + p["q"] for p in todo])
        except OSError as e:
            open(LOG, "a", encoding="utf-8").write(f"{time.strftime('%F %T')} ERR embedder down {e}; {msg}\n"); print("embedder down:", e); return 2
        X = np.vstack([X, E]); np.savez(EMB + ".tmp.npz", X=X); os.replace(EMB + ".tmp.npz", EMB)
        with open(PT, "a", encoding="utf-8", newline="\n") as f:
            for r in new: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    st["rows_total"] = len(cur) + len(new); st["rows_since_retrain"] += len(new); st["last_run"] = time.strftime("%F %T")
    run_hit1 = None
    if a.force_retrain or st["rows_since_retrain"] >= a.retrain_at:
        tag = "trace-" + time.strftime("%Y%m%d"); out = os.path.join(D, f"avb-{tag}.json")
        r = subprocess.run([TRAIN_PY, "avb_rerank.py", "--extra", PT, "--tag", tag], cwd=HERE, creationflags=NOWIN, capture_output=True, text=True, timeout=1500)
        try:
            j = json.load(open(out, encoding="utf-8")); h = j["avb2_head_thr"]; res = f"rc {r.returncode} extra {j['avb2_info'].get('extra_rows')} hit1 {h['hit1'][0]} acc_abstain {h['acc_with_abstain'][0]} (v2 .566/.751) -> {out}"
        except (OSError, KeyError, ValueError): res = f"rc {r.returncode} FAILED: {(r.stderr or r.stdout)[-300:]}"
        st.update(last_retrain=st["last_run"], last_retrain_result=res)
        m = re.search(r"hit1 ([0-9.]+)", res)
        if m: st["retrain_hist"] = (st["retrain_hist"] + [float(m.group(1))])[-30:]; run_hit1 = float(m.group(1))
        if r.returncode == 0: st["rows_since_retrain"] = 0
        msg += " | retrain " + res
    st["history"] = (st.get("history", []) + [{"date": time.strftime("%F"), "n_new": len(new), "agreement": ag, "agreement_read": agr, "explore_hit_rate": ehr, "explore_hit": eh, "explore_miss": em, "suspect_hit_rate": shr, "suspect_hit": sh, "suspect_miss": sm, "falsifier_score": fs, "refuted": fr, "with_prediction": fn, "hit1": run_hit1}])[-30:]
    st["low_surprise"] = low_surprise(st["history"], st["retrain_hist"])
    msg += f" | surprise agree {ag if ag is None else round(ag, 3)} explore_hit_rate {ehr if ehr is None else round(ehr, 3)} ({eh}/{eh + em}) suspect_hit_rate {shr if shr is None else round(shr, 3)} ({sh}/{sh + sm}) falsifier_score {fs if fs is None else round(fs, 3)} ({fr}/{fn}) low_surprise {st['low_surprise']}"
    json.dump(st, open(STATE, "w"), indent=1); open(LOG, "a", encoding="utf-8").write(f"{st['last_run']} {msg}\n"); print(msg); return 0


if __name__ == "__main__":
    sys.exit(main())
