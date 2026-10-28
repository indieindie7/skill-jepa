"""trace_merge.py: nightly loop trace_pairs -> data/pairs_trace.jsonl (+ data/emb_trace.npz) -> AVB head retrain at 300 new rows (SELFIMPROVE_PLAN sec 1/6/8).
Rows in pairs.jsonl format; sids already in data/pairs.jsonl dropped (transcript-labelled = leakage, sec 8).
  py -3.13 -I trace_merge.py [--dry-run] [--retrain-at 300] [--force-retrain]   exit 2 = embedder 8084 down (nothing written)"""
import os, sys, re, json, time, argparse, subprocess, shutil
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(_v, "1")   # before numpy: the holdout gate forward is tiny matmuls
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "data"); sys.path.insert(0, HERE)
NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)
TP, PT, EMB = (os.path.join(D, f) for f in ("trace_pairs.jsonl", "pairs_trace.jsonl", "emb_trace.npz"))
STATE, LOG = os.path.join(D, "trace_state.json"), os.path.join(D, "trace_merge.log")
TRAIN_PY = r"C:\llm\qlora\venv\Scripts\python.exe"
HEAD, CAND, PREV = (os.path.join(D, f) for f in ("avb_head.npz", "avb_head.cand.npz", "avb_head.prev.npz"))   # 10-28: served head (opsd col 8) / fresh retrain candidate / last accepted
GATE_TOL = 0.005   # candidate must score hit1 >= last accepted - GATE_TOL on the frozen holdout (data/holdout.jsonl, holdout_check.py)


def gate(cand, st, apply=True):
    """10-28 frozen-holdout gate. Returns (accepted, msg). Hash mismatch = refuse (never score against a changed holdout). Accepted: cand -> data/avb_head.npz (old kept as avb_head.prev.npz) and
    {date, hit1, hit3, acc_with_abstain} appended to st['holdout_hist']; else keep the old weights, log rollback. apply=False scores and decides but touches nothing."""
    import holdout_check as hc
    ok, vmsg = hc.verify()
    if not ok: return False, "REFUSE holdout hash mismatch (" + vmsg + ")"
    hist = st.setdefault("holdout_hist", [])
    if not hist and os.path.exists(HEAD):   # first gated run: the head currently served is the first accepted baseline
        b = hc.score(HEAD); hist.append({"date": time.strftime("%F"), "hit1": round(float(b["hit1"]), 4), "hit3": round(float(b["hit3"]), 4), "acc_with_abstain": round(float(b["acc_with_abstain"]), 4), "src": "baseline"})
    m = hc.score(cand); last = hist[-1]["hit1"] if hist else -1.0; h1 = round(float(m["hit1"]), 4)
    if h1 >= last - GATE_TOL:
        if apply:
            if os.path.exists(HEAD): shutil.copyfile(HEAD, PREV)
            os.replace(cand, HEAD); hist.append({"date": time.strftime("%F"), "hit1": h1, "hit3": round(float(m["hit3"]), 4), "acc_with_abstain": round(float(m["acc_with_abstain"]), 4), "src": "retrain"}); del hist[:-30]
        return True, f"accept holdout hit1 {h1} >= last accepted {last} - {GATE_TOL}"
    if apply and os.path.exists(cand): os.remove(cand)
    return False, f"rollback holdout hit1 {h1} < last accepted {last} - {GATE_TOL}; kept old weights"


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
    ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--retrain-at", type=int, default=300); ap.add_argument("--force-retrain", action="store_true"); ap.add_argument("--gate-test", metavar="HEAD.npz", help="score this head through the holdout gate against the stored holdout_hist, print the decision, change nothing"); a = ap.parse_args()
    if a.gate_test:
        st = json.load(open(STATE)) if os.path.exists(STATE) else {}; ok, m = gate(a.gate_test, st, apply=False); print(("ACCEPT " if ok else "REJECT ") + m); return 0 if ok else 4
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
        import holdout_check as hc
        hok, hmsg = hc.verify()
        if not hok:   # 10-28: frozen holdout changed = no retrain, no new weights
            msg += " | retrain REFUSED: holdout hash mismatch (" + hmsg + ")"; st["last_retrain_result"] = "REFUSED holdout hash mismatch"
            json.dump(st, open(STATE, "w"), indent=1); open(LOG, "a", encoding="utf-8").write(f"{st['last_run']} {msg}\n"); print(msg); return 3
        tag = "trace-" + time.strftime("%Y%m%d"); out = os.path.join(D, f"avb-{tag}.json")
        r = subprocess.run([TRAIN_PY, "avb_rerank.py", "--extra", PT, "--tag", tag, "--save", CAND], cwd=HERE, creationflags=NOWIN, capture_output=True, text=True, timeout=1500)
        try:
            j = json.load(open(out, encoding="utf-8")); h = j["avb2_head_thr"]; res = f"rc {r.returncode} extra {j['avb2_info'].get('extra_rows')} hit1 {h['hit1'][0]} acc_abstain {h['acc_with_abstain'][0]} (v2 .566/.751) -> {out}"
        except (OSError, KeyError, ValueError): res = f"rc {r.returncode} FAILED: {(r.stderr or r.stdout)[-300:]}"
        st.update(last_retrain=st["last_run"], last_retrain_result=res)
        m = re.search(r"hit1 ([0-9.]+)", res)
        if m: st["retrain_hist"] = (st["retrain_hist"] + [float(m.group(1))])[-30:]; run_hit1 = float(m.group(1))
        if r.returncode == 0: st["rows_since_retrain"] = 0
        if r.returncode == 0 and os.path.exists(CAND):
            gok, gmsg = gate(CAND, st); res += " | gate " + gmsg; st["last_retrain_result"] = res
            if not gok: st["last_rollback"] = {"date": time.strftime("%F %T"), "msg": gmsg}
        msg += " | retrain " + res
    st["history"] = (st.get("history", []) + [{"date": time.strftime("%F"), "n_new": len(new), "agreement": ag, "agreement_read": agr, "explore_hit_rate": ehr, "explore_hit": eh, "explore_miss": em, "suspect_hit_rate": shr, "suspect_hit": sh, "suspect_miss": sm, "falsifier_score": fs, "refuted": fr, "with_prediction": fn, "hit1": run_hit1}])[-30:]
    st["low_surprise"] = low_surprise(st["history"], st["retrain_hist"])
    msg += f" | surprise agree {ag if ag is None else round(ag, 3)} explore_hit_rate {ehr if ehr is None else round(ehr, 3)} ({eh}/{eh + em}) suspect_hit_rate {shr if shr is None else round(shr, 3)} ({sh}/{sh + sm}) falsifier_score {fs if fs is None else round(fs, 3)} ({fr}/{fn}) low_surprise {st['low_surprise']}"
    json.dump(st, open(STATE, "w"), indent=1); open(LOG, "a", encoding="utf-8").write(f"{st['last_run']} {msg}\n"); print(msg); return 0


if __name__ == "__main__":
    sys.exit(main())
