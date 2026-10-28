"""holdout_check.py: FROZEN holdout gate for the AVB head (SELFIMPROVE_PLAN sec 9 / notes/mindfield-sparse-2026-10-28.md sec 8: "held-out set must be frozen + hashed").
  py -3.13 -I holdout_check.py --init            ONE TIME: write data/holdout.jsonl (= the time-split TEST rows of pairs.jsonl: last 30% of every chat, labels + frozen BM25/prior rows) + data/holdout.sha256. Refuses if either exists.
  py -3.13 -I holdout_check.py --verify          sha256 of holdout.jsonl + the pool inputs (pairs.jsonl, emb.npz, skills.json) vs holdout.sha256; exit 0 ok, 3 mismatch
  py -3.13 -I holdout_check.py --score HEAD.npz  verify, then hit1/hit3/acc_with_abstain/... of that head on the holdout (numpy only, no torch); exit 3 on mismatch
Rows: {"i": index in pairs.jsonl, sid, t, q, skills, B: [BM25 per skill, avb_feats.lex at freeze time], Pr: {skill idx: prior}}; knn pool = pairs.jsonl rows before the 70% line (static since 10-22, hashed).
Pipeline = avb_rerank part() for the test rows (K from the head file, union candidates, no shuffle), thr = head's val-tuned none thr (file meta)."""
import hashlib, json, os, sys
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(_v, "1")   # numpy forward on 1237 x 17 slots: 1 thread (default threads thrash on a busy box, 100x slower)
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "data"); sys.path.insert(0, HERE)
HOLD, SHA = os.path.join(D, "holdout.jsonl"), os.path.join(D, "holdout.sha256")
INPUTS = ("pairs.jsonl", "emb.npz", "skills.json")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def verify(hold=HOLD, shaf=SHA, data_dir=D):
    """-> (ok, message). hold is checked against the 'holdout.jsonl' line of shaf, inputs under data_dir against theirs."""
    try: want = {l.split()[1]: l.split()[0] for l in open(shaf, encoding="utf-8") if l.strip()}
    except OSError as e: return False, f"no hash file {shaf}: {e}"
    bad = []
    for nm, p in (("holdout.jsonl", hold),) + tuple((n, os.path.join(data_dir, n)) for n in INPUTS):
        try: got = sha(p)
        except OSError as e: bad.append(f"{nm} unreadable ({e})"); continue
        if want.get(nm) != got: bad.append(f"{nm} hash {got[:12]} != frozen {str(want.get(nm))[:12]}")
    return (not bad), ("holdout hash ok" if not bad else "HOLDOUT HASH MISMATCH: " + "; ".join(bad))


def _base():
    import skilljepa as sj
    X, Y, lab, allp, sid, names, plab = sj.data("prevskill", "all"); n = len(lab); pos, cnt = np.zeros(n), {}
    for i, c in enumerate(sid): pos[i] = cnt.get(c, 0); cnt[c] = pos[i] + 1
    tot = np.array([cnt[c] for c in sid]); te = pos >= 0.7 * tot
    return sj, X, Y, lab, allp, sid, names, plab, te


def init():
    if os.path.exists(HOLD) or os.path.exists(SHA): sys.exit(f"refuse: {HOLD} / {SHA} exist; the holdout is never regenerated")
    import avb_feats as af
    sj, X, Y, lab, allp, sid, names, plab, te = _base(); ei = np.where(te)[0]
    pairs = [json.loads(l) for l in open(os.path.join(D, "pairs.jsonl"), encoding="utf-8")]; rows = [pairs[i] for i in ei]; B, Pr = af.lex(rows, names)
    with open(HOLD, "w", encoding="utf-8", newline="\n") as f:
        for k, (i, p) in enumerate(zip(ei, rows)):
            f.write(json.dumps({"i": int(i), "sid": p["sid"], "t": p["t"], "q": p["q"], "skills": p["skills"], "B": [round(float(v), 5) for v in B[k]],
                                "Pr": {str(j): round(float(Pr[k, j]), 5) for j in np.nonzero(Pr[k])[0]}}, ensure_ascii=False) + "\n")
    with open(SHA, "w", encoding="utf-8", newline="\n") as f:
        for nm, p in (("holdout.jsonl", HOLD),) + tuple((n, os.path.join(D, n)) for n in INPUTS): f.write(f"{sha(p)}  {nm}\n")
    print(f"froze {len(ei)} rows ({int((lab[ei] >= 0).sum())} labelled) -> {HOLD}"); print(open(SHA).read())


def score(head_path, hold=HOLD):
    """Metrics (af.metrics keys) of head_path on the frozen holdout, route = P(none) < head thr."""
    import avb_feats as af, avb_np
    sj, X, Y, lab, allp, sid, names, plab, te = _base(); W = avb_np.load_head(head_path); meta = W["__meta__"]; K = meta["K"]
    rows = [json.loads(l) for l in open(hold, encoding="utf-8")]; ei = np.array([r["i"] for r in rows]); assert (ei == np.where(te)[0]).all(), "holdout rows != current time-split test rows"
    B = np.array([r["B"] for r in rows], np.float32); Pr = np.zeros_like(B)
    for k, r in enumerate(rows):
        for j, v in r["Pr"].items(): Pr[k, int(j)] = v
    n = len(lab); allp_prev = [allp[i - 1] if i and sid[i - 1] == sid[i] else set() for i in range(n)]
    E = sj.knn_energies(X[~te], lab[~te], X[ei], len(Y)); cs = af.cands(E, B, K, True)
    T = af.feats(X[ei], Y, E, B, Pr, cs, plab[ei], [allp_prev[j] for j in ei], True)
    P = {"cands": cs, "allp": [allp[j] for j in ei], "lab": lab[ei]}; Z = avb_np.forward(W, T)
    return af.metrics(Z, P, meta["thr"]) | {"n": len(ei), "n_labelled": int((lab[ei] >= 0).sum()), "thr": meta["thr"]}


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--init" in a: init()
    elif "--verify" in a: ok, m = verify(); print(m); sys.exit(0 if ok else 3)
    elif "--score" in a:
        ok, m = verify(); print(m)
        if not ok: sys.exit(3)
        print(json.dumps(score(a[a.index("--score") + 1])))
    else: print(__doc__)
