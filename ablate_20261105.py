"""ablate_20261105.py: frozen-holdout ablation: A0 (BM25+trigger+prior, frozen snapshot) vs prod head, +/- rank-only session-table nudge. Writes data/ablate-20261105.json."""
import sys, os, json, datetime as dt, numpy as np
sys.path.insert(0, os.getcwd())
import holdout_check as hc, avb_feats as af, avb_np, arms
from a17_user_profile import table_probs
from a6c_nudger import train_stats
LAG = 600
sj, X, Y, lab, allp, sid, names, plab, te = hc._base(); NS = len(names); nidx = {nm: j for j, nm in enumerate(names)}
rows = [json.loads(l) for l in open(hc.HOLD, encoding="utf-8")]; ei = np.array([r["i"] for r in rows]); assert (ei == np.where(te)[0]).all()
n = len(lab); L = lab[ei] >= 0; gold = [allp[i] for i in ei]
pairs, _, _ = sj.load(); t = np.array([dt.datetime.fromisoformat(p["t"].replace("Z", "+00:00")).timestamp() for p in pairs]); cls = np.where(lab >= 0, lab, NS)
pg, Tg = train_stats(lab, plab, np.where(~te)[0], NS)
PT = np.zeros((len(ei), NS + 1))   # session table p(class) from rows of same sid with ts < t_i - LAG (any split, as a6c_holdout.py)
for k, i in enumerate(ei):
    hist = np.where((sid == sid[i]) & (t < t[i] - LAG))[0]; hist = hist[np.argsort(t[hist], kind="stable")]
    ts = np.zeros((NS + 1, NS + 1)); csn = np.zeros(NS + 1); prev = NS
    for h in hist: c = cls[h]; ts[prev, c] += 1; csn[c] += 1; prev = c
    PT[k] = table_probs(prev, ts, csn, Tg, pg)
LPT = np.log(PT + 1e-9)

def met(h1, h3, route):
    return {"hit1": h1[L].mean(), "hit3": h3[L].mean(), "acc_abst": np.where(route, h1, ~L).mean(), "fr_none": route[~L].mean()}
rd = lambda m: {a: round(float(b), 3) for a, b in m.items()}
out = {}
# A0 frozen snapshot
a0 = arms.A0(frozen=True); sc = [a0.score(pairs[i]) for i in ei]; top = np.array([s[0][1] if s else -9.0 for s in sc]); route0 = top >= arms.NOMATCH
def a0_arm(w):
    h1, h3 = [], []
    for k, s in enumerate(sc):
        if not s: h1.append(False); h3.append(False); continue
        nm = [x for x, _ in s]; v = np.array([y for _, y in s]); 
        if w: v = v - v.max(); v = v - np.log(np.exp(v).sum()); v = v + w * LPT[k][[nidx[x] for x in nm]]
        o = np.argsort(-v, kind="stable"); r = [nidx[nm[j]] for j in o]; h1.append(r[0] in gold[k]); h3.append(bool(set(r[:3]) & gold[k]))
    return rd(met(np.array(h1), np.array(h3), route0))
out["X0"] = a0_arm(0)
for w in (0.3, 0.5, 1.0): out[f"X1_w{w}"] = a0_arm(w)
# session table only: rank skills by p_table; route iff argmax class is not none
o = np.argsort(-PT[:, :NS], 1); h1 = np.array([o[k, 0] in gold[k] for k in range(len(ei))]); h3 = np.array([bool(set(o[k, :3]) & gold[k]) for k in range(len(ei))])
out["Xs"] = rd(met(h1, h3, PT.argmax(1) != NS))
# heads
def head_arm(path, w):
    W = avb_np.load_head(path); meta = W["__meta__"]; K = meta["K"]
    E = sj.knn_energies(X[~te], lab[~te], X[ei], len(Y)); B = np.array([r["B"] for r in rows], np.float32); Pr = np.zeros_like(B)
    for k, r in enumerate(rows):
        for j, v in r["Pr"].items(): Pr[k, int(j)] = v
    ap = [allp[i - 1] if i and sid[i - 1] == sid[i] else set() for i in range(n)]
    cs = af.cands(E, B, K, True); T = af.feats(X[ei], Y, E, B, Pr, cs, plab[ei], [ap[j] for j in ei], True)
    P = {"cands": cs, "allp": gold, "lab": lab[ei]}; Z0 = avb_np.forward(W, T); Z = Z0.copy()
    if w: Z[:, :K] += w * np.stack([LPT[k][np.array(cs[k])] for k in range(len(ei))])
    h1, h3 = af.hits(Z, P); return rd(met(h1, h3, af.pnone(Z0) < meta["thr"]))
D = sj.D
out["X2"] = head_arm(os.path.join(D, "avb_head.npz"), 0); out["X3"] = head_arm(os.path.join(D, "avb_head.npz"), 0.3); out["X4"] = head_arm(os.path.join(D, "avb_head_a18b_k1.npz"), 0.3)
out["X4"]["supported"] = bool(out["X4"]["hit1"] >= .642 and out["X4"]["acc_abst"] >= .744)
meta = {"n_lab": int(L.sum()), "n_none": int((~L).sum()), "always_abstain": round(float((~L).mean()), 3)}
json.dump({"numbers": out, **meta}, open(os.path.join(D, "ablate-20261105.json"), "w"), indent=1); print(json.dumps(out, indent=1), meta)
