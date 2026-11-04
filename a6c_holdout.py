"""A6c on frozen holdout: prod head logits + w*log p_table(class | session history older than LAG s). Session split = per-sid hit1."""
import sys, os, json, numpy as np
sys.path.insert(0, os.getcwd())
import holdout_check as hc, avb_feats as af, avb_np
from a17_user_profile import table_probs
from a6c_nudger import train_stats
W_, LAG = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3, 600
sj, X, Y, lab, allp, sid, names, plab, te = hc._base(); W = avb_np.load_head(os.path.join(sj.D, "avb_head.npz")); meta = W["__meta__"]; K = meta["K"]; NS = len(names)
rows = [json.loads(l) for l in open(hc.HOLD, encoding="utf-8")]; ei = np.array([r["i"] for r in rows]); assert (ei == np.where(te)[0]).all()
B = np.array([r["B"] for r in rows], np.float32); Pr = np.zeros_like(B)
for k, r in enumerate(rows):
    for j, v in r["Pr"].items(): Pr[k, int(j)] = v
n = len(lab); allp_prev = [allp[i - 1] if i and sid[i - 1] == sid[i] else set() for i in range(n)]
E = sj.knn_energies(X[~te], lab[~te], X[ei], len(Y)); cs = af.cands(E, B, K, True)
T = af.feats(X[ei], Y, E, B, Pr, cs, plab[ei], [allp_prev[j] for j in ei], True)
P = {"cands": cs, "allp": [allp[j] for j in ei], "lab": lab[ei]}; Z0 = avb_np.forward(W, T)
pairs, _, _ = sj.load(); import datetime as dt; t = np.array([dt.datetime.fromisoformat(p["t"].replace("Z", "+00:00")).timestamp() for p in pairs]); cls = np.where(lab >= 0, lab, NS)
pg, Tg = train_stats(lab, plab, np.where(~te)[0], NS)
LP = np.zeros((len(ei), K))
for k, i in enumerate(ei):
    hist = np.where((sid == sid[i]) & (t < t[i] - LAG))[0]; hist = hist[np.argsort(t[hist], kind="stable")]
    ts = np.zeros((NS + 1, NS + 1)); csn = np.zeros(NS + 1); prev = NS
    for h in hist: c = cls[h]; ts[prev, c] += 1; csn[c] += 1; prev = c
    pt = table_probs(prev, ts, csn, Tg, pg); LP[k] = np.log(pt[np.array(cs[k])] + 1e-9)
out = {}
for w in sorted({0.0, 0.1, W_, 0.5}):
    Z = Z0.copy(); Z[:, :K] += w * LP; h1, h3 = af.hits(Z, P); L = P["lab"] >= 0; ss = sid[ei]; lt = P["lab"]
    route = af.pnone(Z0) < meta["thr"]   # rank-only nudge: abstain gate stays prod
    m = {"hit1": h1[L].mean(), "hit3": h3[L].mean(), "acc_with_abstain": np.where(~route, lt < 0, h1).mean(), "hit1_strict": (route & h1)[L].mean()}; m = {a: round(float(b), 3) for a, b in m.items()}
    m["sess_hit1"] = {str(s): round(float(h1[L & (ss == s)].mean()), 3) for s in np.unique(ss[L]) if (L & (ss == s)).sum() >= 10}
    out[w] = m; print(w, json.dumps(m))
json.dump({str(k): v for k, v in out.items()}, open(os.path.join(sj.D, "a6c-holdout.json"), "w"), indent=1)
