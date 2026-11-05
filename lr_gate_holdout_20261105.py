"""lr_gate_holdout_20261105.py: LR abstain gate on the FROZEN holdout, 5-fold CV grouped by sid. Ranking = prod head unchanged. Writes data/lr-gate-holdout-20261105.json.
gate_feats: imported from avb_serve (string 'gf=a,b,..' parsed back; values rounded to 3 dp as in the live log)."""
import sys, os, json, numpy as np
sys.path.insert(0, os.getcwd())
import holdout_check as hc, avb_feats as af, avb_np
from avb_serve import gate_feats
FN = ["htop", "hmarg", "hent", "znone", "log_chars", "log_words", "url", "qmark", "symfrac", "followup"]
sj, X, Y, lab, allp, sid, names, plab, te = hc._base()
rows = [json.loads(l) for l in open(hc.HOLD, encoding="utf-8")]; ei = np.array([r["i"] for r in rows]); assert (ei == np.where(te)[0]).all()
n = len(lab); gold = [allp[i] for i in ei]; Ln = lab[ei] >= 0
W = avb_np.load_head(os.path.join(hc.D, "avb_head.npz")); meta = W["__meta__"]; K = meta["K"]
E = sj.knn_energies(X[~te], lab[~te], X[ei], len(Y)); B = np.array([r["B"] for r in rows], np.float32); Pr = np.zeros_like(B)
for k, r in enumerate(rows):
    for j, v in r["Pr"].items(): Pr[k, int(j)] = v
ap = [allp[i - 1] if i and sid[i - 1] == sid[i] else set() for i in range(n)]
cs = af.cands(E, B, K, True); T = af.feats(X[ei], Y, E, B, Pr, cs, plab[ei], [ap[j] for j in ei], True)
P = {"cands": cs, "allp": gold, "lab": lab[ei]}; Z = avb_np.forward(W, T)
h1, h3 = af.hits(Z, P); route_prod = af.pnone(Z) < meta["thr"]
F = np.array([[float(v) for v in gate_feats(r["q"], Z[k, :K], Z[k, K])[3:].split(",")] for k, r in enumerate(rows)])
isn = ~Ln; N = len(F)
def acc(route, m=slice(None)): return float(np.where(route, h1, isn)[m].mean())
def frn(route, m=slice(None)): return float(route[m & isn].mean()) if isinstance(m, np.ndarray) else float(route[isn].mean())
def auc(s, y):
    o = np.argsort(s, kind="stable"); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    for v in np.unique(s):
        mk = s == v
        if mk.sum() > 1: r[mk] = r[mk].mean()
    p = y.sum(); q = len(y) - p; return float((r[y].sum() - p * (p + 1) / 2) / (p * q))
def fit(Xt, y, l2=1.0, it=300):
    w = np.zeros(Xt.shape[1] + 1); A = np.c_[Xt, np.ones(len(Xt))]; R = np.eye(len(w)) * l2; R[-1, -1] = 0
    for _ in range(it):
        p = 1 / (1 + np.exp(-A @ w)); g = A.T @ (p - y) + R @ w; H = (A * (p * (1 - p))[:, None]).T @ A + R + 1e-6 * np.eye(len(w))
        d = np.linalg.solve(H, g); w -= d
        if np.abs(d).max() < 1e-8: break
    return w
def pred(w, Xs): return 1 / (1 + np.exp(-(np.c_[Xs, np.ones(len(Xs))] @ w)))
prod_fr = frn(route_prod); out = {}
sids = np.array([r["sid"] for r in rows]); us = np.unique(sids); rng = np.random.default_rng(0); rng.shuffle(us)
fold_of = {s: i % 5 for i, s in enumerate(us)}; fold = np.array([fold_of[s] for s in sids])
score = np.zeros(N); route_cv = np.zeros(N, bool)
for f in range(5):
    tr, tt = fold != f, fold == f; mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-9
    w = fit((F[tr] - mu) / sd, isn[tr].astype(float)); ptr = pred(w, (F[tr] - mu) / sd); pts = pred(w, (F[tt] - mu) / sd)
    best = (-1, 2.0)   # route iff p_none < t ; max train acc_abst s.t. train fr_none <= prod_fr
    for t in np.r_[0.0, np.unique(ptr), 1.0001]:
        rt = ptr < t
        if rt[isn[tr]].mean() <= prod_fr + 1e-12:
            a = np.where(rt, h1[tr], isn[tr]).mean()
            if a > best[0] or (a == best[0] and t > best[1]): best = (a, t)
    score[tt] = pts; route_cv[tt] = pts < best[1]
cv = {"acc_abst": acc(route_cv), "fr_none": frn(route_cv), "hit1": float(h1[Ln].mean()), "hit3": float(h3[Ln].mean()), "auc": auc(score, isn)}
prod = {"acc_abst": acc(route_prod), "fr_none": prod_fr, "hit1": float(h1[Ln].mean()), "hit3": float(h3[Ln].mean())}
aa = {"acc_abst": float(isn.mean()), "fr_none": 0.0}
mu, sd = F.mean(0), F.std(0) + 1e-9; wf = fit((F - mu) / sd, isn.astype(float)); top = np.argsort(-np.abs(wf[:-1]))[:3]
supported = bool(cv["acc_abst"] >= prod["acc_abst"] + .02 and cv["fr_none"] <= prod["fr_none"])
rd = lambda d: {k: round(v, 4) for k, v in d.items()}
res = {"numbers": {"cv": rd(cv), "prod": rd(prod), "always_abstain": rd(aa), "supported": supported,
       "top3_weights_std": {FN[i]: round(float(wf[i]), 3) for i in top}, "n": N, "n_lab": int(Ln.sum()), "n_none": int(isn.sum()), "n_sids": len(us)},
       "note": "ranking=prod head; gate_feats imported from avb_serve; fold-internal standardize; thr on train fold with train fr_none<=prod fr_none; metrics pooled over 5 sid-grouped folds (seed 0)"}
json.dump(res, open(os.path.join(hc.D, "lr-gate-holdout-20261105.json"), "w"), indent=1); print(json.dumps(res, indent=1))
