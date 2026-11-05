"""a14_gate_fit.py: A14 abstain gate offline fit on route.log col 9 gf= (SELFIMPROVE_PLAN A14). Labels = trace_pairs rule (skill read <=600 s same sid = skill, else none).
Prod proxy: avb head rule pnone<0.4485 with pnone from logged feats (only top3/clipped logits logged): S=exp(htop-zn)*exp(hent). Live opsd conf=='none' also scored.
  py -3.13 -I a14_gate_fit.py"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trace_pairs as tp
THR = 0.4485; NAMES = "htop hmarg hent znone lchars lwords url q sym fu".split()

def load():
    use = []
    for l in open(tp.USAGE, encoding="utf-8"):
        try: j = json.loads(l); t = tp.ts(j["t"], "%Y-%m-%d %H:%M")
        except (ValueError, KeyError): continue
        if t: use.append((t, j["skill"], j.get("sid", "")))
    R, seen = [], set()
    for l in open(tp.ROUTE_LOG, encoding="utf-8"):
        p = l.rstrip("\n").split("\t")
        if len(p) < 9 or not p[8].startswith("gf=") or p[3].startswith("<") or len(p[3]) < 8: continue
        sid = p[4]; k = (sid, p[0], p[3])
        if k in seen or sid.startswith("test"): continue
        seen.add(k)
        try: f = [float(x) for x in p[8][3:].split(",")]; t = tp.ts(p[0], "%Y-%m-%d %H:%M:%S")
        except ValueError: continue
        if len(f) != 10 or not t: continue
        read = {sk for ut, sk, us in use if 0 <= ut - t <= tp.WINDOW and (not sid or not us or us == sid) and len(sk) > 2}
        R.append((sid, f, 1 if read else 0, p[1] == "none"))   # label 1 = skill
    return R

def auc(s, y):   # P(score_skill > score_none), ties .5
    from itertools import groupby
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    for _, g in groupby(sorted(zip(s, range(len(s)))), key=lambda x: x[0]):
        g = [i for _, i in g]; r[g] = r[g].mean()
    n1 = y.sum(); n0 = len(y) - n1; return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def fit(X, y, l2=1.0, it=600, lr=0.3):
    w = np.zeros(X.shape[1]); b = 0.0
    for _ in range(it):
        p = 1 / (1 + np.exp(-(X @ w + b))); g = p - y
        w -= lr * (X.T @ g / len(y) + l2 * w / len(y)); b -= lr * g.mean()
    return w, b

def met(route, y):   # route=1 -> routed; skill rows should route, none rows abstain
    return {"acc_abst": float((route == y).mean()), "fr_none": float(route[y == 0].mean())}

def main():
    R = load(); sid = np.array([r[0] for r in R]); X = np.array([r[1] for r in R]); y = np.array([r[2] for r in R]); live = np.array([0 if r[3] else 1 for r in R])
    n = len(y); ht, hm, he, zn = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    pn = 1 / (1 + np.exp(np.clip(ht - zn + he, -50, 50)))   # proxy pnone = 1/(1+S)
    prod_route = (pn < THR).astype(int); prod = met(prod_route, y); prod["auc_znone"] = auc(-zn, y)
    mu, sd = X.mean(0), X.std(0) + 1e-9; Z = (X - mu) / sd
    us = sorted(set(sid)); rng = np.random.RandomState(0); rng.shuffle(us); fold = {s: i % 5 for i, s in enumerate(us)}; fo = np.array([fold[s] for s in sid])
    oof = np.zeros(n); route = np.zeros(n, int)
    for k in range(5):
        tr, te = fo != k, fo == k; w, b = fit(Z[tr], y[tr]); st = 1 / (1 + np.exp(-(Z[tr] @ w + b))); oof[te] = 1 / (1 + np.exp(-(Z[te] @ w + b)))
        best = (-1, 1.0)
        for t in np.unique(np.r_[st, 1.0]):
            r = (st >= t).astype(int)
            if r[y[tr] == 0].mean() <= prod["fr_none"] + 1e-12:
                a = (r == y[tr]).mean()
                if a > best[0]: best = (a, t)
        route[te] = (oof[te] >= best[1]).astype(int)
    lr_m = met(route, y); lr_m["auc"] = auc(oof, y)
    w, b = fit(Z, y); top = sorted(zip(NAMES, w), key=lambda x: -abs(x[1]))[:3]
    out = {"n_rows": n, "n_labelled_skill": int(y.sum()), "n_none": int((y == 0).sum()), "always_abstain_acc": float((y == 0).mean()), "prod": prod, "lr_cv": lr_m,
           "top3_feature_weights": [[a, round(float(v), 3)] for a, v in top], "n_sessions": len(us),
           "prod_live_conf_none": met(live, y), "prod_note": "head pnone proxy from logged feats (full 16 logits not logged)",
           "supported": bool(lr_m["acc_abst"] >= prod["acc_abst"] + .02 and lr_m["fr_none"] <= prod["fr_none"] + 1e-12)}
    json.dump(out, open(os.path.join(tp.HERE, "data", "a14-gate-20261105.json"), "w"), indent=1); print(json.dumps(out))

if __name__ == "__main__": main()
