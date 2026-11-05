"""live_gate_20261105.py: frozen-holdout test of the LIVE router gate (opsd.route replica) vs prod head gate. Writes data/live-gate-20261105.json. Run from skilljepa dir."""
import sys, os, json, datetime as dt, numpy as np
sys.path.insert(0, os.getcwd())
import holdout_check as hc, avb_feats as af, avb_np, arms
from a17_user_profile import table_probs
from a6c_nudger import train_stats
LAG, HL, WIN, NOM = 600, 1800.0, 6 * 3600, 0.35
sj, X, Y, lab, allp, sid, names, plab, te = hc._base(); NS = len(names); nidx = {nm: j for j, nm in enumerate(names)}
rows = [json.loads(l) for l in open(hc.HOLD, encoding="utf-8")]; ei = np.array([r["i"] for r in rows]); assert (ei == np.where(te)[0]).all()
n = len(lab); N = len(ei); L = lab[ei] >= 0; gold = [allp[i] for i in ei]
pairs, _, _ = sj.load(); tt = np.array([dt.datetime.fromisoformat(p["t"].replace("Z", "+00:00")).timestamp() for p in pairs]); cls = np.where(lab >= 0, lab, NS)
pg, Tg = train_stats(lab, plab, np.where(~te)[0], NS)
PT = np.zeros((N, NS + 1))   # session table, as ablate_20261105.py
for k, i in enumerate(ei):
    hist = np.where((sid == sid[i]) & (tt < tt[i] - LAG))[0]; hist = hist[np.argsort(tt[hist], kind="stable")]
    ts = np.zeros((NS + 1, NS + 1)); csn = np.zeros(NS + 1); prev = NS
    for h in hist: c = cls[h]; ts[prev, c] += 1; csn[c] += 1; prev = c
    PT[k] = table_probs(prev, ts, csn, Tg, pg)
LPT = np.log(PT + 1e-9)
# holdout-history prev_prior: earlier same-sid HOLDOUT rows, t older than LAG s, true labels (skill names, len>2), half-life 1800, window 6 h; weights at row time
hs = [r["sid"] for r in rows]; ht = tt[ei]; hsk = [[names[j] for j in allp[i] if j < NS] for i in ei] if all(isinstance(j, (int, np.integer)) for g in allp[:50] for j in g) else None
if hsk is None: hsk = [[x for x in r["skills"] if x != "none"] for r in rows]
def hold_prior(k):
    w = {}
    for m in range(N):
        if hs[m] == hs[k] and m != k and 0 <= ht[k] - ht[m] and ht[k] - ht[m] >= LAG and ht[k] - ht[m] < WIN:
            for sk in hsk[m]:
                if len(sk) > 2: w[sk] = w.get(sk, 0.0) + 0.5 ** ((ht[k] - ht[m]) / HL)
    return w
HP = [hold_prior(k) for k in range(N)]
pairs_l = pairs
def met(h1, h3, route):
    return {"hit1": h1[L].mean(), "hit3": h3[L].mean(), "acc_abst": np.where(route, h1, ~L).mean(), "fr_none": route[~L].mean()}
rd = lambda m: {a: round(float(b), 3) for a, b in m.items()}
a0 = arms.A0(frozen=True); base_use = a0.use
sc_full = [a0.score(pairs[i]) for i in ei]                       # A0 incl. its frozen-usage prior (= ablate X0)
a0.use = {}; sc_np = [a0.score(pairs[i]) for i in ei]; a0.use = base_use   # A0 trigger+bm25 only
nalias = sum(1 for s in a0.skills.values() if s.get("aliases"))
def add_prior(sc, hp):
    d = dict(sc); tot = sum(hp.values()) or 1.0
    if hp:
        d = {k: round(v + hp.get(k, 0.0) / tot, 3) for k, v in d.items()}
        for k, v in hp.items():
            if k not in d and k in a0.skills: d[k] = round(v / tot, 3)
    return sorted(((k, v) for k, v in d.items() if v > 0), key=lambda kv: -kv[1])
sc5 = [add_prior(sc_np[k], HP[k]) for k in range(N)]
def rank_arm(sc, w=0.0):
    h1, h3, top = [], [], []
    for k, s in enumerate(sc):
        top.append(s[0][1] if s else -9.0)
        if not s: h1.append(False); h3.append(False); continue
        nm = [x for x, _ in s]; v = np.array([y for _, y in s])
        if w: v = v - v.max(); v = v - np.log(np.exp(v).sum()); v = v + w * LPT[k][[nidx[x] for x in nm]]
        o = np.argsort(-v, kind="stable"); r = [nidx[nm[j]] for j in o]; h1.append(r[0] in gold[k]); h3.append(bool(set(r[:3]) & gold[k]))
    return np.array(h1), np.array(h3), np.array(top)
out = {}
h1, h3, top5 = rank_arm(sc5); r5 = top5 >= NOM; has5 = np.array([bool(s) for s in sc5])
out["X5"] = rd(met(h1, h3, r5)) | {"fr_none_raw": round(float(has5[~L].mean()), 3)}
h1b, h3b, topb = rank_arm(sc_full); out["X5b"] = rd(met(h1b, h3b, topb >= NOM)) | {"fr_none_raw": round(float(np.array([bool(s) for s in sc_full])[~L].mean()), 3)}
h1n, h3n, topn = rank_arm(sc_np); out["X5b_noprior_at_all"] = rd(met(h1n, h3n, topn >= NOM))
h1c, h3c, topc = rank_arm([add_prior(sc_full[k], HP[k]) for k in range(N)]); out["X5c_double_prior"] = rd(met(h1c, h3c, topc >= NOM))
# head internals (as holdout_check.score)
W = avb_np.load_head(os.path.join(sj.D, "avb_head.npz")); meta = W["__meta__"]; K = meta["K"]
E = sj.knn_energies(X[~te], lab[~te], X[ei], len(Y)); B = np.array([r["B"] for r in rows], np.float32); Pr = np.zeros_like(B)
for k, r in enumerate(rows):
    for j, v in r["Pr"].items(): Pr[k, int(j)] = v
ap = [allp[i - 1] if i and sid[i - 1] == sid[i] else set() for i in range(n)]
cs = af.cands(E, B, K, True); T = af.feats(X[ei], Y, E, B, Pr, cs, plab[ei], [ap[j] for j in ei], True)
P = {"cands": cs, "allp": gold, "lab": lab[ei]}; Z0 = avb_np.forward(W, T); hroute = af.pnone(Z0) < meta["thr"]
hh1, hh3 = af.hits(Z0, P); out["X2_check"] = rd(met(hh1, hh3, hroute))
out["X6"] = rd(met(h1, h3, hroute & has5))
Z = Z0.copy(); Z[:, :K] += 0.3 * np.stack([LPT[k][np.array(cs[k])] for k in range(N)]); nh1, nh3 = af.hits(Z, P)
out["X7"] = rd(met(nh1, nh3, hroute))
h1d, h3d, _ = rank_arm(sc5, 0.3); out["X8"] = rd(met(h1d, h3d, hroute & has5))
# 11-05 X9, pre-registered AFTER the live probe (head gate abstained on "convert this movie folder to hevc to free drive space", A0 top 3.42): hybrid gate =
# route iff has5 & (head routes | A0 top >= F). Bar: fr_none <= .30 AND lab_correct (routed & hit1 on labeled rows) >= .8 * X5's. Caveat: head feats still carry the holdout prior.
cov = lambda route: {"lab_cov": round(float(route[L].mean()), 3), "lab_correct": int((route & h1)[L].sum())}
out["X5"] |= cov(r5); out["X6"] |= cov(hroute & has5)
for F in (1.0, 1.5, 2.0, 3.0):
    rt = has5 & (hroute | (top5 >= F)); m = rd(met(h1, h3, rt)) | cov(rt)
    m["supported"] = bool(m["fr_none"] <= .30 and m["lab_correct"] >= 0.8 * out["X5"]["lab_correct"]); out[f"X9_F{F}"] = m
out["X6"]["supported"] = bool(out["X6"]["acc_abst"] >= out["X5"]["acc_abst"] + .02 and out["X6"]["fr_none"] <= out["X5"]["fr_none"])
out["X5b"]["matches_ablate_X0"] = bool(abs(out["X5b"]["hit1"] - .350) < .0015)
out["X7"]["reproduces_632_815_744"] = bool((out["X7"]["hit1"], out["X7"]["hit3"], out["X7"]["acc_abst"]) == (.632, .815, .744))
meta_o = {"n_lab": int(L.sum()), "n_none": int((~L).sum()), "always_abstain": round(float((~L).mean()), 3), "head_thr": float(meta["thr"]), "n_alias_skills_frozen": nalias,
          "notes": "X5 base = trig+bm25 w/o A0 frozen-usage prior, + holdout-history prev_prior (labels, lag 600s, hl 1800, 6h, sum-1 norm). Omitted: relevancy() usage boost, fuzzy_tier (needs live route.log, not frozen), aliases. X5c adds holdout prior on top of A0's own usage prior (double)."}
json.dump({"numbers": out, **meta_o}, open(os.path.join(sj.D, "live-gate-20261105.json"), "w"), indent=1); print(json.dumps(out, indent=1), meta_o)
