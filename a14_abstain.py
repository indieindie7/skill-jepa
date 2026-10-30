"""a14_abstain.py: ledger A14 learned abstention gate vs A12 C1 (head-score thr). Gate target: accept iff B1 top-1 is a gold skill (SKILL-/wrong = reject).
Gate trains on TRAIN+VAL rows (head scores on TRAIN rows are 2-fold sid cross-fitted so they are out-of-sample); thr/hparams from grouped 5-fold (by sid) OOF scores. TEST untouched.
  C:/llm/qlora/venv/Scripts/python.exe a14_abstain.py [--seeds 3]   (CPU)"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse, hashlib, json, re, sys, time, warnings
import numpy as np, torch
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import arms, shrink, a11_state, a12_risk as a12, avb_rerank as ar, skilljepa as sj, opsd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
torch.set_num_threads(int(os.environ.get("SHRINK_THREADS", "3")))
FN = ["head_top", "head_margin", "head_entropy", "c1_comb", "a0_top", "a0_margin", "bm25_top", "pool_size", "len_chars", "len_words", "has_url", "has_q", "code_chars", "short_followup", "prev_none", "has_prev", "prev_conf"] + [f"qemb{i}" for i in range(16)]
FU = re.compile(r"^\W*(ok|okay|yes|yep|sure|go|go on|continue|proceed|do it|do them all|please do them all|all|next|thanks|thx|y)\W*$", re.I)
MODELS = {"LR": [dict(C=0.1), dict(C=1.0)], "GBM": [dict(max_depth=3, learning_rate=0.05, max_iter=150), dict(max_depth=4, learning_rate=0.05, max_iter=100, l2_regularization=1.0)],
          "MLP": [dict(hidden_layer_sizes=(32,), alpha=1e-2), dict(hidden_layer_sizes=(16,), alpha=1e-1)]}


def mk(name, hp, seed):
    if name == "LR": return LogisticRegression(max_iter=2000, **hp)
    if name == "GBM": return HistGradientBoostingClassifier(random_state=seed, **hp)
    return MLPClassifier(max_iter=400, early_stopping=True, random_state=seed, **hp)


def head_scores(sd, parts_k, T_all):
    orig = ar.run; ar.run = lambda s, *a, _sd=sd, **kw: orig(_sd, *a, **kw); ar.Head = lambda din, attn=True, other=False: shrink.SHead(din, 256, 2)
    ms, _ = ar.run(sd, parts_k, True, refit=False); ar.run = orig; m = ms[0]
    with torch.no_grad(): return [m(torch.tensor(T)).numpy()[:, :-1] for T in T_all]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--out", default=os.path.join(arms.D, "a14-20261030.json")); ap.add_argument("--md", default=os.path.join(HERE, "notes", "a14-abstain-20261030.md")); a = ap.parse_args()
    t0 = time.time(); S, names, tokf, extras = a11_state.build(a)
    X, Y, lab, allp, sid, names2, plab = sj.data("prevskill", "all"); n = len(lab)
    pairs = [json.loads(l) for l in open(os.path.join(arms.D, "pairs.jsonl"), encoding="utf-8")]; a0 = arms.A0(frozen=True)
    pool = [a0.score(pairs[i])[:arms.POOL] for i in range(n)]; bm = np.array([max((a0.bm25(opsd._tok(pairs[i]["q"]), nm) for nm in a0.names), default=0.0) for i in range(n)])
    print("pools+bm25", round(time.time() - t0), flush=True)
    q = [p["q"] for p in pairs]; shape = np.array([[len(s), len(s.split()), float("http" in s or "youtu" in s), float("?" in s), sum(s.count(c) for c in "{}[]();=<>_\\/") / max(len(s), 1), float(bool(FU.match(s)) or len(s.split()) <= 3)] for s in q], np.float32)
    shape[:, :2] = np.log1p(shape[:, :2])
    R = {sp: {} for sp in S}; meta = {}; names_l = names
    for sp in S:
        parts = S[sp]["parts"]; tr, va, te = parts["train"], parts["val"], parts["test"]
        grp = np.array([int(hashlib.md5(("cf" + sid[i]).encode()).hexdigest()[:8], 16) % 2 for i in tr["idx"]])
        ytr = np.full(len(tr["idx"]), -1); ytr[tr["k"]] = tr["y"]
        for sd in range(a.seeds):
            # head scores: main head -> val/test; 2-fold cross-fit heads -> train rows
            Zv, Zt = head_scores(sd, {k: dict(v, T=v["T"][v["k"]]) for k, v in parts.items()}, [va["T"], te["T"]])
            Zl = [None] * len(tr["idx"])
            for f in (0, 1):
                fit = (grp != f) & tr["k"]; pk = {"train": dict(tr, T=tr["T"][fit], y=ytr[fit]), "val": dict(va, T=va["T"][va["k"]], y=va["y"][va["k"]]), "test": dict(te, T=te["T"][te["k"]])}
                Zf, = head_scores(sd, pk, [tr["T"][grp == f]])
                for r, z in zip(np.where(grp == f)[0], Zf): Zl[r] = z
            Ztr = np.array(Zl); Zs = {"train": Ztr, "val": Zv, "test": Zt}
            # delta from VAL hit1 (as a12)
            def comb(k, dl):
                P = parts[k]; pl = [pool[i] for i in P["idx"]]; return arms.combine(pl, P["cands"], Zs[k], names, dl)
            def hit(rk, P): return np.array([bool(r[:1] and r[0] in {names[j] for j in g}) for r, g in zip(rk, P["allp"])]) & (P["lab"] >= 0)
            dl = max(arms.DELTAS, key=lambda d: (hit(comb("val", d)[0], va)[va["lab"] >= 0].mean(), -d))
            D = {}
            for k in ("train", "val", "test"):
                P = parts[k]; rk, tp, _ = comb(k, dl); ok = hit(rk, P); Zc = np.clip(Zs[k], -arms.CLIP, arms.CLIP); Zsort = np.sort(Zc, 1)[:, ::-1]; ex = np.exp(Zc - Zc.max(1, keepdims=True)); pr = ex / ex.sum(1, keepdims=True)
                D[k] = dict(idx=P["idx"], plus=P["lab"] >= 0, ok=ok, c1=np.asarray(tp), htop=Zsort[:, 0], hmarg=Zsort[:, 0] - Zsort[:, 1], hent=-(pr * np.log(pr + 1e-9)).sum(1),
                            a0=np.array([pool[i][0][1] if pool[i] else -9.0 for i in P["idx"]]), a0m=np.array([(pool[i][0][1] - pool[i][1][1]) if len(pool[i]) > 1 else (pool[i][0][1] if pool[i] else 0.0) for i in P["idx"]]),
                            psz=np.array([len(pool[i]) for i in P["idx"]], float))
            c1g = np.full(n, np.nan)
            for k in D: c1g[D[k]["idx"]] = D[k]["c1"]
            def feats(d):
                idx = d["idx"]; prv = np.array([i > 0 and sid[i - 1] == sid[i] for i in idx]); pi = np.maximum(idx - 1, 0)
                pn = np.where(prv, (lab[pi] < 0).astype(float), 0.0); pc = np.where(prv & ~np.isnan(c1g[pi]), np.nan_to_num(c1g[pi]), 0.0)
                return np.column_stack([d["htop"], d["hmarg"], d["hent"], d["c1"], d["a0"], d["a0m"], bm[idx], d["psz"], shape[idx], pn * 0 + pn, prv.astype(float), pc]), X[idx]
            F = {}
            for k in D: F[k] = feats(D[k])
            gate_idx = np.r_[D["train"]["idx"], D["val"]["idx"]]; Xg = np.vstack([F["train"][0], F["val"][0]]); Eg = np.vstack([F["train"][1], F["val"][1]]); yg = np.r_[D["train"]["ok"], D["val"]["ok"]]
            # reorder: shape cols come before prev cols in feats; FN order fix below
            pca = PCA(16, random_state=sd).fit(Eg); A = lambda k: np.hstack([F[k][0], pca.transform(F[k][1])]); Ag, At = np.hstack([Xg, pca.transform(Eg)]), A("test")
            sg = np.array([sid[i] for i in gate_idx]); us = np.unique(sg); rs = np.random.RandomState(sd); fold_of = dict(zip(us, rs.permutation(len(us)) % 5)); fo = np.array([fold_of[s] for s in sg])
            dg = dict(plus=np.r_[D["train"]["plus"], D["val"]["plus"]], ok=yg, c1=np.r_[D["train"]["c1"], D["val"]["c1"]], a0=None, margin=None)
            res = {"delta": dl, "n_gate": int(len(yg))}
            # C1 baselines: val-fit (a12 protocol) and cv-fit (thr from cross-fitted train+val pool)
            dt = dict(plus=D["test"]["plus"], ok=D["test"]["ok"], c1=D["test"]["c1"], a0=D["test"]["a0"], margin=D["test"]["hmarg"]); dv = dict(plus=D["val"]["plus"], ok=D["val"]["ok"], c1=D["val"]["c1"], a0=D["val"]["a0"], margin=D["val"]["hmarg"])
            res["C1"] = a12.curve("C1", dv, dt); res["C1cv"] = a12.curve("C1", dict(dg, a0=None), dt)
            for mn, hps in MODELS.items():
                best = None
                for hp in hps:
                    oof = np.zeros(len(yg))
                    for f in range(5):
                        tri, tei = fo != f, fo == f
                        if yg[tri].min() == yg[tri].max(): oof[tei] = float(yg[tri][0]); continue
                        sc = StandardScaler().fit(Ag[tri]); m = mk(mn, hp, sd).fit(sc.transform(Ag[tri]), yg[tri]); oof[tei] = m.predict_proba(sc.transform(Ag[tei]))[:, 1]
                    do = dict(dg, c1=oof); aurc = float(a12.interp(a12.curve("C1", do, do)).mean())
                    if best is None or aurc < best[0]: best = (aurc, hp, oof)
                aurc, hp, oof = best; sc = StandardScaler().fit(Ag); m = mk(mn, hp, sd).fit(sc.transform(Ag), yg); pt = m.predict_proba(sc.transform(At))[:, 1]
                do = dict(dg, c1=oof); dte = dict(dt, c1=pt); res[mn] = a12.curve("C1", do, dte); res[mn + "_hp"] = hp; res[mn + "_oof_aurc"] = aurc
                res[mn + "_auroc"] = float(roc_auc_score(D["test"]["ok"], pt)) if len(set(D["test"]["ok"])) > 1 else float("nan")
                res[mn + "_pt"] = (m, sc, pt)
            R[sp][sd] = (res, At, D["test"]["ok"], dt)
            meta[sp] = {"gate_rows": int(len(yg)), "gate_pos": float(yg.mean()), "test": int(len(D["test"]["ok"])), "test_pos": float(D["test"]["ok"].mean())}
            print(sp, sd, "delta", dl, {k: round(float(a12.interp(res[k]).mean()), 4) for k in ("C1", "C1cv", "LR", "GBM", "MLP")}, round(time.time() - t0), flush=True)
    ARMS = ["C1", "C1cv", "LR", "GBM", "MLP"]; agg = {}
    for sp in R:
        agg[sp] = {}
        for arm in ARMS:
            Ar = np.array([a12.interp(R[sp][s][0][arm]).mean() for s in R[sp]]); ra = {str(c): [float(np.mean([a12.at(R[sp][s][0][arm], c) for s in R[sp]])), float(np.std([a12.at(R[sp][s][0][arm], c) for s in R[sp]]))] for c in (0.2, 0.3, 0.5)}
            ex = [min(R[sp][s][0][arm], key=lambda p: abs(p["cov"] - 0.3)) for s in R[sp]]
            agg[sp][arm] = dict(aurc=[float(Ar.mean()), float(Ar.std())], risk_at=ra, op={k: float(np.nanmean([e[k] for e in ex])) for k in ("cov", "risk", "full", "far", "fabs")})
            if arm in MODELS: agg[sp][arm]["auroc"] = float(np.nanmean([R[sp][s][0][arm + "_auroc"] for s in R[sp]])); agg[sp][arm]["oof_aurc"] = float(np.mean([R[sp][s][0][arm + "_oof_aurc"] for s in R[sp]]))
    best = min(MODELS, key=lambda m: np.mean([agg[sp][m]["aurc"][0] for sp in agg]))
    vd = {}
    for m in MODELS:
        v = {}
        for sp in agg:
            base = min(("C1", "C1cv"), key=lambda x: agg[sp][x]["aurc"][0]); sdv = max(agg[sp][base]["aurc"][1], agg[sp][m]["aurc"][1]); gain = agg[sp][base]["aurc"][0] - agg[sp][m]["aurc"][0]
            v[sp] = dict(base=base, gain=float(gain), sd=float(sdv), ok=bool(gain > sdv))
        v["risk3_time"] = agg["time"][m]["risk_at"]["0.3"][0]; v["pass"] = bool(all(v[sp]["ok"] for sp in agg) and v["risk3_time"] < 0.55); vd[m] = v
    # permutation importance (AUROC drop on TEST) for best gate, seed 0, 5 perms, qemb grouped
    imp = {}
    for sp in R:
        res, At, ok, dt = R[sp][0]; m, sc, pt = res[best + "_pt"]; base = roc_auc_score(ok, pt); rs = np.random.RandomState(0); groups = {f: [i] for i, f in enumerate(FN[:17])} | {"qemb(16)": list(range(17, 33))}
        imp[sp] = {}
        for g, cols in groups.items():
            d_ = []
            for _ in range(5):
                Ap = At.copy(); perm = rs.permutation(len(Ap)); Ap[:, cols] = At[perm][:, cols]; d_.append(base - roc_auc_score(ok, m.predict_proba(sc.transform(Ap))[:, 1]))
            imp[sp][g] = float(np.mean(d_))
    top = sorted({g: np.mean([imp[sp][g] for sp in imp]) for g in imp["time"]}.items(), key=lambda kv: -kv[1])
    json.dump({"meta": meta, "agg": agg, "verdict": vd, "best_gate": best, "importance": imp, "top_features": top[:8], "hp": {sp: {s: {m: R[sp][s][0][m + "_hp"] for m in MODELS} for s in R[sp]} for sp in R}}, open(a.out, "w"), indent=1)
    f = lambda v: f"{v[0]:.3f}+-{v[1]:.3f}"
    L = ["# A14 learned abstention gate (2026-10-30)", "", "Hypothesis: a learned gate (P(accept) from head score/margin/entropy, A0 top+margin, BM25 top, pool size, prompt shape, prev-row none/conf, query emb PCA16) lowers selective risk vs C1 (head-score thr). Pre-registered pass: AURC < best C1 by > sd on time AND session, and risk@.3 time < .55.", "",
         f"Setup: a14_abstain.py, a12_risk curve/AURC machinery, d256b2 head, K=16, {a.seeds} seeds. Target = B1 top-1 in gold. Gate rows = TRAIN+VAL; TRAIN head scores cross-fitted (2-fold by sid, heads on half train); hparams (2 per model) + thresholds from grouped 5-fold OOF by sid; final gate refit on all gate rows, thr applied to TEST. C1 = a12 protocol (VAL-fit thr); C1cv = same score, thr from the pooled cross-fitted rows (fair to the k-fold fix). n: {meta}", "",
         "| arm | time AURC | session AURC | time risk@.2 | @.3 | @.5 | session risk@.2 | @.3 | @.5 | AUROC t/s |", "|---|---|---|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        t, s = agg["time"][arm], agg["session"][arm]
        L.append(f"| {arm} | {f(t['aurc'])} | {f(s['aurc'])} | " + " | ".join(f(t["risk_at"][c]) for c in ("0.2", "0.3", "0.5")) + " | " + " | ".join(f(s["risk_at"][c]) for c in ("0.2", "0.3", "0.5")) + " | " + (f"{t['auroc']:.3f}/{s['auroc']:.3f}" if arm in MODELS else "-") + " |")
    L += ["", "Operating point nearest coverage .3 (TEST, seed mean): false activation = accepted among SKILL-, false abstention = abstained among SKILL+.", "", "| arm | split | cov | sel. risk | full err | false activation | false abstention |", "|---|---|---|---|---|---|---|"]
    for sp in agg:
        for arm in ARMS: x = agg[sp][arm]["op"]; L.append(f"| {arm} | {sp} | " + " | ".join(f"{x[k]:.3f}" for k in ("cov", "risk", "full", "far", "fabs")) + " |")
    L += ["", f"Best gate (mean test AURC): {best}. Permutation importance (TEST AUROC drop, seed 0, 5 perms, mean of splits), top 8:", ""] + [f"- {g}: {v:.4f} (time {imp['time'][g]:.4f}, session {imp['session'][g]:.4f})" for g, v in top[:8]]
    L += ["", "## Verdict", "", "SUPPORTED" if vd[best]["pass"] else "NOT SUPPORTED (best gate " + best + ")", ""]
    for m, v in vd.items(): L.append(f"- {m}: time gain {v['time']['gain']:+.4f} vs sd {v['time']['sd']:.4f} ({v['time']['base']}), session gain {v['session']['gain']:+.4f} vs sd {v['session']['sd']:.4f} ({v['session']['base']}), risk@.3 time {v['risk3_time']:.3f} -> {'PASS' if v['pass'] else 'fail'}")
    L += ["", "Caveats: train-row head scores come from half-train heads (weaker than prod head); session split test pos rate low so AURC ceilings near .87-.9; hparam choice = 2 settings per model; not committed."]
    open(a.md, "w", encoding="utf-8").write("\n".join(L) + "\n"); print("\n".join(L)); print("done", round(time.time() - t0))


if __name__ == "__main__":
    main()
