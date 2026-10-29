"""a12_risk.py: ledger A12 selective risk-coverage. C0 never abstain; C1 thr on B1 combined top score (head); C1b thr on A0 top score; C3 2-D grid (A0 thr x head margin thr).
Error = execute wrong | execute on SKILL- | abstain on SKILL+. Selective risk(c) = error among ACCEPTED (full err incl. false abstain also reported). thr fit on VAL, report TEST; oracle = fit on TEST (flagged).
  C:/llm/qlora/venv/Scripts/python.exe a12_risk.py [--seeds 3]   (CPU)"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse, json, sys, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import arms, shrink, a11_state, avb_rerank as ar
torch.set_num_threads(int(os.environ.get("SHRINK_THREADS", "3")))
GRID = np.round(np.arange(0.1, 1.0001, 0.05), 3); QS = np.linspace(0, 1, 26)
ARMS4 = ("C0", "C1", "C1b", "C3")


def row_stats(sp_, names, Zv, Zt):
    per = {}
    for dl in arms.DELTAS:
        rkv, tpv, _ = arms.combine(sp_["pv"], sp_["parts"]["val"]["cands"], Zv, names, dl); rkt, tpt, _ = arms.combine(sp_["pt"], sp_["parts"]["test"]["cands"], Zt, names, dl)
        h = np.mean([bool(r[:1] and r[0] in {names[j] for j in g}) for r, g, l in zip(rkv, sp_["parts"]["val"]["allp"], sp_["parts"]["val"]["lab"]) if l >= 0]); per[dl] = (h, rkv, tpv, rkt, tpt)
    dl = max(arms.DELTAS, key=lambda d: (per[d][0], -d)); _, rkv, tpv, rkt, tpt = per[dl]; out = {}
    for k, rk, tp, Z, pl in (("val", rkv, tpv, Zv, sp_["pv"]), ("test", rkt, tpt, Zt, sp_["pt"])):
        P = sp_["parts"][k]; plus = P["lab"] >= 0
        ok = np.array([bool(r[:1] and r[0] in {names[j] for j in g}) for r, g in zip(rk, P["allp"])]) & plus
        Zc = np.sort(np.clip(Z, -arms.CLIP, arms.CLIP), 1)[:, ::-1]
        out[k] = dict(plus=plus, ok=ok, a0=np.array([p[0][1] if p else -9.0 for p in pl]), c1=np.asarray(tp), margin=Zc[:, 0] - Zc[:, 1])
    out["delta"] = dl; return out


def ev(acc, d):
    plus, ok = d["plus"], d["ok"]; n = len(acc); na = acc.sum()
    return dict(cov=float(na / n), risk=float((~ok[acc]).mean()) if na else 0.0, full=float(np.where(acc, ~ok, plus).mean()),
                hit1_acc=float(ok[acc & plus].mean()) if (acc & plus).any() else float("nan"),
                far=float(acc[~plus].mean()) if (~plus).any() else float("nan"), fabs=float((~acc[plus]).mean()) if plus.any() else float("nan"))


def thr_sets(arm, dv):
    if arm == "C1": return [(t,) for t in np.quantile(dv["c1"], QS[:-1])]
    if arm == "C1b": return [(t,) for t in np.quantile(dv["a0"], QS[:-1])]
    if arm == "C3": return [(a, m) for a in np.quantile(dv["a0"], QS[:-1]) for m in np.quantile(dv["margin"], QS[:-1])]
    return [()]


def accept(arm, th, d):
    if arm == "C0": return np.ones(len(d["ok"]), bool)
    if arm == "C1": return d["c1"] >= th[0]
    if arm == "C1b": return d["a0"] >= th[0]
    return (d["a0"] >= th[0]) & (d["margin"] >= th[1])


def curve(arm, fit, rep):
    """per target coverage c: config with min fit-risk s.t. fit-cov >= c; evaluate on rep."""
    E = [(th, ev(accept(arm, th, fit), fit)) for th in thr_sets(arm, fit)]; pts = []
    for c in GRID:
        cand = [(th, e) for th, e in E if e["cov"] >= c - 1e-9]
        if not cand: continue
        th, _ = min(cand, key=lambda x: (x[1]["risk"], -x[1]["cov"])); pts.append(ev(accept(arm, th, rep), rep) | {"target": float(c), "th": [float(x) for x in th]})
    return pts


def interp(pts):
    pts = sorted(pts, key=lambda p: p["cov"]); return np.interp(GRID, [p["cov"] for p in pts], [p["risk"] for p in pts])


def at(pts, c):
    return float(interp(pts)[int(np.argmin(abs(GRID - c)))])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--out", default=os.path.join(arms.D, "a12-20261029.json")); ap.add_argument("--md", default=os.path.join(HERE, "notes", "a12-risk-20261029.md")); a = ap.parse_args()
    t0 = time.time(); S, names, tokf, extras = a11_state.build(a); R = {sp: {} for sp in S}; orig = ar.run; meta = {}
    for sp in S:
        sp_ = S[sp]; parts = sp_["parts"]
        meta[sp] = {"val": int(parts["val"]["n"]), "test": int(parts["test"]["n"]), "val_plus": int((parts["val"]["lab"] >= 0).sum()), "test_plus": int((parts["test"]["lab"] >= 0).sum())}
        parts_k = {k: dict(v, T=v["T"][v["k"]]) for k, v in parts.items()}
        for sd in range(a.seeds):
            ar.run = lambda s, *args, _sd=sd, **kw: orig(_sd, *args, **kw); ar.Head = lambda din, attn=True, other=False: shrink.SHead(din, 256, 2)
            ms, _ = ar.run(sd, parts_k, True, refit=False); m = ms[0]
            with torch.no_grad(): Zv, Zt = (m(torch.tensor(parts[k]["T"])).numpy()[:, :-1] for k in ("val", "test"))
            ar.run = orig; d = row_stats(sp_, names, Zv, Zt); res = {}
            for arm in ARMS4:
                if arm == "C0": e = ev(accept(arm, (), d["test"]), d["test"]); res[arm] = {"fit": [e], "oracle": [e]}; continue
                res[arm] = {"fit": curve(arm, d["val"], d["test"]), "oracle": curve(arm, d["test"], d["test"])}
            R[sp][sd] = res | {"delta": d["delta"]}
            print(sp, sd, "delta", d["delta"], {arm: round(float(interp(res[arm]["fit"]).mean()), 4) for arm in ARMS4[1:]}, round(time.time() - t0), flush=True)
    agg = {}
    for sp in R:
        agg[sp] = {}
        for arm in ARMS4:
            for mode in ("fit", "oracle"):
                if arm == "C0":
                    c0 = [R[sp][s][arm]["fit"][0] for s in R[sp]]; v = float(np.mean([c["risk"] for c in c0]))
                    agg[sp][f"{arm}/{mode}"] = {"aurc": [v, float(np.std([c["risk"] for c in c0]))], "risk_at": {str(c): [v, 0.0] for c in (0.2, 0.3, 0.5)}, "at_cov~.3": {k: float(np.mean([c[k] for c in c0])) for k in ("cov", "risk", "full", "hit1_acc", "far", "fabs")}}; continue
                A = np.array([interp(R[sp][s][arm][mode]).mean() for s in R[sp]]); ra = {str(c): [float(np.mean([at(R[sp][s][arm][mode], c) for s in R[sp]])), float(np.std([at(R[sp][s][arm][mode], c) for s in R[sp]]))] for c in (0.2, 0.3, 0.5)}
                ex = [min(R[sp][s][arm][mode], key=lambda p: abs(p["cov"] - 0.3)) for s in R[sp]]
                agg[sp][f"{arm}/{mode}"] = {"aurc": [float(A.mean()), float(A.std())], "risk_at": ra, "at_cov~.3": {k: float(np.nanmean([e[k] for e in ex])) for k in ("cov", "risk", "full", "hit1_acc", "far", "fabs")}}
    ok = True; vd = {}
    for sp in agg:
        g = lambda arm: agg[sp][f"{arm}/fit"]["aurc"]; best = min(("C1", "C1b"), key=lambda x: g(x)[0]); sd_ = max(g(best)[1], g("C3")[1]); gain = g(best)[0] - g("C3")[0]
        vd[sp] = dict(best=best, gain=float(gain), sd=float(sd_), pass_=bool(gain > sd_)); ok &= bool(gain > sd_)
    json.dump({"meta": meta, "agg": agg, "verdict": vd, "pass": bool(ok), "grid": GRID.tolist(), "per_seed": {sp: {str(s): {arm: R[sp][s][arm] for arm in ARMS4} | {"delta": R[sp][s]["delta"]} for s in R[sp]} for sp in R}}, open(a.out, "w"), indent=1)
    f = lambda v: f"{v[0]:.3f}+-{v[1]:.3f}"
    L = ["# A12 selective risk-coverage (2026-10-29)", "", "Hypothesis: C3 dual gate (A0 top thr AND head margin thr) gives lower selective risk than C1 (thr on B1 combined head score) / C1b (thr on A0 top score) at matched coverage. Pre-registered pass: C3 AURC lower than best of C1/C1b by > sd on BOTH splits.", "",
         f"Setup: a12_risk.py, a11_state.build machinery (frozen A0, pairs.jsonl rows via sj.data, same time/session splits as arms.py), prod head d256b2, K=16, {a.seeds} seeds. SKILL- = lab<0 (skills==none), SKILL+ otherwise. Route = B1 top-1 (delta picked on VAL hit1) or abstain. Error = execute wrong | execute on SKILL- | abstain on SKILL+. risk(c) = error among ACCEPTED (selective risk); coverage = accepted frac. Head margin = top1-top2 clipped head logit. Per target coverage c in 0.10..1.0 step .05 the config with min VAL risk s.t. VAL cov >= c is applied to TEST; AURC = mean over the coverage grid of TEST risk (interpolated; lower = better). ORACLE rows pick on TEST itself (upper bound, NOT a result). n: {meta}", "",
         "| arm | time AURC | session AURC | time risk@.2 | @.3 | @.5 | session risk@.2 | @.3 | @.5 |", "|---|---|---|---|---|---|---|---|---|"]
    for arm in ARMS4:
        for mode in (("fit", "oracle") if arm != "C0" else ("fit",)):
            t, s = agg["time"][f"{arm}/{mode}"], agg["session"][f"{arm}/{mode}"]
            L.append(f"| {arm}{' (ORACLE on test)' if mode == 'oracle' else ''} | {f(t['aurc'])} | {f(s['aurc'])} | " + " | ".join(f(t["risk_at"][c]) for c in ("0.2", "0.3", "0.5")) + " | " + " | ".join(f(s["risk_at"][c]) for c in ("0.2", "0.3", "0.5")) + " |")
    L += ["", "Operating point nearest coverage .3 (VAL-fit thresholds, test, seed mean). full err includes false abstentions.", "", "| arm | split | cov | sel. risk | full err | SKILL+ hit1 given accepted | false activation (SKILL-) | false abstention (SKILL+) |", "|---|---|---|---|---|---|---|---|"]
    for sp in agg:
        for arm in ARMS4:
            x = agg[sp][f"{arm}/fit"]["at_cov~.3"]; L.append(f"| {arm} | {sp} | " + " | ".join(f"{x[k]:.3f}" for k in ("cov", "risk", "full", "hit1_acc", "far", "fabs")) + " |")
    L += ["", "## Verdict", "", "SUPPORTED" if ok else "NOT SUPPORTED", "", *[f"- {sp}: best single = {v['best']}, C3 gain {v['gain']:+.4f} vs sd {v['sd']:.4f} -> {'pass' if v['pass_'] else 'fail'}" for sp, v in vd.items()],
          "", "Caveats: session VAL is tiny (see n), so VAL-fit thresholds there are noisy; C3 has 650 grid points vs 25 for C1/C1b (more VAL-overfit capacity). C0 cells = its single-point risk. Not committed."]
    open(a.md, "w", encoding="utf-8").write("\n".join(L) + "\n"); print("\n".join(L)); print("done", round(time.time() - t0))


if __name__ == "__main__":
    main()
