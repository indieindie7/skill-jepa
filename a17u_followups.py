"""A17u follow-ups: kernel / low-rank per-session residuals, decoupled + rank fusion. Spec: notes/a17-user-profile-20261031.md section 5.
Out: data/a17u-fu-20261031.json + table appended to the spec."""
import argparse, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import skilljepa as sj
from a17_user_profile import norm, allp_sub, table_probs

ARMS = ["T0", "T1", "T2d", "T2r", "T3", "T3k", "T3lr", "T5", "T3perm", "T3kperm"]

def ridge_lowrank(Pp, Rr, x, r=8, lam=1.0):
    """fit W in top-r PCA dims of past preds Pp (m,d) -> residuals Rr (m,d); return delta for x (d,)."""
    mu = Pp.mean(0); Z = Pp - mu; U, S, Vt = np.linalg.svd(Z, full_matrices=False); V = Vt[:r].T
    Zr = Z @ V; W = np.linalg.solve(Zr.T @ Zr + lam * np.eye(Zr.shape[1]), Zr.T @ (Rr - Rr.mean(0)))
    return Rr.mean(0) + ((x - mu) @ V) @ W

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--min", type=int, default=30)
    ap.add_argument("--cold", type=int, default=10); ap.add_argument("--w", type=float, default=0.3); ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--rank", type=int, default=8); ap.add_argument("--ctx", default="prevskill"); ap.add_argument("--out", default=os.path.join(sj.D, "a17u-fu-20261031.json"))
    ap.add_argument("--md", default=os.path.join(HERE, "notes", "a17-user-profile-20261031.md")); a = ap.parse_args()
    X, Y, lab, allp, sid, names, plab = sj.data(a.ctx, "all"); K = len(names)
    pairs, _, _ = sj.load(); t = np.array([p["t"] for p in pairs])
    cnt = {s: int(((sid == s) & (lab >= 0)).sum()) for s in np.unique(sid)}
    evs = sorted([s for s, c in cnt.items() if c >= a.min], key=lambda s: -cnt[s]); cls = np.where(lab >= 0, lab, K); per = []
    for seed in range(a.seeds):
        for s in evs:
            tr = np.where(sid != s)[0]; te = np.where(sid == s)[0]; te = te[np.argsort(t[te], kind="stable")]
            P = sj.fit(X[tr], Y, lab[tr], seed=seed); Etr = sj.energies(P, X[tr], Y); tau0 = sj.tune_tau(Etr, lab[tr], allp_sub(allp, tr))
            pg = np.bincount(cls[tr], minlength=K + 1) + 1.0; pg = pg / pg.sum()
            pcls_tr = np.where(plab[tr] >= 0, plab[tr], K); Tg = np.zeros((K + 1, K + 1)); np.add.at(Tg, (pcls_tr, cls[tr]), 1.0)
            z = np.zeros((K + 1, K + 1)); zc = np.zeros(K + 1)
            Pt = np.stack([table_probs(pcls_tr[i], z, zc, Tg, pg) for i in range(len(tr))])
            rk_tr = np.argsort(np.argsort(-Pt[:, :K], 1), 1) / K
            tau_r = sj.tune_tau(Etr + a.w * rk_tr, lab[tr], allp_sub(allp, tr))
            pred, _ = P.fwd(X[te]); E0 = 1 - pred @ Y.T
            ts = np.zeros((K + 1, K + 1)); cs = np.zeros(K + 1); prev_c = K; nl = 0; rows = {ar: [] for ar in ARMS}
            rng = np.random.default_rng(seed + 1000); li = [j for j, i in enumerate(te) if lab[i] >= 0]; pl = np.array([lab[te[j]] for j in li]); rng.shuffle(pl); permlab = {j: int(l) for j, l in zip(li, pl)}
            Pp, Rr, Rp = [], [], []
            for j, i in enumerate(te):
                L = bool(lab[i] >= 0); bucket = "cold" if nl < a.cold else "warm"; m = len(Pp); sc = m / (m + 5.0)
                pt = table_probs(prev_c, ts, cs, Tg, pg); lp = a.w * np.log(pt[:K] + 1e-9); rk = np.argsort(np.argsort(-pt[:K])) / K
                if m:
                    PP = np.stack(Pp); RR = np.stack(Rr); dmean = RR.mean(0); RRp = np.stack(Rp); dmp = RRp.mean(0)
                    wk = np.exp((PP @ pred[j] - (PP @ pred[j]).max()) / a.temp); wk /= wk.sum(); dk = wk @ RR; dkp = wk @ RRp
                    dlr = ridge_lowrank(PP, RR, pred[j], a.rank) if m >= a.rank else dmean
                else: dmean = dk = dlr = dmp = dkp = np.zeros(Y.shape[1])
                E3 = 1 - norm(pred[j] + sc * dmean) @ Y.T; E3k = 1 - norm(pred[j] + sc * dk) @ Y.T; E3lr = 1 - norm(pred[j] + sc * dlr) @ Y.T
                E3p = 1 - norm(pred[j] + sc * dmp) @ Y.T; E3kp = 1 - norm(pred[j] + sc * dkp) @ Y.T
                ab0 = bool(E0[j].min() > tau0)
                outs = {"T0": (int(E0[j].argmin()), ab0), "T1": (int(pt[:K].argmax()), bool(pt.argmax() == K)),
                        "T2d": (int((E0[j] - lp).argmin()), ab0), "T2r": (int((E0[j] + a.w * rk).argmin()), bool((E0[j] + a.w * rk).min() > tau_r)),
                        "T3": (int(E3.argmin()), bool(E3.min() > tau0)), "T3k": (int(E3k.argmin()), bool(E3k.min() > tau0)),
                        "T3lr": (int(E3lr.argmin()), bool(E3lr.min() > tau0)), "T5": (int((E3k - lp).argmin()), ab0),
                        "T3perm": (int(E3p.argmin()), bool(E3p.min() > tau0)), "T3kperm": (int(E3kp.argmin()), bool(E3kp.min() > tau0))}
                for ar, (top, ab) in outs.items(): rows[ar].append((bucket, L, (top in allp[i]) if L else None, ab))
                c = cls[i]; ts[prev_c, c] += 1; cs[c] += 1; prev_c = c
                if L: Pp.append(pred[j]); Rr.append(Y[lab[i]] - pred[j]); Rp.append(Y[permlab[j]] - pred[j]); nl += 1
            rec = {"seed": seed, "sid": s, "n": len(te), "labelled": int(nl)}
            for ar in ARMS:
                for b in ("cold", "warm", "all"):
                    rr = [x for x in rows[ar] if b == "all" or x[0] == b]; lb = [x for x in rr if x[1]]; nn = [x for x in rr if not x[1]]
                    rec[f"{ar}.{b}"] = {"hit1": float(np.mean([x[2] for x in lb])) if lb else None,
                                        "acc_abst": float(np.mean([(not x[3] and x[2]) if x[1] else x[3] for x in rr])) if rr else None,
                                        "false_route_none": float(np.mean([not x[3] for x in nn])) if nn else None, "n_lab": len(lb), "n": len(rr)}
            per.append(rec); print(f"seed {seed} sid {s} " + " ".join(f"{ar}={rec[ar + '.warm']['hit1']:.3f}" for ar in ARMS), flush=True)
    def agg(ar, b, k):
        ms = [float(np.mean([r[f"{ar}.{b}"][k] for r in per if r["seed"] == sd and r[f"{ar}.{b}"][k] is not None])) for sd in range(a.seeds)]
        return float(np.mean(ms)), float(np.std(ms))
    summ = {ar: {b: {k: agg(ar, b, k) for k in ("hit1", "acc_abst", "false_route_none")} for b in ("cold", "warm", "all")} for ar in ARMS}
    w = {ar: summ[ar]["warm"]["hit1"][0] * 100 for ar in ARMS}; acc = {ar: summ[ar]["all"]["acc_abst"][0] * 100 for ar in ARMS}; fr = {ar: summ[ar]["all"]["false_route_none"][0] * 100 for ar in ARMS}
    verdict = [f"{ar} SUPPORTED (latent beats table)" for ar in ("T3k", "T3lr", "T5") if w[ar] >= w["T1"] + 2.0 and acc[ar] >= acc["T0"]]
    verdict += [f"{ar} FUSION FIXED" for ar in ("T2d", "T2r") if w[ar] >= w["T1"] + 2.0 and fr[ar] <= fr["T1"] + 2.0]
    verdict = verdict or ["TABLE STANDS: no follow-up arm meets its rule"]
    d = w["T3k"] - w["T3kperm"]; verdict.append(("SUBTEXT" if d >= 2.0 else "FREQUENCY") + f": T3k - T3kperm = {d:+.1f} warm hit1 (T3 - T3perm = {w['T3'] - w['T3perm']:+.1f})")
    json.dump({"args": vars(a), "summary": summ, "per": per, "verdict": verdict}, open(a.out, "w", encoding="utf-8"), indent=1, default=float)
    f = lambda ar, b, k: "%.1f +- %.1f" % (summ[ar][b][k][0] * 100, summ[ar][b][k][1] * 100)
    lines = ["", f"### Follow-up run {time.strftime('%Y-%m-%d %H:%M')} (seeds={a.seeds}, w={a.w}, temp={a.temp}, rank={a.rank})", "",
             "| arm | cold hit1 | warm hit1 | all acc_abst | all false_route_none |", "|---|---|---|---|---|"]
    lines += [f"| {ar} | {f(ar, 'cold', 'hit1')} | {f(ar, 'warm', 'hit1')} | {f(ar, 'all', 'acc_abst')} | {f(ar, 'all', 'false_route_none')} |" for ar in ARMS]
    lines += ["", "Verdict: " + "; ".join(verdict), "", "Per-session warm hit1 seed 0 (" + "/".join(ARMS) + "): " + ", ".join(
        f"{r['sid']} " + "/".join(f"{(r[ar + '.warm']['hit1'] or 0) * 100:.0f}" for ar in ARMS) for r in per if r["seed"] == 0)]
    open(a.md, "a", encoding="utf-8").write("\n".join(lines) + "\n"); print("\n".join(lines))

if __name__ == "__main__": main()
