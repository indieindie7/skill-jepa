"""A17: per-session latent profile (JEPA residual) vs per-session transition table. Prequential leave-one-session-out.
Spec: notes/a17-user-profile-20261031.md. Out: data/a17-20261031.json + results appended to the spec."""
import argparse, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import skilljepa as sj

ARMS = ["T0", "T1", "T1perm", "T2", "T3", "T4"]

def norm(v): return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)

def allp_sub(allp, idx): return [allp[i] for i in idx]

def table_probs(prev, ts, cs, Tg, pg, a=2.0, b=2.0):
    """p(class | prev) over K+1 classes (last = none). ts: session transitions (K+1,K+1), cs: session class counts."""
    ps = (cs + b * pg) / (cs.sum() + b)
    tg = Tg[prev]; pgt = (tg + a * pg) / (tg.sum() + a)
    row = ts[prev]; return (row + a * (0.5 * ps + 0.5 * pgt)) / (row.sum() + a)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--min", type=int, default=30)
    ap.add_argument("--cold", type=int, default=10); ap.add_argument("--w", type=float, default=0.3); ap.add_argument("--c", type=float, default=1.0)
    ap.add_argument("--ctx", default="prevskill"); ap.add_argument("--out", default=os.path.join(sj.D, "a17-20261031.json"))
    ap.add_argument("--md", default=os.path.join(HERE, "notes", "a17-user-profile-20261031.md")); a = ap.parse_args()
    X, Y, lab, allp, sid, names, plab = sj.data(a.ctx, "all"); K = len(names)
    pairs, _, _ = sj.load(); t = np.array([p["t"] for p in pairs])
    cnt = {s: int(((sid == s) & (lab >= 0)).sum()) for s in np.unique(sid)}
    evs = sorted([s for s, c in cnt.items() if c >= a.min], key=lambda s: -cnt[s]); print("eval sessions", [(s, cnt[s]) for s in evs], flush=True)
    cls = np.where(lab >= 0, lab, K); per = []
    for seed in range(a.seeds):
        for s in evs:
            tr = np.where(sid != s)[0]; te = np.where(sid == s)[0]; te = te[np.argsort(t[te], kind="stable")]
            P = sj.fit(X[tr], Y, lab[tr], seed=seed); Etr = sj.energies(P, X[tr], Y); tau0 = sj.tune_tau(Etr, lab[tr], allp_sub(allp, tr))
            pg = np.bincount(cls[tr], minlength=K + 1) + 1.0; pg = pg / pg.sum()
            pcls_tr = np.where(plab[tr] >= 0, plab[tr], K); Tg = np.zeros((K + 1, K + 1)); np.add.at(Tg, (pcls_tr, cls[tr]), 1.0)
            z = np.zeros((K + 1, K + 1)); zc = np.zeros(K + 1)
            Pt = np.stack([table_probs(pcls_tr[i], z, zc, Tg, pg) for i in range(len(tr))])
            tau2 = sj.tune_tau(Etr - a.w * np.log(Pt[:, :K] + 1e-9), lab[tr], allp_sub(allp, tr))
            pred, _ = P.fwd(X[te]); E0 = 1 - pred @ Y.T
            rng = np.random.default_rng(seed + 1000); perm_lab = cls[te].copy(); rng.shuffle(perm_lab)
            ts = np.zeros((K + 1, K + 1)); cs = np.zeros(K + 1); tsp = np.zeros((K + 1, K + 1)); csp = np.zeros(K + 1)
            dsum = np.zeros(Y.shape[1]); m = 0; prev_c = K; prev_cp = K; nl = 0; rows = {ar: [] for ar in ARMS}
            for j, i in enumerate(te):
                L = bool(lab[i] >= 0); bucket = "cold" if nl < a.cold else "warm"
                pt = table_probs(prev_c, ts, cs, Tg, pg); ptp = table_probs(prev_cp, tsp, csp, Tg, pg)
                sc = m / (m + 5.0); p3 = norm(pred[j] + a.c * sc * dsum / max(m, 1)); E3 = 1 - p3 @ Y.T
                lp = a.w * np.log(pt[:K] + 1e-9); E2 = E0[j] - lp; E4 = E3 - lp
                en = lambda E, tau: (int(E.argmin()), bool(E.min() > tau))
                tb = lambda p: (int(p[:K].argmax()), bool(p.argmax() == K))
                outs = {"T0": en(E0[j], tau0), "T1": tb(pt), "T1perm": tb(ptp), "T2": en(E2, tau2), "T3": en(E3, tau0), "T4": en(E4, tau2)}
                for ar, (top, ab) in outs.items(): rows[ar].append((bucket, L, (top in allp[i]) if L else None, ab))
                c = cls[i]; ts[prev_c, c] += 1; cs[c] += 1; prev_c = c
                cp = perm_lab[j]; tsp[prev_cp, cp] += 1; csp[cp] += 1; prev_cp = cp
                if L: dsum += Y[lab[i]] - pred[j]; m += 1; nl += 1
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
    w = {ar: summ[ar]["warm"]["hit1"][0] * 100 for ar in ARMS}; cold = {ar: summ[ar]["cold"]["hit1"][0] * 100 for ar in ARMS}
    verdict = []
    if w["T3"] - max(w["T1"], w["T2"]) >= 2.0 and cold["T3"] >= cold["T0"] - 1.0: verdict.append("SUPPORTED: per-session latent correction beats table + fusion warm, no cold harm")
    elif max(w["T1"], w["T2"]) >= w["T3"]: verdict.append("TABLE WINS: per-session table/fusion >= latent profile on warm hit1")
    else: verdict.append("INCONCLUSIVE")
    if abs(w["T1perm"] - w["T1"]) <= 1.0: verdict.append("FINGERPRINT WARNING: shuffled history ties the table")
    json.dump({"args": vars(a), "eval_sessions": [(s, cnt[s]) for s in evs], "summary": summ, "per": per, "verdict": verdict, "names": names},
              open(a.out, "w", encoding="utf-8"), indent=1, default=float)
    f = lambda ar, b, k: "%.1f +- %.1f" % (summ[ar][b][k][0] * 100, summ[ar][b][k][1] * 100)
    lines = ["", f"### Run {time.strftime('%Y-%m-%d %H:%M')} (seeds={a.seeds}, ctx={a.ctx}, w={a.w}, c={a.c}, cold={a.cold}, min={a.min})", "",
             "| arm | cold hit1 | warm hit1 | all acc_abst | all false_route_none |", "|---|---|---|---|---|"]
    lines += [f"| {ar} | {f(ar, 'cold', 'hit1')} | {f(ar, 'warm', 'hit1')} | {f(ar, 'all', 'acc_abst')} | {f(ar, 'all', 'false_route_none')} |" for ar in ARMS]
    lines += ["", "Verdict: " + "; ".join(verdict), "", "Per-session warm hit1 seed 0 (" + "/".join(ARMS) + "): " + ", ".join(
        f"{r['sid']} " + "/".join(f"{(r[ar + '.warm']['hit1'] or 0) * 100:.0f}" for ar in ARMS) for r in per if r["seed"] == 0)]
    open(a.md, "a", encoding="utf-8").write("\n".join(lines) + "\n"); print("\n".join(lines))

if __name__ == "__main__": main()
