"""A6b: per-chat prior as the nudger, offline replay. Spec: notes/context-nudger-PLAN.md line 38. Reuses a17_user_profile (table_probs, norm).
Part A labelled prequential (sids >=30 labels, leave-one-session-out, same protocol as A17u). Part B replay of ALL ops/route.log prompts per sid, no future leak
(a row's label only enters the sid's state 600 s after the row). Out: data/a6b-results.json, notes/a6b-table-raw.md."""
import argparse, json, os, sys, time, math, collections
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import skilljepa as sj
from a17_user_profile import norm, table_probs
WINDOW = 600; TS = 0.05   # softmax temperature for energy -> distribution (predictor training temp)

def ts_(s):
    try: return time.mktime(time.strptime(s, "%Y-%m-%d %H:%M:%S"))
    except ValueError: return None

def softmax(E, T=TS):
    z = -E / T; z = z - z.max(); p = np.exp(z); return p / p.sum()

def kl_bits(p, q): return float(np.sum(p * (np.log2(p + 1e-12) - np.log2(q + 1e-12))))

def ent(c):
    n = sum(c.values()); return -sum(v / n * math.log2(v / n) for v in c.values()) if n else 0.0

def kernel_delta(PP, RR, x, temp):
    s = PP @ x; w = np.exp((s - s.max()) / temp); w /= w.sum(); return w @ RR

def train_stats(lab, plab, tr, K):
    cls = np.where(lab >= 0, lab, K); pg = np.bincount(cls[tr], minlength=K + 1) + 1.0; pg /= pg.sum()
    pc = np.where(plab[tr] >= 0, plab[tr], K); Tg = np.zeros((K + 1, K + 1)); np.add.at(Tg, (pc, cls[tr]), 1.0); return pg, Tg

def parse_route(names):
    use = []
    for l in open(os.path.expanduser("~/.claude/skills/usage.jsonl"), encoding="utf-8"):
        try: j = json.loads(l); t = time.mktime(time.strptime(j["t"], "%Y-%m-%d %H:%M"))
        except (ValueError, KeyError): continue
        use.append((t, j["skill"], j.get("sid", "")))
    idx = {n: i for i, n in enumerate(names)}; rows, seen = [], set()
    for l in open(os.path.join(HERE, "..", "ops", "route.log"), encoding="utf-8"):
        p = l.rstrip("\n").split("\t")
        if len(p) < 4: continue
        t = ts_(p[0])
        if not t or p[3].startswith("<") or len(p[3]) < 8: continue
        sid = p[4] if len(p) > 4 else ""
        if (sid, p[0], p[3]) in seen: continue
        seen.add((sid, p[0], p[3]))
        reads = sorted([(ut, sk) for ut, sk, us in use if 0 <= ut - t <= WINDOW and (not sid or not us or us == sid) and sk in idx])
        cands = [c.split(":")[0] for c in p[2].split(",") if c]
        rows.append(dict(t=t, day=p[0][:10], sid=sid, q=p[3], lab=idx[reads[0][1]] if reads else -1, prod=cands[0] if cands else ""))
    return rows

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--min", type=int, default=30); ap.add_argument("--cold", type=int, default=10)
    ap.add_argument("--w", type=float, default=0.3); ap.add_argument("--temp", type=float, default=0.1); ap.add_argument("--mindayrows", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(sj.D, "a6b-results.json")); a = ap.parse_args()
    X, Y, lab, allp, sid, names, plab = sj.data("prevskill", "all"); K = len(names)
    pairs, _, _ = sj.load(); t = np.array([p["t"] for p in pairs]); cls = np.where(lab >= 0, lab, K)
    cnt = {s: int(((sid == s) & (lab >= 0)).sum()) for s in np.unique(sid)}
    evs = sorted([s for s, c in cnt.items() if c >= a.min], key=lambda s: -cnt[s]); print("eval sids", [(s, cnt[s]) for s in evs], flush=True)
    cache = {}
    def predictor(seed, held):
        k = (seed, held)
        if k not in cache:
            tr = np.where(sid != held)[0] if held else np.arange(len(sid)); t0 = time.time()
            cache[k] = (sj.fit(X[tr], Y, lab[tr], seed=seed), tr); print(f"  fit seed {seed} held {held} {time.time()-t0:.0f}s", flush=True)
        return cache[k]
    # ---- Part A
    AR = ["N0", "N1", "N2"]; per = []
    for seed in range(a.seeds):
        for s in evs:
            P, tr = predictor(seed, s); pg, Tg = train_stats(lab, plab, tr, K)
            te = np.where(sid == s)[0]; te = te[np.argsort(t[te], kind="stable")]; pred, _ = P.fwd(X[te]); E0 = 1 - pred @ Y.T
            ts = np.zeros((K + 1, K + 1)); cs = np.zeros(K + 1); prev_c = K; Pp, Rr = [], []; rows = {ar: [] for ar in AR}
            for j, i in enumerate(te):
                L = bool(lab[i] >= 0); m = len(Pp); bucket = "cold" if m < a.cold else "warm"; sc = m / (m + 5.0)
                pt = table_probs(prev_c, ts, cs, Tg, pg); lp = a.w * np.log(pt[:K] + 1e-9)
                dk = kernel_delta(np.stack(Pp), np.stack(Rr), pred[j], a.temp) if m else np.zeros(Y.shape[1])
                E3k = 1 - norm(pred[j] + sc * dk) @ Y.T
                tops = {"N0": int(E0[j].argmin()), "N1": int((E0[j] - lp).argmin()), "N2": int(pt[:K].argmax()) if bucket == "cold" else int(E3k.argmin())}
                if L:
                    for ar in AR: rows[ar].append((bucket, tops[ar] in allp[i]))
                c = cls[i]; ts[prev_c, c] += 1; cs[c] += 1; prev_c = c
                if L: Pp.append(pred[j]); Rr.append(Y[lab[i]] - pred[j])
            rec = {"seed": seed, "sid": s}
            for ar in AR:
                for b in ("cold", "warm", "all"):
                    r = [x[1] for x in rows[ar] if b == "all" or x[0] == b]; rec[f"{ar}.{b}"] = float(np.mean(r)) if r else None
            per.append(rec); print("A", seed, s, " ".join(f"{ar}={rec[ar+'.warm']:.3f}" for ar in AR), flush=True)
    hitA = {ar: {b: [float(np.mean([r[f"{ar}.{b}"] for r in per if r["seed"] == sd and r[f"{ar}.{b}"] is not None])) for sd in range(a.seeds)] for b in ("cold", "warm", "all")} for ar in AR}
    # ---- Part B
    rows = parse_route(names); print("route rows", len(rows), flush=True)
    ec = os.path.join(sj.D, "a6b-emb.npz"); qs = [r["q"] for r in rows]
    if os.path.exists(ec) and len(np.load(ec)["Xq"]) == len(qs): Xq = np.load(ec)["Xq"]
    else: Xq = sj.embed([sj.Q_PRE + q for q in qs]); np.savez(ec, Xq=Xq)
    bysid = collections.defaultdict(list)
    for i, r in enumerate(rows): bysid[r["sid"]].append(i)
    for s in bysid: bysid[s].sort(key=lambda i: rows[i]["t"])
    pairsids = set(sid.tolist()); B = {ar: {sd: [] for sd in range(a.seeds)} for ar in AR}   # (day, top, kl, bucket)
    base_prod = collections.defaultdict(collections.Counter)
    for s, ix in bysid.items():
        held = s if s in pairsids else ""
        pl = np.full(len(ix), -1)     # causal previous-row label: known only if prev row is >= WINDOW old
        for k in range(1, len(ix)):
            if rows[ix[k-1]]["t"] + WINDOW <= rows[ix[k]]["t"]: pl[k] = rows[ix[k-1]]["lab"]
        prv = np.array([k - 1 if k else -1 for k in range(len(ix))]); Xs = sj.with_ctx(Xq[ix], Y, prv, pl, "prevskill")
        for seed in range(a.seeds):
            P, tr = predictor(seed, held); pg, Tg = train_stats(lab, plab, tr, K); pred, _ = P.fwd(Xs); E0 = 1 - pred @ Y.T
            ts = np.zeros((K + 1, K + 1)); cs = np.zeros(K + 1); Pp, Rr = [], []; absorbed = 0
            for k, i in enumerate(ix):
                while absorbed < k and rows[ix[absorbed]]["t"] + WINDOW <= rows[i]["t"]:   # absorb labels now known
                    lb = rows[ix[absorbed]]["lab"]; c = lb if lb >= 0 else K
                    pb = rows[ix[absorbed-1]]["lab"] if absorbed else -1; pc = pb if pb >= 0 else K
                    ts[pc, c] += 1; cs[c] += 1
                    if lb >= 0: Pp.append(pred[absorbed]); Rr.append(Y[lb] - pred[absorbed])
                    absorbed += 1
                m = len(Pp); sc = m / (m + 5.0); prev_c = int(pl[k]) if pl[k] >= 0 else K
                pt = table_probs(prev_c, ts, cs, Tg, pg); lp = a.w * np.log(pt[:K] + 1e-9); p0 = softmax(E0[k])
                dk = kernel_delta(np.stack(Pp), np.stack(Rr), pred[k], a.temp) if m else np.zeros(Y.shape[1]); E3k = 1 - norm(pred[k] + sc * dk) @ Y.T
                p1 = softmax(E0[k] - lp); cold = m < a.cold; p2 = pt[:K] / pt[:K].sum() if cold else softmax(E3k)
                for ar, p in (("N0", p0), ("N1", p1), ("N2", p2)):
                    B[ar][seed].append((rows[i]["day"], int(p.argmax()), kl_bits(p, p0), "cold" if (ar == "N2" and cold) else "warm"))
        for i in ix: base_prod[rows[i]["day"]][rows[i]["prod"]] += 1
    def daily(ar, seed):
        byd = collections.defaultdict(collections.Counter)
        for d, top, _, _ in B[ar][seed]: byd[d][top] += 1
        ds = [d for d in sorted(byd) if sum(byd[d].values()) >= a.mindayrows]
        return {d: (ent(byd[d]), len(byd[d]), sum(byd[d].values())) for d in ds}
    resB = {}
    for ar in AR:
        e, dc, med, p90, medw, medc, nc = [], [], [], [], [], [], 0
        for sd in range(a.seeds):
            dd = daily(ar, sd); e.append(np.mean([v[0] for v in dd.values()])); dc.append(np.mean([v[1] for v in dd.values()]))
            kl = np.array([x[2] for x in B[ar][sd]]); med.append(np.median(kl)); p90.append(np.percentile(kl, 90))
            klw = np.array([x[2] for x in B[ar][sd] if x[3] == "warm"]); klc = np.array([x[2] for x in B[ar][sd] if x[3] == "cold"])
            medw.append(np.median(klw) if len(klw) else float("nan")); medc.append(np.median(klc) if len(klc) else float("nan")); nc = len(klc)
        resB[ar] = dict(entropy=e, distinct=dc, kl_med=med, kl_p90=p90, kl_med_warm=medw, kl_med_cold=medc, n_cold_rows=nc, n_days=len(dd), n_rows=len(B[ar][0]))
    dd0 = daily("N0", 0); prod_days = {d: (ent(base_prod[d]), len(base_prod[d])) for d in dd0}
    res = dict(args=vars(a), eval_sids=[(s, cnt[s]) for s in evs], hitA=hitA, B=resB, per=per, days={ar: daily(ar, 0) for ar in AR},
               prod_mean_entropy=float(np.mean([v[0] for v in prod_days.values()])), prod_mean_distinct=float(np.mean([v[1] for v in prod_days.values()])))
    f = lambda v: float(np.mean(v)); sd_ = lambda v: float(np.std(v))
    h0 = f(hitA["N2"]["all"]) - f(hitA["N0"]["all"])
    p1 = h0 >= 0.05; p2 = f(resB["N2"]["entropy"]) >= f(resB["N0"]["entropy"]); p3 = f(resB["N2"]["kl_med"]) <= 1.0
    res["pass"] = dict(hit1_gain=h0, hit1=bool(p1), entropy=bool(p2), kl=bool(p3), verdict="PASS" if (p1 and p2 and p3) else "FAIL")
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1, default=float)
    L = [f"# A6b nudger replay {time.strftime('%Y-%m-%d %H:%M')} (a6b_nudger.py, seeds={a.seeds}, w={a.w}, temp={a.temp}, cold={a.cold}, softmax T={TS}, route rows={len(rows)}, days with >= {a.mindayrows} rows: {resB['N0']['n_days']})", "",
         "| arm | hit1 cold | hit1 warm | hit1 all | top-1 entropy bits/day | distinct/day | KL med bits | KL p90 bits | KL med warm | KL med cold |", "|---|---|---|---|---|---|---|---|---|---|"]
    for ar in AR:
        h = hitA[ar]; r = resB[ar]; g = lambda v: f"{f(v):.3f}"
        L.append(f"| {ar} | {f(h['cold'])*100:.1f}+-{sd_(h['cold'])*100:.1f} | {f(h['warm'])*100:.1f}+-{sd_(h['warm'])*100:.1f} | {f(h['all'])*100:.1f}+-{sd_(h['all'])*100:.1f} | {f(r['entropy']):.2f}+-{sd_(r['entropy']):.2f} | {f(r['distinct']):.1f} | {g(r['kl_med'])} | {g(r['kl_p90'])} | {g(r['kl_med_warm'])} | {g(r['kl_med_cold'])} |")
    L += ["", f"Production router top-1 (route.log col 3) same days: entropy {res['prod_mean_entropy']:.2f} bits, distinct {res['prod_mean_distinct']:.1f}.", "",
          f"Rule: N2 hit1(all) >= N0 + .05 -> gain {h0*100:+.1f} pts ({'ok' if p1 else 'FAIL'}); N2 entropy >= N0 -> {f(resB['N2']['entropy']):.2f} vs {f(resB['N0']['entropy']):.2f} ({'ok' if p2 else 'FAIL'}); N2 median KL <= 1 -> {f(resB['N2']['kl_med']):.3f} ({'ok' if p3 else 'FAIL'}). VERDICT {res['pass']['verdict']}"]
    open(os.path.join(HERE, "notes", "a6b-table-raw.md"), "w", encoding="utf-8").write("\n".join(L) + "\n"); print("\n".join(L))

if __name__ == "__main__": main()
