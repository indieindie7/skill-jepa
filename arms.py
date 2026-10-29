"""arms.py: held-out 3-arm eval (notes/mindfield-sparse-2026-10-28.md sec 3, SELFIMPROVE_PLAN sec 9): same K, same candidate pool, same briefs.
  C:/llm/qlora/venv/Scripts/python.exe arms.py [--seeds 3] [--out data/arms-20261028.json]     (CPU, ~10 min)
A0 = production-style router ranking: opsd trigger score + 0.1*BM25(SKILL.md) + per-sid prev-skill prior (treeroute_eval.py 'root+prev a=1.0': prior normalised to sum 1), pool = A0 top-P (P=8, score > 0), returns top-K (K=3).
A1 = explore/suspect slot: online-only, see data/trace_state.json (explore_hit_rate, suspect_hit_rate).
B1 = A0 + delta * clip(avb_logit, -c, c) inside the same pool and K (c=3, delta in {0.1,0.3,1.0} picked on VAL only); avb_logit = AVB v2 head (avb_rerank.Head, val-fit, trained on the split's TRAIN part only);
     a pool skill outside the head's 16 candidates gets the median logit of its row (neutral). Abstain for both arms = route iff top combined score >= thr, thr tuned on VAL (acc_with_abstain; hit1 is thr-free).
Splits: time = avb_rerank's (train head pos<0.49, VAL [0.49,0.70), TEST last 30 % of every chat); session = 30 % of sids by md5 held out as TEST, 20 % of the rest as VAL, the rest head TRAIN
(knn pool for VAL = TRAIN rows, for TEST = TRAIN+VAL rows, as time). Metrics: avb_feats.metrics definitions (hit1/hit3 on labelled rows, false_route_on_none, abstain_on_labelled, acc_with_abstain)
+ tokens = mean over ALL test rows of sum(len(opsd.brief(skill))/4) for the returned skills (0 when abstained; brief built without the query: only its branch pick depends on q)."""
import argparse, collections, datetime, hashlib, json, math, os, random, re, sys, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "data"); sys.path.insert(0, HERE)
OPS = os.path.join(os.path.expanduser("~"), "Documents", "Tools", "ops"); sys.path.insert(0, OPS)
import skilljepa as sj, avb_feats as af, avb_rerank as ar
import opsd
opsd.LOG_ROUTES = False
FROZEN_F = os.path.join(D, "a0_index.json"); FROZEN = os.path.exists(FROZEN_F)   # default ON when the snapshot exists (freeze_index.py); --live turns it off
K_SHOW, POOL, CLIP, DELTAS, NOMATCH = 3, 8, 3.0, (0.1, 0.3, 1.0), 0.35


class A0:
    """treeroute_eval.py root_scores + prev_prior, per pairs row (t = pairs UTC time, uses must be >= 60 s old like avb_feats.lex)."""
    def __init__(s, frozen=None):
        s.frozen = FROZEN if frozen is None else frozen; print("A0 mode:", "FROZEN " + FROZEN_F if s.frozen else "LIVE (opsd + ~/.claude/skills)", flush=True); tok = opsd._tok; s.tf, s.ln = {}, {}
        if s.frozen:
            fz = json.load(open(FROZEN_F, encoding="utf-8")); s.skills = fz["skills"]; s.idf = fz["idf"]; bodies = fz["bodies"]; s.names = list(s.skills); urows = fz["usage"]; s.brief_tok = {nm: fz["brief_chars"][nm] / 4 for nm in s.names}
        else:
            s.skills = opsd.load_skills(); s.idf = opsd._SK["idf"]; s.names = list(s.skills); bodies = {}
            for nm in s.names:
                try: bodies[nm] = open(os.path.join(af.SK, nm, "SKILL.md"), encoding="utf-8", errors="replace").read()
                except OSError: bodies[nm] = None
            urows = []
            for l in open(os.path.join(af.SK, "usage.jsonl"), encoding="utf-8"):
                try: j = json.loads(l); urows.append({"t": j["t"], "sid": j.get("sid"), "skill": j.get("skill")})
                except (ValueError, KeyError): continue
            s.brief_tok = {nm: len(opsd.brief(s.skills[nm])) / 4 for nm in s.names}
        for nm in s.names:
            body = bodies.get(nm); body = s.skills[nm].get("description", "") if body is None else body
            c = collections.Counter(tok(body)); s.tf[nm] = c; s.ln[nm] = sum(c.values())
        s.N = len(s.names); s.avg = sum(s.ln.values()) / s.N; s.df = collections.Counter()
        for c in s.tf.values(): s.df.update(c.keys())
        s.use = collections.defaultdict(list)
        for j in urows:
            try: t = time.mktime(time.strptime(j["t"], "%Y-%m-%d %H:%M"))
            except (ValueError, KeyError): continue
            if len(j.get("skill") or "") > 2 and j.get("sid"): s.use[j["sid"]].append((t, j["skill"]))

    def bm25(s, qt, nm, k1=1.2, b=0.75):
        c = s.tf[nm]; r = 0.0
        for w in set(qt):
            f = c.get(w)
            if f: r += math.log(1 + (s.N - s.df[w] + 0.5) / (s.df[w] + 0.5)) * f * (k1 + 1) / (f + k1 * (1 - b + b * s.ln[nm] / s.avg))
        return r

    def trig(s, q, nm):
        ql = q.lower(); qw = set(w for w in re.findall(r"[a-z][a-z0-9_.+-]{2,}", ql) if w not in opsd._STOP); sk = s.skills[nm]; tr = sk.get("triggers", {}); idf = s.idf
        hit = sum(wt * idf.get(w, 1.0) for w, wt in tr.items() if (w in qw) or (" " in w and w in ql)); r = hit / ((sum(wt * idf.get(w, 1.0) for w, wt in tr.items()) or 1) ** 0.5)
        if nm.replace("-", " ") in ql or nm in ql or nm.replace("-", "") in ql.replace(" ", ""): r += 1.0
        if any(x in ql for x in sk.get("exclusions", [])): return None
        for part in nm.split("-"):
            if len(part) > 3 and part in qw: r += 0.25 * idf.get(part, 1.0)
        return r

    def score(s, p):
        """-> [(skill, score)] sorted, score > 0."""
        q, qt = p["q"], opsd._tok(p["q"]); out = {}
        for nm in s.names:
            r = s.trig(q, nm)
            if r is None: continue
            r += 0.1 * s.bm25(qt, nm)
            if r > 0: out[nm] = r
        t = datetime.datetime.fromisoformat(p["t"].replace("Z", "+00:00")).timestamp(); pp = {}
        for ut, sk in s.use.get(p["sid"], ()):
            if ut + 60 <= t < ut + 6 * 3600: pp[sk] = pp.get(sk, 0.0) + 0.5 ** ((t - ut) / 1800.0)
        tot = sum(pp.values()) or 1.0
        for k, v in pp.items(): out[k] = out.get(k, 0.0) + v / tot
        return sorted(((k, v) for k, v in out.items() if v > 0 and k in s.skills), key=lambda kv: -kv[1])


def tune_thr(top, route_fn_hit1, lab_none):
    """thr maximising acc_with_abstain on VAL: route iff top >= thr; routed row right iff hit1, abstained row right iff gold is none."""
    best = (-1.0, 0.0)
    for t in np.r_[0.0, np.quantile(top, np.linspace(0, 1, 101)), top.max() + 1.0]:
        best = max(best, (np.where(top >= t, route_fn_hit1, lab_none).mean(), -float(t)))
    return -best[1]


def metrics(R, thr, tokf):
    """R: dict(top (n,), rank [[names]], gold [set], none (n,) bool) -> metric dict."""
    n = len(R["top"]); L = ~R["none"]; route = R["top"] >= thr
    h1 = np.array([bool(r[:1] and r[0] in g) for r, g in zip(R["rank"], R["gold"])]); h3 = np.array([bool(set(r[:K_SHOW]) & g) for r, g in zip(R["rank"], R["gold"])])
    tok = np.array([sum(tokf[x] for x in r[:K_SHOW]) if rt else 0.0 for r, rt in zip(R["rank"], route)])
    m = {"hit1": h1[L].mean(), "hit3": h3[L].mean(), "false_route_on_none": route[R["none"]].mean(), "abstain_on_labelled": (~route[L]).mean(), "acc_with_abstain": np.where(route, h1, R["none"]).mean(), "tokens": tok.mean()}
    return {k: round(float(v), 3 if k != "tokens" else 1) for k, v in m.items()} | {"thr": round(float(thr), 4)}


def fit_arm(Rv, Rt, tokf, thr=None):
    """thr (None = tune on VAL) -> val + test metrics."""
    h1 = np.array([bool(r[:1] and r[0] in g) for r, g in zip(Rv["rank"], Rv["gold"])])
    thr = tune_thr(Rv["top"], h1, Rv["none"]) if thr is None else thr
    return {"val": metrics(Rv, thr, tokf), "test": metrics(Rt, thr, tokf)}


def combine(a0rows, cands, Z, names, delta):
    """B1 scoring: A0 pool (top POOL) + delta*clip(logit); -> rank, top, coverage."""
    top, rank, cov = [], [], []
    for pool, c, z in zip(a0rows, cands, Z):
        d = {names[j]: float(np.clip(z[k], -CLIP, CLIP)) for k, j in enumerate(c)}; med = float(np.median(list(d.values())))
        sc = sorted(((nm, v + delta * d.get(nm, med)) for nm, v in pool), key=lambda kv: -kv[1]); rank.append([nm for nm, _ in sc]); top.append(sc[0][1] if sc else -9.0)
        cov.append(np.mean([nm in d for nm, _ in pool]) if pool else 1.0)
    return rank, np.array(top), float(np.mean(cov))


def main():
    global POOL
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--out", default=os.path.join(D, "arms-20261028.json")); ap.add_argument("--topk", type=int, default=16); ap.add_argument("--pool", type=int, default=POOL, help="A0 candidates reranked by B1/B2 (default 8)")
    ap.add_argument("--live", action="store_true", help="A0 reads live opsd/SKILL.md instead of data/a0_index.json"); ap.add_argument("--a0-only", action="store_true", help="score A0 only (hit1 time/session), no B1")
    ap.add_argument("--distill", help="teacher_dump.jsonl: adds arm B2 = B1 with the head trained CE + W*KL(teacher) (+ unlabelled route.log rows, KL only)"); ap.add_argument("--kl", type=float, default=1.0); a = ap.parse_args(); POOL = a.pool
    global FROZEN
    if a.live: FROZEN = False
    if a.distill and a.out == os.path.join(D, "arms-20261028.json"): a.out = os.path.join(D, "arms-20261028-distill.json")
    t0 = time.time(); X, Y, lab, allp, sid, names, plab = sj.data("prevskill", "all"); n = len(lab); K = a.topk
    pairs = [json.loads(l) for l in open(os.path.join(D, "pairs.jsonl"), encoding="utf-8")]; B, Pr = af.lex(pairs, names)
    pos, cnt = np.zeros(n), {}
    for i, c in enumerate(sid): pos[i] = cnt.get(c, 0); cnt[c] = pos[i] + 1
    tot = np.array([cnt[c] for c in sid]); allp_prev = [allp[i - 1] if i and sid[i - 1] == sid[i] else set() for i in range(n)]
    h = lambda s, salt: int(hashlib.md5((salt + s).encode()).hexdigest()[:8], 16) % 100
    te_s = np.array([h(s, "test") < 30 for s in sid]); va_s = np.array([(not t) and h(s, "val") < 20 for s, t in zip(sid, te_s)]); tr_s = ~te_s & ~va_s
    tr_t, sub = pos < 0.7 * tot, pos < 0.49 * tot
    splits = {"time": (np.where(sub)[0], np.where(tr_t & ~sub)[0], np.where(~tr_t)[0], np.where(tr_t)[0]),   # head-train, val, test, test-knn pool
              "session": (np.where(tr_s)[0], np.where(va_s)[0], np.where(te_s)[0], np.where(~te_s)[0])}
    unl = np.zeros(n, bool); TQ0 = None
    if a.distill:   # teacher rows for pairs prompts + unlabelled route.log prompts appended to the pool arrays (rows >= n are train-only, never val/test)
        TV, unl_rows = ar.teacher_load(a.distill, names); TQ0 = np.full((n, len(names) + 1), np.nan, np.float32)
        for j in range(n):
            v = TV.get(ar.nzq(pairs[j]["q"]))
            if v is not None: TQ0[j] = v
        held_all = {ar.nzq(pairs[j]["q"]) for sp_ in splits.values() for j in np.r_[sp_[1], sp_[2]]}
        ux, ukeep = ar.load_unl(unl_rows, names, Y, held_all, os.path.join(D, "emb_teacher_unl.npz")); m_ = len(ukeep); en = np.arange(n, n + m_)
        X, lab, plab, B, Pr = np.vstack([X, ux["X"]]), np.r_[lab, ux["lab"]], np.r_[plab, ux["plab"]], np.vstack([B, ux["B"]]), np.vstack([Pr, ux["Pr"]]); allp, allp_prev = allp + ux["allp"], allp_prev + ux["allp_prev"]
        unl = np.r_[unl, np.ones(m_, bool)]; TQ0 = np.vstack([TQ0, np.stack([TV[ar.nzq(r["q"])] for r in ukeep])]) if m_ else TQ0; print("distill: teacher rows", len(TV), "unlabelled kept", m_, flush=True)
    a0 = A0(); need = sorted(set(np.r_[splits["time"][1], splits["time"][2], splits["session"][1], splits["session"][2]].tolist())); pool = {}
    for i in need: pool[i] = a0.score(pairs[i])
    print("A0 scored", len(pool), "rows", round(time.time() - t0), "s", flush=True)
    nm_ = lambda s_: {names[j] for j in s_}
    def R_of(idx, rank=None, top=None):
        rk = rank if rank is not None else [[nm for nm, _ in pool[i][:K_SHOW]] for i in idx]; tp = top if top is not None else np.array([pool[i][0][1] if pool[i] else -9.0 for i in idx])
        return {"top": tp, "rank": rk, "gold": [nm_(allp[i]) for i in idx], "none": lab[idx] < 0}
    if a.a0_only:
        for sp, (si, vi, ei, ti) in splits.items(): r0 = fit_arm(R_of(vi), R_of(ei), a0.brief_tok); print("A0ONLY", sp, "mode", "frozen" if a0.frozen else "live", "test", r0["test"], flush=True)
        return
    def loo(si, k=10):
        Xs, ls = X[si], lab[si]; lt = ls[ls >= 0]; Sm = Xs @ Xs[ls >= 0].T; Sm[np.arange(len(si))[:, None] == np.where(ls >= 0)[0][None]] = -9
        nn_ = np.argsort(-Sm, 1)[:, :k]; E = np.ones((len(si), len(Y)), np.float32)
        for r in range(len(si)): np.add.at(E[r], lt[nn_[r]], -Sm[r, nn_[r]] / k)
        return E
    def part(i, E, shuf=None, fit=False, TQs=None):
        cs = af.cands(E, B[i], K, True)
        if shuf: [shuf.shuffle(c) for c in cs]
        y, out = af.target(lab[i], [allp[j] for j in i], cs, False); T = af.feats(X[i], Y, E, B[i], Pr[i], cs, plab[i], [allp_prev[j] for j in i], True); k = (y >= 0) if fit else np.ones(len(y), bool)
        P = {"lab": lab[i], "cands": cs, "allp": [allp[j] for j in i], "n": len(i)}
        if TQs is not None:   # B2 train part: unlabelled rows y = -2 (KL only); tq/tm = teacher over [cands, none, other]
            y = np.where(unl[i], -2, y); k = (y >= 0) | (y == -2); P["tq"], P["tm"] = (z[k] for z in ar.tq_for(cs, TQs[i]))
        return {"T": T[k], "y": y[k]} | P
    res = {"meta": {"K_show": K_SHOW, "pool": POOL, "clip": CLIP, "deltas": DELTAS, "head_K": K, "seeds": a.seeds, "distill_W": a.kl if a.distill else None, "nomatch_prod_thr": NOMATCH, "date": time.strftime("%F %T"),
                    "A1": "online, see trace_state (explore_hit_rate / suspect_hit_rate in data/trace_state.json history)"}}
    for sp, (si, vi, ei, ti) in splits.items():
        tokf = a0.brief_tok; print(sp, "train", len(si), "val", len(vi), "test", len(ei), "labelled test", int((lab[ei] >= 0).sum()), flush=True)
        Rv0, Rt0 = R_of(vi), R_of(ei); r0 = fit_arm(Rv0, Rt0, tokf); r0["test_at_prod_thr"] = metrics(Rt0, NOMATCH, tokf)
        pv = [pool[i][:POOL] for i in vi]; pt = [pool[i][:POOL] for i in ei]; r0["recall_pool_test"] = round(float(np.mean([bool(nm_(allp[i]) & {nm for nm, _ in p}) for i, p in zip(ei, pt) if lab[i] >= 0])), 3)
        parts = {"train": part(si, loo(si), random.Random(3), True), "val": part(vi, sj.knn_energies(X[si], lab[si], X[vi], len(Y))), "test": part(ei, sj.knn_energies(X[ti], lab[ti], X[ei], len(Y)))}
        def arm(prts, kl, tag):
            runs = []
            for sd in range(a.seeds):
                ms, r = ar.run(sd, prts, True, refit=False, kl=kl); m = ms[0]
                with torch.no_grad(): Zv, Zt = (m(torch.tensor(prts[k]["T"])).numpy()[:, :-1] for k in ("val", "test"))
                per = {}
                for dl in DELTAS:
                    rkv, tpv, cv = combine(pv, prts["val"]["cands"], Zv, names, dl); rkt, tpt, ct = combine(pt, prts["test"]["cands"], Zt, names, dl)
                    Rv = R_of(vi, rkv, tpv); Rt = R_of(ei, rkt, tpt); f = fit_arm(Rv, Rt, tokf); f["coverage_test"] = round(ct, 3); per[dl] = f
                best = max(DELTAS, key=lambda dl: (per[dl]["val"]["hit1"] + per[dl]["val"]["acc_with_abstain"], -dl)); runs.append({"seed": sd, "delta": best, "val_ce": r["val_ce"], "chosen": per[best], "all_deltas": {str(k): v for k, v in per.items()}})
                print(sp, tag, "seed", sd, "delta", best, per[best]["test"], flush=True)
            keys = runs[0]["chosen"]["test"].keys(); return {k: [round(float(np.mean([r["chosen"]["test"][k] for r in runs])), 3), round(float(np.std([r["chosen"]["test"][k] for r in runs])), 3)] for k in keys}, runs
        agg, runs = arm(parts, 0.0, "B1")
        res[sp] = {"n": {"train_head": len(si), "val": len(vi), "test": len(ei), "test_labelled": int((lab[ei] >= 0).sum()), "test_none": int((lab[ei] < 0).sum())}, "A0": r0, "B1": agg, "B1_runs": runs, "A1": "online"}
        if a.distill:   # B2 = same head/deltas/pool, head trained CE + W*KL(teacher) on TRAIN rows + unlabelled rows; teacher rows whose text is a val/test prompt of THIS split are masked
            hs = {ar.nzq(pairs[j]["q"]) for j in np.r_[vi, ei]}; TQs = TQ0.copy()
            for j in range(n):
                if ar.nzq(pairs[j]["q"]) in hs: TQs[j] = np.nan
            si2 = np.r_[si, np.arange(n, len(lab))]; pd_ = dict(parts); pd_["train"] = part(si2, loo(si2), random.Random(3), True, TQs)
            res[sp]["B2"], res[sp]["B2_runs"] = arm(pd_, a.kl, "B2"); res[sp]["B2_info"] = {"W": a.kl, "train_rows": int(len(pd_["train"]["y"])), "train_rows_teacher": int(pd_["train"]["tm"].sum()), "teacher_file": a.distill}
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1); print("secs", round(time.time() - t0))
    for sp in splits:
        print(sp, "A0", res[sp]["A0"]["test"]); print(sp, "B1", {k: v[0] for k, v in res[sp]["B1"].items()})
        if a.distill: print(sp, "B2", {k: v[0] for k, v in res[sp]["B2"].items()}, res[sp]["B2_info"])


if __name__ == "__main__":
    main()
