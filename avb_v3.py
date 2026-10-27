"""avb_v3.py: AVB head v3 = avb_rerank.py v2 + three switchable changes (v2 untouched; Head/load_extra imported from it, cands/target/metrics from avb_feats).
  C:/llm/qlora/venv/Scripts/python.exe avb_v3.py --extra data/pairs_trace.jsonl --name v3_base [--listwise [--temp 0.5] [--margin M]] [--body feats|mean|max] [--fn-filter [--fn-cos 0.90]] --out data/avb-v3-20261027.json
All flags off = v2 (same data, splits, thr tuning, refit). Keys written: <name>_head(_thr), <name>_refit(_thr), <name>_info, <name>_runs.
--listwise : v2's loss F.cross_entropy over the K+1 logits (softmax across candidate set + none) was ALREADY listwise. This switch = SkillRouter variant: logits/temp (default 0.5; the same 1/temp is applied at scoring so P(none) stays consistent)
             and, if --margin M > 0, + mean relu(M - (z_gold - z_hardest_negative)) on the scaled logits (gold may be the none slot).
--body M   : skill-body embeddings (build_body(), cached data/emb_body.npz: SKILL.md minus frontmatter, <=6 chunks of ~1450 chars, EmbeddingGemma). All modes add per-candidate scalars max/mean cos(query, chunk);
             feats = scalars only; mean = scalars + candidate vector Y replaced by renormalised mean chunk embedding; max = scalars + 2nd candidate vector = embedding of the chunk most similar to the query.
--fn-filter: false-negative mask: in a TRAIN/REFIT row, every non-gold candidate with cos(Y[cand],Y[gold]) >= --fn-cos or in the row's accepted label set allp is removed from the CE denominator (and hinge negatives)."""
import argparse, json, os, random, time
import numpy as np, torch, torch.nn.functional as F
import policy_eval as pe, skilljepa as sj, avb_feats as af
from avb_rerank import Head, load_extra


def build_body(names, force=False, chunk=1450, maxc=6):
    """-> C (nchunks, d) chunk embeddings, idx (nchunks,) skill index, cached in data/emb_body.npz (skill without SKILL.md/body falls back to its description text)."""
    p = os.path.join(pe.DATA, "emb_body.npz")
    if os.path.exists(p) and not force:
        z = np.load(p); assert list(z["names"]) == list(names); return z["C"], z["idx"]
    desc = json.load(open(os.path.join(sj.D, "skills.json"), encoding="utf-8")); texts, idx = [], []
    for i, nm in enumerate(names):
        try: body = open(os.path.join(af.SK, nm, "SKILL.md"), encoding="utf-8", errors="replace").read()
        except OSError: body = ""
        if body.startswith("---"):
            e = body.find("\n---", 3); body = body[e + 4:] if e > 0 else body
        ch, cur = [], ""
        for ln in body.strip().splitlines(True):
            while len(ln) > chunk:   # over-long line: cut
                if cur.strip(): ch.append(cur); cur = ""
                ch.append(ln[:chunk]); ln = ln[chunk:]
            if len(cur) + len(ln) > chunk and cur.strip(): ch.append(cur); cur = ""
            cur += ln
        if cur.strip(): ch.append(cur)
        ch = ch[:maxc] or [desc[nm]]; texts += ch; idx += [i] * len(ch)
    C = sj.embed([sj.S_PRE.format(names[i]) + t for i, t in zip(idx, texts)])
    np.savez(p, C=C, idx=np.array(idx), names=np.array(names), texts=np.array(texts)); return C, np.array(idx)


def body_tables(X, C, idx, nS):
    """per row, per skill: max cos, mean cos, global index of the max chunk -> (n,nS) x3."""
    S = X @ C.T; mx, mn = (np.zeros((len(X), nS), np.float32) for _ in range(2)); am = np.zeros((len(X), nS), np.int64)
    for s in range(nS):
        w = np.where(idx == s)[0]; mx[:, s] = S[:, w].max(1); mn[:, s] = S[:, w].mean(1); am[:, s] = w[S[:, w].argmax(1)]
    return mx, mn, am


def feats3(Xq, Y, E, B, Pr, cs, plab, allp_prev, body, bt):
    """af.feats (lex feats on) + body columns. mean mode swaps Y for the mean chunk embedding inside af.feats."""
    mode = body[0] if body else None; T = af.feats(Xq, body[1] if mode == "mean" else Y, E, B, Pr, cs, plab, allp_prev, True)
    if not mode: return T
    c = np.array(cs); mx, mn, am = bt; g = lambda M: np.take_along_axis(M, c, 1); n, K = c.shape; sc = np.zeros((n, K + 1, 2), np.float32)
    sc[:, :K, 0], sc[:, :K, 1] = g(mx), g(mn); sc[:, K, 0] = g(mx).max(1); T = np.concatenate([T, sc], 2)
    if mode == "max": ex = np.zeros((n, K + 1, Xq.shape[1]), np.float32); ex[:, :K] = body[2][g(am)]; T = np.concatenate([T, ex], 2)
    return T


def fn_mask(y, cs, allp, Y, thr):
    """(n,K+1) bool, True = false-negative candidate to exclude; -> (mask, n_by_cos, n_by_allp_only, n_rows_with_mask). Only rows whose gold is a candidate slot (y<K)."""
    n, K = len(y), len(cs[0]); M = np.zeros((n, K + 1), bool); sim = Y @ Y.T; nc = na = 0
    for i in range(n):
        if y[i] >= K: continue
        g = cs[i][y[i]]
        for k, s in enumerate(cs[i]):
            if k == y[i]: continue
            a, b = sim[g, s] >= thr, s in allp[i]
            if a or b: M[i, k] = True; nc += int(a); na += int(b and not a)
    return M, nc, na, int(M.any(1).sum())


def lossf(Z, y, M, tau, margin):
    Z = Z / tau
    if M is not None: Z = Z.masked_fill(M, -1e4)
    l = F.cross_entropy(Z, y)
    if margin > 0:
        zg = Z.gather(1, y[:, None]).squeeze(1); neg = Z.scatter(1, y[:, None], -1e4).max(1).values; l = l + F.relu(margin - (zg - neg)).mean()
    return l


def epoch(m, opt, X, y, M, tau, margin):
    m.train(); perm = torch.randperm(len(X))
    for b in range(0, len(perm), 64):
        i = perm[b:b + 64]; loss = lossf(m(X[i]), y[i], None if M is None else M[i], tau, margin); opt.zero_grad(); loss.backward(); opt.step()


def run(seed, parts, cfg, ep=60, pat=8):
    tau, margin = cfg["tau"], cfg["margin"]; tr, va = parts["train"], parts["val"]; din = tr["T"].shape[2]; mk = lambda: Head(din)
    torch.manual_seed(seed); np.random.seed(seed); m = mk(); opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-2)
    tt = lambda P: (torch.tensor(P["T"]), torch.tensor(P["y"]), torch.tensor(P["M"]) if "M" in P else None)
    Xt, yt, Mt = tt(tr); Xv, yv = torch.tensor(va["T"]), torch.tensor(va["y"]); ok = yv >= 0; best = (9e9, None, 0)
    for e in range(ep):
        epoch(m, opt, Xt, yt, Mt, tau, margin); m.eval()
        with torch.no_grad(): vl = F.cross_entropy(m(Xv)[ok] / tau, yv[ok]).item()
        if vl < best[0]: best = (vl, {k: v.clone() for k, v in m.state_dict().items()}, e)
        elif e - best[2] >= pat: break
    m.load_state_dict(best[1]); m.eval(); sc = lambda mm, P: mm(torch.tensor(P["T"])).detach().numpy() / tau
    thr = af.tune_thr(sc(m, va), va); r = {"thr_val": round(thr, 4), "epochs": best[2] + 1, "val_ce": round(best[0], 4)}; Zt = sc(m, parts["test"])
    r["argmax"], r["thr"] = af.metrics(Zt, parts["test"]), af.metrics(Zt, parts["test"], thr)
    torch.manual_seed(seed); m2 = mk(); opt = torch.optim.AdamW(m2.parameters(), 1e-3, weight_decay=1e-2); Xr, yr, Mr = tt(parts["refit"])
    for e in range(best[2] + 1): epoch(m2, opt, Xr, yr, Mr, tau, margin)
    m2.eval(); Z2 = sc(m2, parts["test"]); r["refit_argmax"], r["refit_thr"] = af.metrics(Z2, parts["test"]), af.metrics(Z2, parts["test"], thr)
    return m, r


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--topk", type=int, default=16); ap.add_argument("--name", default="v3_base"); ap.add_argument("--out"); ap.add_argument("--extra")
    ap.add_argument("--listwise", action="store_true"); ap.add_argument("--temp", type=float, default=0.5); ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--body", choices=["feats", "mean", "max"]); ap.add_argument("--fn-filter", action="store_true"); ap.add_argument("--fn-cos", type=float, default=0.90); ap.add_argument("--threads", type=int, default=0); a = ap.parse_args()
    assert a.out; K = a.topk; nm = a.name; t0 = time.time()
    if a.threads: torch.set_num_threads(a.threads)
    cfg = {"tau": a.temp if a.listwise else 1.0, "margin": a.margin if a.listwise else 0.0}
    S = pe.build(); X, Y, lab, allp, sid, names, plab = sj.data("prevskill", "all"); n = len(lab); pairs = [json.loads(l) for l in open(os.path.join(pe.DATA, "pairs.jsonl"), encoding="utf-8")]
    B, Pr = af.lex(pairs, names); pos, cnt = np.zeros(n), {}
    for i, c in enumerate(sid): pos[i] = cnt.get(c, 0); cnt[c] = pos[i] + 1
    tot = np.array([cnt[c] for c in sid]); te = pos >= 0.7 * tot; tr = ~te; sub = pos < pe.VAL_LO * tot
    allp_prev = [allp[i - 1] if i and sid[i - 1] == sid[i] else set() for i in range(n)]
    body = None; bst = {}; bidx = None
    if a.body:
        C, bidx = build_body(names); Ym = np.zeros_like(Y)
        for s in range(len(Y)): v = C[bidx == s].mean(0); Ym[s] = v / max(np.linalg.norm(v), 1e-9)
        body = (a.body, Ym, C); bst = {"chunks": len(C), "skills": len(Y)}
    mask_stats = {}
    def part(i, E, shuf=None, fit=False, tag=None):
        cs = af.cands(E, B[i], K, True)
        if shuf: [shuf.shuffle(c) for c in cs]
        ap_ = [allp[j] for j in i]; y, out = af.target(lab[i], ap_, cs, False)
        bt = body_tables(X[i], body[2], bidx, len(Y)) if body else None
        T = feats3(X[i], Y, E, B[i], Pr[i], cs, plab[i], [allp_prev[j] for j in i], body, bt)
        rec = float(np.mean([bool(set(c) & allp[j]) for c, j in zip(cs, i) if lab[j] >= 0])); P = {"T": T, "y": y, "lab": lab[i], "cands": cs, "allp": ap_, "out": out, "n": len(i), "recall": rec}
        if fit and a.fn_filter:
            M, nc, na, nr = fn_mask(y, cs, ap_, Y, a.fn_cos); P["M"] = M; mask_stats[tag] = {"masked_pairs": int(M.sum()), "by_cos": nc, "by_allp_only": na, "rows_with_mask": nr, "rows": len(i)}
        return P
    def loo(si, k=10):
        Xs, ls = X[si], lab[si]; lt = ls[ls >= 0]; Sm = Xs @ Xs[ls >= 0].T; Sm[np.arange(len(si))[:, None] == np.where(ls >= 0)[0][None]] = -9
        nn_ = np.argsort(-Sm, 1)[:, :k]; E = np.ones((len(si), len(Y)), np.float32)
        for r in range(len(si)): np.add.at(E[r], lt[nn_[r]], -Sm[r, nn_[r]] / k)
        return E
    si, ti = np.where(sub)[0], np.where(tr)[0]; vi, ei = S["val"]["idx"], S["test"]["idx"]
    parts = {"val": part(vi, sj.knn_energies(X[sub], lab[sub], X[vi], len(Y))), "test": part(ei, sj.knn_energies(X[tr], lab[tr], X[ei], len(Y)))}
    ex = load_extra(a.extra, names, Y) if a.extra and os.path.getsize(a.extra) else None
    if ex:
        m_ = len(ex["lab"]); en = np.arange(n, n + m_); si, ti = np.r_[si, en], np.r_[ti, en]; X, lab, plab, B, Pr = np.vstack([X, ex["X"]]), np.r_[lab, ex["lab"]], np.r_[plab, ex["plab"]], np.vstack([B, ex["B"]]), np.vstack([Pr, ex["Pr"]])
        allp, allp_prev = allp + ex["allp"], allp_prev + ex["allp_prev"]
    parts["train"] = part(si, loo(si), random.Random(3), True, "train"); parts["refit"] = part(ti, loo(ti), random.Random(4), True, "refit")
    parts = {k: parts[k] for k in ("train", "val", "test", "refit")}; lt_ = S["test"]["lab"]; res = []
    for sd in range(a.seeds): m, r = run(sd, parts, cfg); res.append(r); print(nm, sd, r, flush=True)
    agg = lambda k: {q: [round(float(np.mean([r[k][q] for r in res])), 3), round(float(np.std([r[k][q] for r in res])), 3)] for q in res[0][k]}
    out = json.load(open(a.out, encoding="utf-8")) if os.path.exists(a.out) else {}
    info = {"flags": {"listwise": a.listwise, "temp": cfg["tau"], "margin": cfg["margin"], "body": a.body, "fn_filter": a.fn_filter, "fn_cos": a.fn_cos}, "K": K, "recallK": {k: round(p["recall"], 3) for k, p in parts.items()},
            "rows": {k: p["n"] for k, p in parts.items()}, "params": sum(p.numel() for p in m.parameters()), "secs": round(time.time() - t0, 1), "din": parts["train"]["T"].shape[2]} | ({"extra_rows": len(ex["lab"])} if ex else {}) | ({"mask": mask_stats} if mask_stats else {}) | bst
    out.update({"n": {"train_head": len(parts["train"]["y"]), "val": len(parts["val"]["y"]), "test": len(lt_), "refit": len(parts["refit"]["y"])}, nm + "_head": agg("argmax"), nm + "_head_thr": agg("thr"), nm + "_refit": agg("refit_argmax"), nm + "_refit_thr": agg("refit_thr"), nm + "_runs": res, nm + "_info": info})
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1); print(nm, "head_thr", out[nm + "_head_thr"]["hit1"], out[nm + "_head_thr"]["acc_with_abstain"], "refit_thr", out[nm + "_refit_thr"]["hit1"], out[nm + "_refit_thr"]["acc_with_abstain"], info.get("mask", ""))


if __name__ == "__main__":
    main()
