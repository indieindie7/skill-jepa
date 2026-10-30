"""avb_rerank.py v2: AVB-style decision head over EXISTING embeddings (no LLM pass). Order-invariant self-attention across K candidates + 1 learned "none" token -> one logit each.
  C:/llm/qlora/venv/Scripts/python.exe avb_rerank.py [--topk 16] [--no-bm25-feats] [--v1] [--no-attn] [--no-refit] [--seeds 3] [--name avb2] --out data/avb-<date>-v2.json   (CPU; every run merges its keys into --out)
Rows/splits as policy_eval.build (time split: test = last 30% of each chat; VAL = train positions [0.49,0.70); head TRAIN = positions < 0.49 with leave-one-out knn). v2 = candidates: knn+BM25 union filled to K (avb_feats.cands);
feats + BM25 score/rank/top1/top3 + prev-skill prior; rows whose gold is outside the K candidates stay, labelled none; early stop on VAL CE + none-thr on VAL, then REFIT on the full train part (positions < 0.70, leave-one-out knn)
for the chosen epoch count with the tuned thr. Keys per run: <name>_head(_thr) = val-fit head, <name>_refit(_thr) = refit head. --v1 = old behaviour (knn top-8, no lex feats, out-of-8 rows dropped, no refit) -> ~avb-20261026.json numbers.
Token = [query(prevskill ctx), cand skill emb, query*cand, scalars of avb_feats.feats]; no positional encoding."""
import argparse, json, os, random, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import policy_eval as pe, skilljepa as sj, avb_feats as af


ARCH = {"d": 256, "blocks": 2, "mixer": "attn", "lexf": True}   # 10-29 --d/--blocks (shrink step): defaults = production head; ff = 2d, out hidden = max(d//2, 8) (= shrink.SHead)


class GrassmannBlock(nn.Module):
    """10-30 A15 (arxiv 2512.19428, causal fix): LN -> Linear d->r -> for k in OFFS pair token i with i-k (zero/masked if i<k; lower index = higher rank) -> Pluecker coords p_ab = u_a v_b - u_b v_a (a<b, r(r-1)/2)
    -> L2 normalise -> mean over valid offsets -> Linear -> d, sigmoid gate from token, residual add."""
    OFFS = (1, 2, 4, 8, 12, 16)
    def __init__(s, d, r=32, p=0.1):
        super().__init__(); s.r = r; s.ln = nn.LayerNorm(d); s.down = nn.Linear(d, r); s.proj = nn.Linear(r * (r - 1) // 2, d); s.gate = nn.Linear(d, d); s.drop = nn.Dropout(p)
        iu = torch.triu_indices(r, r, 1); s.register_buffer("ia", iu[0], persistent=False); s.register_buffer("ib", iu[1], persistent=False)
    def forward(s, x):
        L = x.shape[1]; hn = s.ln(x); u = s.down(hn); acc = torch.zeros(x.shape[0], L, s.proj.in_features, device=x.device, dtype=x.dtype); cnt = torch.zeros(1, L, 1, device=x.device, dtype=x.dtype)
        for k in s.OFFS:
            if k >= L: continue
            v = torch.zeros_like(u); v[:, k:] = u[:, :L - k]; m = torch.zeros(1, L, 1, device=x.device, dtype=x.dtype); m[:, k:] = 1
            p = u[..., s.ia] * v[..., s.ib] - u[..., s.ib] * v[..., s.ia]; p = F.normalize(p, dim=-1) * m; acc = acc + p; cnt = cnt + m
        g = acc / cnt.clamp_min(1)
        return x + torch.sigmoid(s.gate(hn)) * s.drop(s.proj(g))


class GrassmannEnc(nn.Module):
    """blocks x GrassmannBlock over [candidates sorted by an A0-rank proxy ..., none token last]. Proxy key (T scalars, higher-ranked = smaller): knn rank + log BM25 global rank - prior (A0 = trigger + 0.1 BM25 + prev-skill prior;
    the trigger score is not a head feature). Tokens are shuffled in training, so the sort is what makes 'i-k' mean 'higher-ranked'. Without lex feats: knn rank only."""
    def __init__(s, d, blocks, p=0.1):
        super().__init__(); s.blocks = nn.ModuleList([GrassmannBlock(d, 32, p) for _ in range(blocks)])
    def forward(s, h, T):
        K = h.shape[1] - 1
        key = T[:, :K, -10] + T[:, :K, -4] - T[:, :K, -1] if ARCH["lexf"] else T[:, :K, -4]
        perm = key.argsort(dim=1, stable=True); idx = perm[..., None].expand(-1, -1, h.shape[2]); hs = torch.cat([h[:, :K].gather(1, idx), h[:, K:]], 1)
        for b in s.blocks: hs = b(hs)
        out = torch.zeros_like(hs[:, :K]).scatter(1, idx, hs[:, :K]); return torch.cat([out, hs[:, K:]], 1)


def match_d(din, target=None, **kw):
    """grassmann d (multiple of 4) whose Head param count is closest to the production attn d256b2 head."""
    tgt = target or sum(p.numel() for p in Head(din, 256, mixer="attn", **kw).parameters())
    return min(range(16, 512, 4), key=lambda d: abs(sum(p.numel() for p in Head(din, d, mixer="grassmann", **kw).parameters()) - tgt)), tgt


class Head(nn.Module):
    def __init__(s, din, d=None, attn=True, p=0.1, other=False, blocks=None, mixer=None):
        d = d or ARCH["d"]; blocks = ARCH["blocks"] if blocks is None else blocks; s_mix = mixer or ARCH["mixer"]; hid = max(d // 2, 8)
        super().__init__(); s.oth = nn.Linear(d, 1) if other else None   # other = distillation-only extra class (EOT STOP analog); absent in the deployed forward
        s.inp = nn.Sequential(nn.Dropout(p), nn.Linear(din, d), nn.GELU()); s.attn = attn
        s.grass = attn and s_mix == "grassmann" and bool(blocks)
        s.enc = GrassmannEnc(d, blocks, p) if s.grass else (nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 2 * d, p, batch_first=True, norm_first=True), blocks, enable_nested_tensor=False) if blocks else nn.Identity()) if attn else             nn.Sequential(*[nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(p), nn.Linear(2 * d, d)) for _ in range(2)])
        s.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hid), nn.GELU(), nn.Linear(hid, 1))
    def forward(s, T, with_other=False):
        h = s.inp(T); h = (s.enc(h, T) if s.grass else s.enc(h)) if s.attn else h + s.enc(h); z = s.out(h).squeeze(-1)    # (n, K+1)
        return torch.cat([z, s.oth(h[:, -1])], 1) if with_other else z                     # (n, K+2): last = other


def epoch(m, opt, X, y, tq=None, tm=None, W=0.0):
    """CE(gold) on rows with y >= 0 (+ W * KL(teacher || student) over K cands + none + other on rows with tm; y == -2 rows = unlabelled, KL only)."""
    m.train(); perm = torch.randperm(len(X))
    for b in range(0, len(perm), 64):
        i = perm[b:b + 64]
        if tq is None: loss = F.cross_entropy(m(X[i]), y[i])
        else:
            z = m(X[i], with_other=True); yi, ok = y[i], y[i] >= 0; loss = F.cross_entropy(z[ok], yi[ok]) if ok.any() else z.sum() * 0
            mi = tm[i]
            if W > 0 and mi.any(): lp = F.log_softmax(z[mi], 1); t = tq[i][mi]; loss = loss + W * (t * (torch.log(t.clamp_min(1e-9)) - lp)).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()


def run(seed, parts, attn, ep=None, pat=8, refit=True, kl=0.0):
    """val-fit head (early stop on VAL CE, thr on VAL) and, if refit, a fresh head trained on parts['refit'] for the chosen epoch count; both scored on test with the same thr."""
    ep = ep or ARCH.get("ep", 60); tr, va = parts["train"], parts["val"]; din = tr["T"].shape[2]; mk = lambda: Head(din, attn=attn, other=kl > 0)
    tt = lambda P: (torch.tensor(P["tq"]), torch.tensor(P["tm"])) if kl > 0 else (None, None); tqt, tmt = tt(tr)
    torch.manual_seed(seed); np.random.seed(seed); m = mk(); opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-2)
    Xt, yt, Xv, yv = torch.tensor(tr["T"]), torch.tensor(tr["y"]), torch.tensor(va["T"]), torch.tensor(va["y"]); ok = yv >= 0; best = (9e9, None, 0)
    for e in range(ep):
        epoch(m, opt, Xt, yt, tqt, tmt, kl); m.eval()
        with torch.no_grad(): vl = F.cross_entropy(m(Xv)[ok], yv[ok]).item()
        if vl < best[0]: best = (vl, {k: v.clone() for k, v in m.state_dict().items()}, e)
        elif e - best[2] >= pat: break
    m.load_state_dict(best[1]); m.eval(); sc = lambda mm, P: mm(torch.tensor(P["T"])).detach().numpy()
    Zv_ = sc(m, va); thr = af.tune_thr(Zv_, va); r = {"thr_val": round(thr, 4), "epochs": best[2] + 1, "val_ce": round(best[0], 4), "val_thr": af.metrics(Zv_, va, thr)}; Zt = sc(m, parts["test"])
    r["argmax"], r["thr"] = af.metrics(Zt, parts["test"]), af.metrics(Zt, parts["test"], thr); ms = [m]
    if refit:
        torch.manual_seed(seed); m2 = mk(); opt = torch.optim.AdamW(m2.parameters(), 1e-3, weight_decay=1e-2); rf = parts["refit"]; Xr, yr = torch.tensor(rf["T"]), torch.tensor(rf["y"]); tqr, tmr = tt(rf)
        for e in range(best[2] + 1): epoch(m2, opt, Xr, yr, tqr, tmr, kl)
        m2.eval(); Z2 = sc(m2, parts["test"]); r["refit_argmax"], r["refit_thr"] = af.metrics(Z2, parts["test"]), af.metrics(Z2, parts["test"], thr); ms.append(m2)
    return ms, r

NONE_I = 77


def nzq(t):
    return " ".join(t.lower().split())


def teacher_load(path, names):
    """data/teacher_dump.jsonl -> ({nz(q): (len(names)+1,) float32 teacher probs, last = none}, unlabelled rows [{q, ...}]). Failed rows skipped."""
    ix = {n: i for i, n in enumerate(names)}; V, unl = {}, []
    for l in open(path, encoding="utf-8"):
        r = json.loads(l)
        if r.get("fail") or "dist" not in r: continue
        v = np.full(len(names) + 1, r["rest"], np.float32)
        for k, p in r["dist"].items(): v[len(names) if k == "none" else ix[k]] = p
        V[nzq(r["q"])] = v / v.sum()
        if r["label"] is None: unl.append(r)
    return V, unl


def tq_for(cs, TV):
    """cs candidate lists (n, K), TV (n, nS+1) teacher rows (nan = none) -> tq (n, K+2) float32 = [teacher p of each candidate, none, mass outside the candidates = other], tm bool (has teacher)."""
    tm = ~np.isnan(TV[:, 0]); c = np.array(cs); TVf = np.where(tm[:, None], TV, 0.0); pc = np.take_along_axis(TVf, c, 1); pn = TVf[:, -1:]; po = np.clip(1 - pc.sum(1, keepdims=True) - pn, 0, 1)
    tq = np.concatenate([pc, pn, po], 1); tq /= np.maximum(tq.sum(1, keepdims=True), 1e-9); return tq.astype(np.float32), tm


def load_unl(rows, names, Y, drop_q, cache):
    """unlabelled dump rows (route.log prompts, label None) -> extra pool rows like load_nomatch (lab -1, no ctx, unique sid) + teacher vecs; text equal to a val/test prompt (drop_q) dropped."""
    keep = [r for r in rows if nzq(r["q"]) not in drop_q]; qs = [r["q"] for r in keep]
    if os.path.exists(cache) and list(np.load(cache, allow_pickle=True)["q"]) == qs: X0 = np.load(cache, allow_pickle=True)["X"]
    else: X0 = sj.embed([sj.Q_PRE + q for q in qs]); np.savez(cache, X=X0, q=np.array(qs, dtype=object))
    prs = [{"q": r["q"], "t": "2026-10-27T00:00:00Z", "sid": f"unl{i}"} for i, r in enumerate(keep)]; m = len(prs); lab = np.full(m, -1); prv = np.full(m, -1); plab = np.full(m, -1); B, Pr = af.lex(prs, names)
    return {"X": sj.with_ctx(X0, Y, prv, plab, "prevskill"), "lab": lab, "allp": [set() for _ in prs], "sid": np.array([p["sid"] for p in prs]), "plab": plab, "B": B, "Pr": Pr, "allp_prev": [set() for _ in prs]}, keep


def load_extra(path, names, Y):
    """--extra rows (trace_merge pairs_trace.jsonl + emb_trace.npz) -> arrays aligned like sj.data('prevskill','all'); prev = previous row of the same sid in the file."""
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]; X0 = np.load(os.path.join(os.path.dirname(path), "emb_trace.npz"))["X"]; assert len(X0) == len(rows), "rerun trace_merge"
    ix = {n: i for i, n in enumerate(names)}; lab = np.array([ix.get(p["skills"][0], -1) for p in rows]); last, prv = {}, np.full(len(rows), -1)
    for i, p in enumerate(rows): prv[i] = last.get(p["sid"], -1); last[p["sid"]] = i
    plab = np.where(prv >= 0, lab[np.maximum(prv, 0)], -1); B, Pr = af.lex(rows, names)
    return {"X": sj.with_ctx(X0, Y, prv, plab, "prevskill"), "lab": lab, "allp": [{l} if l >= 0 else set() for l in lab], "sid": np.array([p["sid"] for p in rows]), "plab": plab, "B": B, "Pr": Pr,
            "allp_prev": [{lab[j]} if j >= 0 and lab[j] >= 0 else set() for j in prv]}


def load_nomatch(path, names, Y, drop_q):
    """--nomatch rows (data/nomatch.jsonl: prompts that matched no skill) -> NONE rows (lab -1, allp empty, no prev ctx, unique sid per row). Embeddings cached in emb_nomatch.npz (keyed by prompt text).
    Prompts whose text equals a val/test row of pairs.jsonl (drop_q, normalised) are dropped. -> (arrays like load_extra, n_dropped)."""
    nz = lambda t: " ".join(t.lower().split()); raw = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]; keep = [r for r in raw if nz(r["prompt"]) not in drop_q]
    cache = os.path.join(os.path.dirname(path), "emb_nomatch.npz"); qs = [r["prompt"] for r in keep]
    if os.path.exists(cache) and list(np.load(cache, allow_pickle=True)["q"]) == qs: X0 = np.load(cache, allow_pickle=True)["X"]
    else: X0 = sj.embed([sj.Q_PRE + q for q in qs]); np.savez(cache, X=X0, q=np.array(qs, dtype=object))
    rows = [{"q": r["prompt"], "t": r.get("t", "2026-10-26T00:00:00Z"), "sid": r.get("sid") or f"nomatch{i}"} for i, r in enumerate(keep)]   # unique sid -> no prev chaining between unrelated prompts
    m = len(rows); lab = np.full(m, -1); prv = np.full(m, -1); plab = np.full(m, -1); B, Pr = af.lex(rows, names)
    return {"X": sj.with_ctx(X0, Y, prv, plab, "prevskill"), "lab": lab, "allp": [set() for _ in rows], "sid": np.array([r["sid"] for r in rows]), "plab": plab, "B": B, "Pr": Pr,
            "allp_prev": [set() for _ in rows]}, len(raw) - len(keep)


def sweep(a, parts, S, K, nm, lt_, n_teacher, n_unl, t0):
    """--distill: W in {0 (control: CE only on labelled/none rows) + --kl list}; W picked on VAL only (mean over seeds of val hit1 + val acc_with_abstain at the val-tuned thr); best head saved (lowest val CE of the best W)."""
    Ws = [0.0] + [float(x) for x in a.kl.split(",")]; tr0 = parts["train"]; k0 = tr0["y"] >= 0; p0 = dict(parts); p0["train"] = {"T": tr0["T"][k0], "y": tr0["y"][k0]}; R, MS = {}, {}
    for W in Ws:
        rs, mss = [], []
        for sd in range(a.seeds): m, r = run(sd, parts if W > 0 else p0, True, refit=False, kl=W); rs.append(r); mss.append(m[0]); print(nm, "W", W, "seed", sd, r["val_thr"], r["thr"], "ep", r["epochs"], flush=True)
        ag = lambda k, q: [round(float(np.mean([r[k][q] for r in rs])), 3), round(float(np.std([r[k][q] for r in rs])), 3)]
        R[str(W)] = {"val_score": round(float(np.mean([r["val_thr"]["hit1"] + r["val_thr"]["acc_with_abstain"] for r in rs])), 4), "val_ce": round(float(np.mean([r["val_ce"] for r in rs])), 4),
                     "val_thr": {q: ag("val_thr", q) for q in rs[0]["val_thr"]}, "test_thr": {q: ag("thr", q) for q in rs[0]["thr"]}, "test_argmax": {q: ag("argmax", q) for q in rs[0]["argmax"]}, "runs": rs}; MS[W] = mss
    best = max(Ws[1:], key=lambda W: R[str(W)]["val_score"]); rs = R[str(best)]["runs"]; b = min(range(len(rs)), key=lambda i: rs[i]["val_ce"]); m = MS[best][b]
    st = {k: v for k, v in m.state_dict().items() if not k.startswith("oth.")}   # deployed forward has no `other` logit
    torch.save({"state": m.state_dict(), "W": best, "K": K, "din": parts["train"]["T"].shape[2], "seed": b, "thr": rs[b]["thr_val"]}, a.save_distill or os.path.join(pe.DATA, "avb_head_distill.pt"))
    import avb_np; avb_np.save_head((a.save_distill or os.path.join(pe.DATA, "avb_head_distill.pt")).replace(".pt", ".npz"), st, {"thr": rs[b]["thr_val"], "K": K, "din": int(parts["train"]["T"].shape[2]), "seed": b, "val_ce": rs[b]["val_ce"], "epochs": rs[b]["epochs"], "W": best, "teacher": a.distill, "created": time.strftime("%F %T"), "src": a.out})
    out = json.load(open(a.out, encoding="utf-8")) if os.path.exists(a.out) else {}
    out.update({"distill": {"best_W": best, "W": R, "teacher_rows": n_teacher, "unlabelled_kept": n_unl, "seeds": a.seeds, "K": K, "train_rows": int(len(parts["train"]["y"])), "train_rows_teacher": int(parts["train"]["tm"].sum()),
                "n_test": len(lt_), "secs": round(time.time() - t0, 1)}})
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps({"best_W": best} | {w: {k: v for k, v in r.items() if k != "runs"} for w, r in R.items()}, indent=1))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--no-attn", action="store_true"); ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--no-bm25-feats", action="store_true"); ap.add_argument("--v1", action="store_true"); ap.add_argument("--no-refit", action="store_true"); ap.add_argument("--name", default="avb2"); ap.add_argument("--out"); ap.add_argument("--extra"); ap.add_argument("--nomatch"); ap.add_argument("--tag"); ap.add_argument("--save"); ap.add_argument("--d", type=int, default=256); ap.add_argument("--blocks", type=int, default=2); ap.add_argument("--mixer", choices=["attn", "grassmann"], default="attn"); ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--distill"); ap.add_argument("--kl", default="0.5,1,2"); ap.add_argument("--save-distill"); a = ap.parse_args()
    ARCH.update(d=a.d, blocks=a.blocks, mixer=a.mixer, ep=a.epochs, lexf=not (a.no_bm25_feats or a.v1))
    if a.tag: a.out = a.out or os.path.join(pe.DATA, f"avb-{a.tag}.json")
    assert a.out, "--out or --tag"
    if a.v1: a.topk, a.no_bm25_feats, a.no_refit = 8, True, True
    K, union, lexf, drop = a.topk, not a.v1, not a.no_bm25_feats, a.v1; nm = a.name + ("_noattn" if a.no_attn else "")
    t0 = time.time(); S = pe.build(); X, Y, lab, allp, sid, names, plab = sj.data("prevskill", "all"); n = len(lab); pairs = [json.loads(l) for l in open(os.path.join(pe.DATA, "pairs.jsonl"), encoding="utf-8")]
    B, Pr = af.lex(pairs, names); pos, cnt = np.zeros(n), {}
    for i, c in enumerate(sid): pos[i] = cnt.get(c, 0); cnt[c] = pos[i] + 1
    tot = np.array([cnt[c] for c in sid]); te = pos >= 0.7 * tot; tr = ~te; sub = pos < pe.VAL_LO * tot
    allp_prev = [allp[i - 1] if i and sid[i - 1] == sid[i] else set() for i in range(n)]
    def part(i, E, shuf=None, fit=False):   # i = row indices, E = knn energies; fit = training part (v1 drops its y=-1 rows)
        cs = af.cands(E, B[i], K, union)
        if shuf: [shuf.shuffle(c) for c in cs]
        y, out = af.target(lab[i], [allp[j] for j in i], cs, drop); T = af.feats(X[i], Y, E, B[i], Pr[i], cs, plab[i], [allp_prev[j] for j in i], lexf); k = (y >= 0) if fit else np.ones(len(y), bool)
        rec = float(np.mean([bool(set(c) & allp[j]) for c, j in zip(cs, i) if lab[j] >= 0])); P = {"lab": lab[i], "cands": cs, "allp": [allp[j] for j in i], "out": out, "n": len(i), "recall": rec}
        if a.distill and fit:   # unlabelled rows (no gold): y = -2, KL only; every fit row carries its teacher row (tm False = no teacher for it)
            y = np.where(unl[i], -2, y); k = (y >= 0) | (y == -2); P["tq"], P["tm"] = (z[k] for z in tq_for(cs, TQ[i]))
        return {"T": T[k], "y": y[k]} | P
    def loo(si, k=10):   # head TRAIN rows: leave-one-out knn within si (self never a neighbour)
        Xs, ls = X[si], lab[si]; lt = ls[ls >= 0]; Sm = Xs @ Xs[ls >= 0].T; Sm[np.arange(len(si))[:, None] == np.where(ls >= 0)[0][None]] = -9
        nn_ = np.argsort(-Sm, 1)[:, :k]; E = np.ones((len(si), len(Y)), np.float32)
        for r in range(len(si)): np.add.at(E[r], lt[nn_[r]], -Sm[r, nn_[r]] / k)
        return E
    si, ti = np.where(sub)[0], np.where(tr)[0]; vi, ei = S["val"]["idx"], S["test"]["idx"]
    unl, TQ = np.zeros(n, bool), np.full((n, len(names) + 1), np.nan, np.float32)
    if a.distill:   # teacher rows of pairs prompts whose text is NOT a val/test prompt of this split (leak guard) + unlabelled route.log prompts (KL only)
        TV, unl_rows = teacher_load(a.distill, names); held = {nzq(pairs[j]["q"]) for j in np.r_[vi, ei]}
        for j in range(n):
            v = TV.get(nzq(pairs[j]["q"]))
            if v is not None and nzq(pairs[j]["q"]) not in held: TQ[j] = v
    parts = {"val": part(vi, sj.knn_energies(X[sub], lab[sub], X[vi], len(Y))), "test": part(ei, sj.knn_energies(X[tr], lab[tr], X[ei], len(Y)))}
    ex = load_extra(a.extra, names, Y) if a.extra and os.path.getsize(a.extra) else None   # extra rows join the head TRAIN/REFIT pools only (val/test knn pools unchanged)
    nmd = None
    if a.nomatch:   # NONE rows; leakage guard vs val/test prompts of pairs.jsonl
        nz = lambda t: " ".join(t.lower().split()); nm_, nmd = load_nomatch(a.nomatch, names, Y, {nz(pairs[i]["q"]) for i in np.r_[vi, ei]}); print("nomatch rows", len(nm_["lab"]), "dropped (in val/test)", nmd, flush=True)
        ex = nm_ if ex is None else {k: (np.r_[ex[k], nm_[k]] if k in ("lab", "plab", "sid") else np.vstack([ex[k], nm_[k]]) if k in ("X", "B", "Pr") else ex[k] + nm_[k]) for k in ex}
    if ex:
        m_ = len(ex["lab"]); en = np.arange(n, n + m_); si, ti = np.r_[si, en], np.r_[ti, en]; X, lab, plab, B, Pr = np.vstack([X, ex["X"]]), np.r_[lab, ex["lab"]], np.r_[plab, ex["plab"]], np.vstack([B, ex["B"]]), np.vstack([Pr, ex["Pr"]])
        allp, allp_prev = allp + ex["allp"], allp_prev + ex["allp_prev"]
    if a.distill:
        ux, ukeep = load_unl(unl_rows, names, Y, held, os.path.join(pe.DATA, "emb_teacher_unl.npz")); m_ = len(ukeep); en = np.arange(len(lab), len(lab) + m_); si, ti = np.r_[si, en], np.r_[ti, en]
        X, lab, plab, B, Pr = np.vstack([X, ux["X"]]), np.r_[lab, ux["lab"]], np.r_[plab, ux["plab"]], np.vstack([B, ux["B"]]), np.vstack([Pr, ux["Pr"]]); allp, allp_prev = allp + ux["allp"], allp_prev + ux["allp_prev"]
        unl = np.r_[unl, np.ones(m_, bool)]; TQ = np.vstack([TQ, np.stack([TV[nzq(r["q"])] for r in ukeep])]) if m_ else TQ; print("distill: teacher rows", len(TV), "unlabelled kept", m_, "train rows with teacher", int((~np.isnan(TQ[si, 0])).sum()), flush=True)
    parts["train"] = part(si, loo(si), random.Random(3), True)
    if not a.no_refit: parts["refit"] = part(ti, loo(ti), random.Random(4), True)
    parts = {k: parts[k] for k in ("train", "val", "test", "refit") if k in parts}
    lt_ = S["test"]["lab"]; res, ms = [], []
    if a.distill:
        sweep(a, parts, S, K, nm, lt_, len(TV), len(ukeep), t0); return
    for sd in range(a.seeds): m, r = run(sd, parts, not a.no_attn, refit=not a.no_refit); res.append(r); ms.append(m); print(nm, sd, r, flush=True)
    agg = lambda k: {q: [round(float(np.mean([r[k][q] for r in res])), 3), round(float(np.std([r[k][q] for r in res])), 3)] for q in res[0][k]}
    out = json.load(open(a.out, encoding="utf-8")) if os.path.exists(a.out) else {}
    info = {"K": K, "union": union, "bm25_feats": lexf, "refit": not a.no_refit, "recallK": {k: round(p["recall"], 3) for k, p in parts.items()}, "n_out_of_k": {k: p["out"] for k, p in parts.items()},
            "rows": {k: p["n"] for k, p in parts.items()}, "params": sum(p.numel() for p in ms[0][0].parameters()), "secs": round(time.time() - t0, 1), "prior_rows_test": int((Pr[ei].sum(1) > 0).sum())} | ({"extra_rows": len(ex["lab"])} if ex else {}) | ({"nomatch_file": a.nomatch, "nomatch_dropped_leak": nmd} if a.nomatch else {})
    out.update({"n": {"train_head": int(len(parts["train"]["y"])), "val": len(parts["val"]["y"]), "test": len(lt_), "test_labelled": int((lt_ >= 0).sum()), "test_none": int((lt_ < 0).sum()), "refit": len(parts["refit"]["y"]) if "refit" in parts else None},
                "base_knn": S["base"], "recall8": round(S["recall8"], 3), "always_abstain_acc": round(float((lt_ < 0).mean()), 3), nm + "_head": agg("argmax"), nm + "_head_thr": agg("thr"), nm + "_runs": res, nm + "_info": info})
    if not a.no_refit: out.update({nm + "_refit": agg("refit_argmax"), nm + "_refit_thr": agg("refit_thr")})
    if a.name == "avb2" and not a.no_attn: out.update({"recallK": info["recallK"], "n_out_of_k": info["n_out_of_k"], "K": K}); torch.save({"state": [[m.state_dict() for m in s] for s in ms], "din": parts["train"]["T"].shape[2], "K": K}, a.out.replace(".json", ".pt"))
    if a.save:   # 10-28: torch-free head file for opsd col 8 / holdout_check / trace_merge gate: val-fit head of the seed with the lowest val CE + its val-tuned none thr (avb_np.py format)
        import avb_np; b = min(range(len(res)), key=lambda i: res[i]["val_ce"])
        avb_np.save_head(a.save, ms[b][0].state_dict(), {"thr": res[b]["thr_val"], "K": K, "din": int(parts["train"]["T"].shape[2]), "seed": b, "val_ce": res[b]["val_ce"], "epochs": res[b]["epochs"], "extra_rows": len(ex["lab"]) if ex else 0, "created": time.strftime("%F %T"), "src": a.out})
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1); print(json.dumps({k: out[k] for k in out if k.startswith(nm) and not k.endswith("_runs")} | {"n": out["n"]}, indent=1))


if __name__ == "__main__":
    main()
