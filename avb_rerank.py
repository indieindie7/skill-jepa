"""avb_rerank.py v2: AVB-style decision head over EXISTING embeddings (no LLM pass). Order-invariant self-attention across K candidates + 1 learned "none" token -> one logit each.
  C:/llm/qlora/venv/Scripts/python.exe avb_rerank.py [--topk 16] [--no-bm25-feats] [--v1] [--no-attn] [--no-refit] [--seeds 3] [--name avb2] --out data/avb-<date>-v2.json   (CPU; every run merges its keys into --out)
Rows/splits as policy_eval.build (time split: test = last 30% of each chat; VAL = train positions [0.49,0.70); head TRAIN = positions < 0.49 with leave-one-out knn). v2 = candidates: knn+BM25 union filled to K (avb_feats.cands);
feats + BM25 score/rank/top1/top3 + prev-skill prior; rows whose gold is outside the K candidates stay, labelled none; early stop on VAL CE + none-thr on VAL, then REFIT on the full train part (positions < 0.70, leave-one-out knn)
for the chosen epoch count with the tuned thr. Keys per run: <name>_head(_thr) = val-fit head, <name>_refit(_thr) = refit head. --v1 = old behaviour (knn top-8, no lex feats, out-of-8 rows dropped, no refit) -> ~avb-20261026.json numbers.
Token = [query(prevskill ctx), cand skill emb, query*cand, scalars of avb_feats.feats]; no positional encoding."""
import argparse, json, os, random, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import policy_eval as pe, skilljepa as sj, avb_feats as af


class Head(nn.Module):
    def __init__(s, din, d=256, attn=True, p=0.1):
        super().__init__(); s.inp = nn.Sequential(nn.Dropout(p), nn.Linear(din, d), nn.GELU()); s.attn = attn
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 512, p, batch_first=True, norm_first=True), 2) if attn else \
            nn.Sequential(*[nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 512), nn.GELU(), nn.Dropout(p), nn.Linear(512, d)) for _ in range(2)])
        s.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 128), nn.GELU(), nn.Linear(128, 1))
    def forward(s, T):
        h = s.inp(T); h = s.enc(h) if s.attn else h + s.enc(h); return s.out(h).squeeze(-1)    # (n, K+1)


def epoch(m, opt, X, y):
    m.train(); perm = torch.randperm(len(X))
    for b in range(0, len(perm), 64):
        i = perm[b:b + 64]; loss = F.cross_entropy(m(X[i]), y[i]); opt.zero_grad(); loss.backward(); opt.step()


def run(seed, parts, attn, ep=60, pat=8, refit=True):
    """val-fit head (early stop on VAL CE, thr on VAL) and, if refit, a fresh head trained on parts['refit'] for the chosen epoch count; both scored on test with the same thr."""
    tr, va = parts["train"], parts["val"]; din = tr["T"].shape[2]; mk = lambda: Head(din, attn=attn)
    torch.manual_seed(seed); np.random.seed(seed); m = mk(); opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-2)
    Xt, yt, Xv, yv = torch.tensor(tr["T"]), torch.tensor(tr["y"]), torch.tensor(va["T"]), torch.tensor(va["y"]); ok = yv >= 0; best = (9e9, None, 0)
    for e in range(ep):
        epoch(m, opt, Xt, yt); m.eval()
        with torch.no_grad(): vl = F.cross_entropy(m(Xv)[ok], yv[ok]).item()
        if vl < best[0]: best = (vl, {k: v.clone() for k, v in m.state_dict().items()}, e)
        elif e - best[2] >= pat: break
    m.load_state_dict(best[1]); m.eval(); sc = lambda mm, P: mm(torch.tensor(P["T"])).detach().numpy()
    thr = af.tune_thr(sc(m, va), va); r = {"thr_val": round(thr, 4), "epochs": best[2] + 1, "val_ce": round(best[0], 4)}; Zt = sc(m, parts["test"])
    r["argmax"], r["thr"] = af.metrics(Zt, parts["test"]), af.metrics(Zt, parts["test"], thr); ms = [m]
    if refit:
        torch.manual_seed(seed); m2 = mk(); opt = torch.optim.AdamW(m2.parameters(), 1e-3, weight_decay=1e-2); rf = parts["refit"]; Xr, yr = torch.tensor(rf["T"]), torch.tensor(rf["y"])
        for e in range(best[2] + 1): epoch(m2, opt, Xr, yr)
        m2.eval(); Z2 = sc(m2, parts["test"]); r["refit_argmax"], r["refit_thr"] = af.metrics(Z2, parts["test"]), af.metrics(Z2, parts["test"], thr); ms.append(m2)
    return ms, r


def load_extra(path, names, Y):
    """--extra rows (trace_merge pairs_trace.jsonl + emb_trace.npz) -> arrays aligned like sj.data('prevskill','all'); prev = previous row of the same sid in the file."""
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]; X0 = np.load(os.path.join(os.path.dirname(path), "emb_trace.npz"))["X"]; assert len(X0) == len(rows), "rerun trace_merge"
    ix = {n: i for i, n in enumerate(names)}; lab = np.array([ix.get(p["skills"][0], -1) for p in rows]); last, prv = {}, np.full(len(rows), -1)
    for i, p in enumerate(rows): prv[i] = last.get(p["sid"], -1); last[p["sid"]] = i
    plab = np.where(prv >= 0, lab[np.maximum(prv, 0)], -1); B, Pr = af.lex(rows, names)
    return {"X": sj.with_ctx(X0, Y, prv, plab, "prevskill"), "lab": lab, "allp": [{l} if l >= 0 else set() for l in lab], "sid": np.array([p["sid"] for p in rows]), "plab": plab, "B": B, "Pr": Pr,
            "allp_prev": [{lab[j]} if j >= 0 and lab[j] >= 0 else set() for j in prv]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--no-attn", action="store_true"); ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--no-bm25-feats", action="store_true"); ap.add_argument("--v1", action="store_true"); ap.add_argument("--no-refit", action="store_true"); ap.add_argument("--name", default="avb2"); ap.add_argument("--out"); ap.add_argument("--extra"); ap.add_argument("--tag"); a = ap.parse_args()
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
        rec = float(np.mean([bool(set(c) & allp[j]) for c, j in zip(cs, i) if lab[j] >= 0]))
        return {"T": T[k], "y": y[k], "lab": lab[i], "cands": cs, "allp": [allp[j] for j in i], "out": out, "n": len(i), "recall": rec}
    def loo(si, k=10):   # head TRAIN rows: leave-one-out knn within si (self never a neighbour)
        Xs, ls = X[si], lab[si]; lt = ls[ls >= 0]; Sm = Xs @ Xs[ls >= 0].T; Sm[np.arange(len(si))[:, None] == np.where(ls >= 0)[0][None]] = -9
        nn_ = np.argsort(-Sm, 1)[:, :k]; E = np.ones((len(si), len(Y)), np.float32)
        for r in range(len(si)): np.add.at(E[r], lt[nn_[r]], -Sm[r, nn_[r]] / k)
        return E
    si, ti = np.where(sub)[0], np.where(tr)[0]; vi, ei = S["val"]["idx"], S["test"]["idx"]
    parts = {"val": part(vi, sj.knn_energies(X[sub], lab[sub], X[vi], len(Y))), "test": part(ei, sj.knn_energies(X[tr], lab[tr], X[ei], len(Y)))}
    ex = load_extra(a.extra, names, Y) if a.extra and os.path.getsize(a.extra) else None   # extra rows join the head TRAIN/REFIT pools only (val/test knn pools unchanged)
    if ex:
        m_ = len(ex["lab"]); en = np.arange(n, n + m_); si, ti = np.r_[si, en], np.r_[ti, en]; X, lab, plab, B, Pr = np.vstack([X, ex["X"]]), np.r_[lab, ex["lab"]], np.r_[plab, ex["plab"]], np.vstack([B, ex["B"]]), np.vstack([Pr, ex["Pr"]])
        allp, allp_prev = allp + ex["allp"], allp_prev + ex["allp_prev"]
    parts["train"] = part(si, loo(si), random.Random(3), True)
    if not a.no_refit: parts["refit"] = part(ti, loo(ti), random.Random(4), True)
    parts = {k: parts[k] for k in ("train", "val", "test", "refit") if k in parts}
    lt_ = S["test"]["lab"]; res, ms = [], []
    for sd in range(a.seeds): m, r = run(sd, parts, not a.no_attn, refit=not a.no_refit); res.append(r); ms.append(m); print(nm, sd, r, flush=True)
    agg = lambda k: {q: [round(float(np.mean([r[k][q] for r in res])), 3), round(float(np.std([r[k][q] for r in res])), 3)] for q in res[0][k]}
    out = json.load(open(a.out, encoding="utf-8")) if os.path.exists(a.out) else {}
    info = {"K": K, "union": union, "bm25_feats": lexf, "refit": not a.no_refit, "recallK": {k: round(p["recall"], 3) for k, p in parts.items()}, "n_out_of_k": {k: p["out"] for k, p in parts.items()},
            "rows": {k: p["n"] for k, p in parts.items()}, "params": sum(p.numel() for p in ms[0][0].parameters()), "secs": round(time.time() - t0, 1), "prior_rows_test": int((Pr[ei].sum(1) > 0).sum())} | ({"extra_rows": len(ex["lab"])} if ex else {})
    out.update({"n": {"train_head": int(len(parts["train"]["y"])), "val": len(parts["val"]["y"]), "test": len(lt_), "test_labelled": int((lt_ >= 0).sum()), "test_none": int((lt_ < 0).sum()), "refit": len(parts["refit"]["y"]) if "refit" in parts else None},
                "base_knn": S["base"], "recall8": round(S["recall8"], 3), "always_abstain_acc": round(float((lt_ < 0).mean()), 3), nm + "_head": agg("argmax"), nm + "_head_thr": agg("thr"), nm + "_runs": res, nm + "_info": info})
    if not a.no_refit: out.update({nm + "_refit": agg("refit_argmax"), nm + "_refit_thr": agg("refit_thr")})
    if a.name == "avb2" and not a.no_attn: out.update({"recallK": info["recallK"], "n_out_of_k": info["n_out_of_k"], "K": K}); torch.save({"state": [[m.state_dict() for m in s] for s in ms], "din": parts["train"]["T"].shape[2], "K": K}, a.out.replace(".json", ".pt"))
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1); print(json.dumps({k: out[k] for k in out if k.startswith(nm) and not k.endswith("_runs")} | {"n": out["n"]}, indent=1))


if __name__ == "__main__":
    main()
