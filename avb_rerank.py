"""avb_rerank.py: AVB-style decision head over EXISTING embeddings (no LLM pass). Order-invariant self-attention across 8 knn candidates + 1 learned "none" token -> one logit each.
  C:/llm/qlora/venv/Scripts/python.exe avb_rerank.py [--no-attn] [--seeds 3] [--out data/avb-<date>.json]   (CPU; --no-attn = per-choice MLP ablation, merged into the same json)
Rows/splits/candidates exactly as policy_eval.build (time split: test = last 30% of each chat; VAL = train positions [0.49,0.70); TRAIN of the head = positions < 0.49 with leave-one-out knn).
Token = [query(prevskill ctx), cand skill emb, query*cand, cos, knn energy, rank/8, cand==prev skill, cand in prev row's used set, is_none]; no positional encoding. Early stop on VAL CE."""
import argparse, datetime, json, os, random, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import policy_eval as pe, skilljepa as sj
DATA = pe.DATA; K = 8


def feats(Xq, Y, E, cands, plab, allp_prev):
    """-> (n, 9, 3d+6) float32; slot 8 = none token (cand emb 0, scalars = query summary + is_none)."""
    n, d = Xq.shape; c = np.array(cands); e = np.take_along_axis(E, c, 1); cos = np.einsum("nd,nkd->nk", Xq, Y[c]); rank = e.argsort(1).argsort(1) / K
    ip = (c == plab[:, None]).astype(np.float32); ia = np.array([[s in pp for s in row] for row, pp in zip(c, allp_prev)], np.float32)
    T = np.zeros((n, 9, 3 * d + 6), np.float32); T[:, :8, :d] = Xq[:, None]; T[:, :8, d:2 * d] = Y[c]; T[:, :8, 2 * d:3 * d] = Xq[:, None] * Y[c]
    T[:, :8, 3 * d:] = np.stack([cos, e, rank, ip, ia, np.zeros_like(cos)], 2); T[:, 8, :d] = Xq
    T[:, 8, 3 * d:] = np.stack([cos.max(1), e.min(1), 0 * cos[:, 0], (plab < 0).astype(np.float32), 0 * cos[:, 0], np.ones(n)], 1); return T


class Head(nn.Module):
    def __init__(s, din, d=256, attn=True, p=0.1):
        super().__init__(); s.inp = nn.Sequential(nn.Dropout(p), nn.Linear(din, d), nn.GELU()); s.attn = attn
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 512, p, batch_first=True, norm_first=True), 2) if attn else \
            nn.Sequential(*[nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 512), nn.GELU(), nn.Dropout(p), nn.Linear(512, d)) for _ in range(2)])
        s.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 128), nn.GELU(), nn.Linear(128, 1))
    def forward(s, T):
        h = s.inp(T); h = s.enc(h) if s.attn else h + s.enc(h); return s.out(h).squeeze(-1)    # (n, 9)


def target(lab, allp, cands):
    """label skill if among the 8 candidates, else any used skill among them, else -1 (row dropped); none -> 8."""
    t = np.full(len(lab), -1)
    for i, (l, a, c) in enumerate(zip(lab, allp, cands)):
        if l < 0: t[i] = 8
        elif l in c: t[i] = c.index(l)
        else:
            h = [j for j, s in enumerate(c) if s in a]; t[i] = h[0] if h else -1
    return t


def run(seed, parts, attn, ep=60, pat=8):
    torch.manual_seed(seed); np.random.seed(seed); tr, va = parts["train"], parts["val"]
    m = Head(tr["T"].shape[2], attn=attn); opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-2)
    Xt, yt = torch.tensor(tr["T"]), torch.tensor(tr["y"]); Xv, yv = torch.tensor(va["T"]), torch.tensor(va["y"]); ok = yv >= 0; best = (9e9, None, 0)
    for e in range(ep):
        m.train(); perm = torch.randperm(len(Xt))
        for b in range(0, len(perm), 64):
            i = perm[b:b + 64]; loss = F.cross_entropy(m(Xt[i]), yt[i]); opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad(): vl = F.cross_entropy(m(Xv)[ok], yv[ok]).item()
        if vl < best[0]: best = (vl, {k: v.clone() for k, v in m.state_dict().items()}, e)
        elif e - best[2] >= pat: break
    m.load_state_dict(best[1]); m.eval(); sc = lambda P: m(torch.tensor(P["T"])).detach().numpy()
    Zv, Zt = sc(parts["val"]), sc(parts["test"]); thr = pe.tune_thr(Zv, parts["val"]); pa, _, _ = pe.metrics(Zt, parts["test"]); pt, _, _ = pe.metrics(Zt, parts["test"], thr)
    return m, {"argmax": pa, "thr": pt, "thr_val": round(thr, 4), "epochs": best[2] + 1, "val_ce": round(best[0], 4)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--no-attn", action="store_true"); ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--out"); a = ap.parse_args()
    t0 = time.time(); S = pe.build(); X, Y, lab, allp, sid, names, plab = sj.data("prevskill", "all"); n = len(lab)
    pos, cnt = np.zeros(n), {}
    for i, c in enumerate(sid): pos[i] = cnt.get(c, 0); cnt[c] = pos[i] + 1
    tot = np.array([cnt[c] for c in sid]); te = pos >= 0.7 * tot; tr = ~te; sub = pos < pe.VAL_LO * tot
    allp_prev = [allp[i - 1] if i and sid[i - 1] == sid[i] else set() for i in range(n)]
    def part(P, E):   # P = policy_eval part (idx, cands, lab, allp), E = knn energies of its rows
        i = P["idx"]; return {"T": feats(X[i], Y, E, P["cands"], plab[i], [allp_prev[j] for j in i]), "y": target(P["lab"], P["allp"], P["cands"]), **{k: P[k] for k in ("lab", "cands", "allp")}}
    # head TRAIN rows: sub part, leave-one-out knn within sub (self never a neighbour), candidates = top-8 shuffled
    si = np.where(sub)[0]; Xs, ls = X[si], lab[si]; Sm = Xs @ Xs[ls >= 0].T; lt = ls[ls >= 0]; Sm[np.arange(len(si))[:, None] == np.where(ls >= 0)[0][None]] = -9
    nn_ = np.argsort(-Sm, 1)[:, :10]; Etr = np.ones((len(si), len(Y)), np.float32)
    for r in range(len(si)): np.add.at(Etr[r], lt[nn_[r]], -Sm[r, nn_[r]] / 10)
    rng = random.Random(3); cs = [list(map(int, c)) for c in Etr.argsort(1)[:, :K]]; [rng.shuffle(c) for c in cs]
    parts = {"train": part({"idx": si, "cands": cs, "lab": lab[si], "allp": [allp[i] for i in si]}, Etr),
             "val": part(S["val"], sj.knn_energies(X[sub], lab[sub], X[S["val"]["idx"]], len(Y))), "test": part(S["test"], sj.knn_energies(X[tr], lab[tr], X[S["test"]["idx"]], len(Y)))}
    parts["train"] = {**parts["train"], "T": parts["train"]["T"][parts["train"]["y"] >= 0], "y": parts["train"]["y"][parts["train"]["y"] >= 0]}
    lt_ = S["test"]["lab"]; nm = "avb_noattn" if a.no_attn else "avb_head"; res, ms = [], []
    for sd in range(a.seeds): m, r = run(sd, parts, not a.no_attn); res.append(r); ms.append(m); print(nm, sd, r, flush=True)
    agg = lambda k: {q: [round(float(np.mean([r[k][q] for r in res])), 3), round(float(np.std([r[k][q] for r in res])), 3)] for q in res[0][k]}
    out = json.load(open(a.out, encoding="utf-8")) if a.out and os.path.exists(a.out) else {}
    out.update({"n": {"train_head": int(len(parts["train"]["y"])), "val": len(parts["val"]["y"]), "test": len(lt_), "test_labelled": int((lt_ >= 0).sum()), "test_none": int((lt_ < 0).sum())},
                "base_knn": S["base"], "recall8": round(S["recall8"], 3), "always_abstain_acc": round(float((lt_ < 0).mean()), 3), "params": sum(p.numel() for p in ms[0].parameters()),
                nm: agg("argmax"), nm + "_thr": agg("thr"), nm + "_runs": res, nm + "_secs": round(time.time() - t0, 1)})
    if not a.no_attn: torch.save({"state": [m.state_dict() for m in ms], "din": parts["train"]["T"].shape[2]}, a.out.replace(".json", ".pt"))
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1); print(json.dumps({k: out[k] for k in (nm, nm + "_thr", "n", "params", nm + "_secs")}, indent=1))


if __name__ == "__main__":
    main()
