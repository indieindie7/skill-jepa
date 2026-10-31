"""a13b_shrink.py: A13b session-aware shrink selection. Grouped 5-fold (by sid) over TRAIN+VAL rows, per split; mean CV hit1 (head argmax, held-out fold).
  C:/llm/qlora/venv/Scripts/python.exe a13b_shrink.py --seed 0 --out data/a13b-s0.json   (CPU only)
Fold: head trained on 4/5 sids (15% of those sids = inner early-stop val), scored on held-out fold. TEST (shrink.evaluate) reported for reference only."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse, hashlib, json, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import arms, shrink, avb_rerank as ar, skilljepa as sj
ARMS = [(256, 2), (64, 1), (32, 1), (32, 0), (16, 1), (16, 0)]
cat = lambda a, b, k: np.concatenate([a[k], b[k]]) if not isinstance(a[k], list) else a[k] + b[k]


def pool(parts, sid):
    tr, va = parts["train"], parts["val"]
    d = {k: cat(tr, va, k) for k in ("T", "y", "lab", "cands", "allp", "idx")}
    d["sid"] = np.array([sid[i] for i in d["idx"]]); return d


def sub(d, m):
    ix = np.where(m)[0]; return {"T": d["T"][ix], "y": d["y"][ix], "lab": d["lab"][ix], "cands": [d["cands"][i] for i in ix], "allp": [d["allp"][i] for i in ix], "n": len(ix)}


def cv(d, mk, seed, drop=None):
    us = np.unique(d["sid"]); rs = np.random.RandomState(100 + seed); nl = {s: int(((d["sid"] == s) & (d["lab"] >= 0)).sum()) for s in us}; order = sorted(rs.permutation(us), key=lambda s: -nl[s]); load = np.zeros(5); f_of = {}
    for s in order: f = int(np.argmin(load + 1e-3 * np.bincount(list(f_of.values()), minlength=5))) if f_of else 0; f_of[s] = f; load[f] += nl[s]   # greedy: balance labelled rows per fold
    fo = np.array([f_of[s] for s in d["sid"]]); hs = []
    ar.Head = lambda din, attn=True, other=False: mk(din)
    if drop is not None: d = dict(d, T=d["T"].copy()); d["T"][:, :, drop] = 0
    for f in range(5):
        tr_s = [s for s in us if f_of[s] != f]; r2 = np.random.RandomState(7 * f + seed); trl = [s for s in tr_s if nl[s] > 0]; tru = [s for s in tr_s if nl[s] == 0]; iv = set(r2.choice(trl, max(1, int(.2 * len(trl))), replace=False)) | set(r2.choice(tru, int(.15 * len(tru)), replace=False) if tru else [])
        iv_m = np.array([s in iv for s in d["sid"]]); ho = fo == f; trm = ~ho & ~iv_m; vam = ~ho & iv_m
        ms, r = ar.run(seed, {"train": sub(d, trm), "val": sub(d, vam), "test": sub(d, ho)}, True, refit=False); hs.append((r["argmax"]["hit1"], int((d["lab"][ho] >= 0).sum())))
    w = np.array([h[1] for h in hs], float); return float(np.nanmean([h[0] for h in hs])), float(np.average([h[0] for h in hs], weights=w))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True); ap.add_argument("--only", default=""); ap.add_argument("--noref", action="store_true"); a = ap.parse_args(); t0 = time.time()
    S, names, tokf = shrink.setup(a); X, Y, lab, allp, sid, nm2, plab = sj.data("prevskill", "all")
    din = S["time"]["parts"]["train"]["T"].shape[2]; d3 = (din - 12) // 3; s0 = 3 * d3
    G = {"query_emb": range(0, d3), "query*cand": range(2 * d3, 3 * d3), "is_none_flag": [s0 + 5], "bm25": [s0 + 6, s0 + 7, s0 + 8, s0 + 9, s0 + 10]}
    P = {sp: pool(S[sp]["parts"], sid) for sp in S}; R = {}; want = [x for x in a.only.split("|") if x]
    def save(): json.dump(R, open(a.out, "w"), indent=1)
    for d, b in ARMS:
        nm = f"d{d}b{b}"
        if want and nm not in want: continue
        mk = lambda din, d=d, b=b: shrink.SHead(din, d, b); R[nm] = {}
        for sp in S:
            c, cw = cv(P[sp], mk, a.seed); params = sum(p.numel() for p in mk(din).parameters()); r = {"cv_hit1": c, "cv_hit1_w": cw, "params": params, "n_pool": int(len(P[sp]["y"]))}
            if not a.noref: e = shrink.evaluate(S[sp], names, tokf, 1, mk); r.update(test_hit1=e["test_hit1"], test_acc=e["test_acc"], ms100=e["ms100"])
            R[nm][sp] = r
        print(nm, {sp: {k: round(v, 3) for k, v in r.items()} for sp, r in R[nm].items()}, round(time.time() - t0), flush=True); save()
    # drops on smallest arm whose CV holds both splits (this seed) else d16b1
    base = R.get("d256b2")
    if base and not want:
        ok = [k for k in R if k != "d256b2" and all(R[k][sp]["cv_hit1"] >= base[sp]["cv_hit1"] - .005 for sp in S)]; tgt = min(ok, key=lambda k: R[k]["time"]["params"]) if ok else "d16b1"; d, b = int(tgt[1:].split("b")[0]), int(tgt[-1])
        for g, cols in G.items():
            nm = f"{tgt}-drop:{g}"; R[nm] = {}
            for sp in S: c, cw = cv(P[sp], lambda din: shrink.SHead(din, d, b), a.seed, list(cols)); R[nm][sp] = {"cv_hit1": c, "cv_hit1_w": cw}
            print(nm, {sp: round(r["cv_hit1"], 3) for sp, r in R[nm].items()}, round(time.time() - t0), flush=True); save()
    print("done", round(time.time() - t0))


if __name__ == "__main__":
    main()
