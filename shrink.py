"""shrink.py: self-shrinking test of the AVB head (reuses arms.py data/split/part/combine/fit_arm). CPU only.
  C:/llm/qlora/venv/Scripts/python.exe shrink.py [--seeds 2] [--out data/shrink-20261029.json] [--md notes/shrink-20261029.md]
Variants: width d in {256,128,64,32,16} x blocks {2,1,0} (d=256,b=2 baseline); feature-group drops on baseline.
Selection (pre-registered): smallest params with VAL hit1 >= baseline VAL hit1 - 0.005 on BOTH splits -> report TEST vs baseline. Prunable group: val hit1 drop < 0.005 on both splits."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse, hashlib, json, random, sys, time
import numpy as np, torch, torch.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import arms, avb_rerank as ar, skilljepa as sj, avb_feats as af
torch.set_num_threads(int(os.environ.get("SHRINK_THREADS", "3")))


class SHead(nn.Module):
    def __init__(s, din, d=256, blocks=2, p=0.1):
        super().__init__(); s.inp = nn.Sequential(nn.Dropout(p), nn.Linear(din, d), nn.GELU())
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 2 * d, p, batch_first=True, norm_first=True), blocks, enable_nested_tensor=False) if blocks else nn.Identity()
        s.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, max(d // 2, 8)), nn.GELU(), nn.Linear(max(d // 2, 8), 1))
    def forward(s, T, with_other=False):
        return s.out(s.enc(s.inp(T))).squeeze(-1)


def setup(a):
    """arms.main() setup up to parts per split -> dict sp -> (parts, pv, pt, vi, ei, R_of, tokf, names)."""
    X, Y, lab, allp, sid, names, plab = sj.data("prevskill", "all"); n = len(lab); K = 16
    pairs = [json.loads(l) for l in open(os.path.join(arms.D, "pairs.jsonl"), encoding="utf-8")]; B, Pr = af.lex(pairs, names)
    pos, cnt = np.zeros(n), {}
    for i, c in enumerate(sid): pos[i] = cnt.get(c, 0); cnt[c] = pos[i] + 1
    tot = np.array([cnt[c] for c in sid]); allp_prev = [allp[i - 1] if i and sid[i - 1] == sid[i] else set() for i in range(n)]
    h = lambda s, salt: int(hashlib.md5((salt + s).encode()).hexdigest()[:8], 16) % 100
    te_s = np.array([h(s, "test") < 30 for s in sid]); va_s = np.array([(not t) and h(s, "val") < 20 for s, t in zip(sid, te_s)]); tr_s = ~te_s & ~va_s
    tr_t, sub = pos < 0.7 * tot, pos < 0.49 * tot
    splits = {"time": (np.where(sub)[0], np.where(tr_t & ~sub)[0], np.where(~tr_t)[0], np.where(tr_t)[0]), "session": (np.where(tr_s)[0], np.where(va_s)[0], np.where(te_s)[0], np.where(~te_s)[0])}
    a0 = arms.A0(); need = sorted(set(np.r_[splits["time"][1], splits["time"][2], splits["session"][1], splits["session"][2]].tolist())); pool = {i: a0.score(pairs[i]) for i in need}
    nm_ = lambda s_: {names[j] for j in s_}
    def R_of(idx, rank=None, top=None):
        rk = rank if rank is not None else [[nm for nm, _ in pool[i][:arms.K_SHOW]] for i in idx]; tp = top if top is not None else np.array([pool[i][0][1] if pool[i] else -9.0 for i in idx])
        return {"top": tp, "rank": rk, "gold": [nm_(allp[i]) for i in idx], "none": lab[idx] < 0}
    def loo(si, k=10):
        Xs, ls = X[si], lab[si]; lt = ls[ls >= 0]; Sm = Xs @ Xs[ls >= 0].T; Sm[np.arange(len(si))[:, None] == np.where(ls >= 0)[0][None]] = -9
        nn_ = np.argsort(-Sm, 1)[:, :k]; E = np.ones((len(si), len(Y)), np.float32)
        for r in range(len(si)): np.add.at(E[r], lt[nn_[r]], -Sm[r, nn_[r]] / k)
        return E
    def part(i, E, shuf=None, fit=False):
        cs = af.cands(E, B[i], K, True)
        if shuf: [shuf.shuffle(c) for c in cs]
        y, out = af.target(lab[i], [allp[j] for j in i], cs, False); T = af.feats(X[i], Y, E, B[i], Pr[i], cs, plab[i], [allp_prev[j] for j in i], True); k = (y >= 0) if fit else np.ones(len(y), bool)
        return {"T": T[k], "y": y[k], "lab": lab[i], "cands": cs, "allp": [allp[j] for j in i], "n": len(i)}
    S = {}
    for sp, (si, vi, ei, ti) in splits.items():
        parts = {"train": part(si, loo(si), random.Random(3), True), "val": part(vi, sj.knn_energies(X[si], lab[si], X[vi], len(Y))), "test": part(ei, sj.knn_energies(X[ti], lab[ti], X[ei], len(Y)))}
        S[sp] = dict(parts=parts, pv=[pool[i][:arms.POOL] for i in vi], pt=[pool[i][:arms.POOL] for i in ei], vi=vi, ei=ei, R_of=R_of)
    return S, names, a0.brief_tok


def evaluate(sp_, names, tokf, seeds, mk):
    """B1 path for one split + head factory mk(din) -> dict(val hit1/acc mean, test hit1/acc mean, params, ms100)."""
    ar.Head = lambda din, attn=True, other=False: mk(din); parts = sp_["parts"]; rows = []
    for sd in range(seeds):
        ms, r = ar.run(sd, parts, True, refit=False); m = ms[0]
        with torch.no_grad(): Zv, Zt = (m(torch.tensor(parts[k]["T"])).numpy()[:, :-1] for k in ("val", "test"))
        per = {}
        for dl in arms.DELTAS:
            rkv, tpv, _ = arms.combine(sp_["pv"], parts["val"]["cands"], Zv, names, dl); rkt, tpt, _ = arms.combine(sp_["pt"], parts["test"]["cands"], Zt, names, dl)
            per[dl] = arms.fit_arm(sp_["R_of"](sp_["vi"], rkv, tpv), sp_["R_of"](sp_["ei"], rkt, tpt), tokf)
        best = max(arms.DELTAS, key=lambda dl: (per[dl]["val"]["hit1"] + per[dl]["val"]["acc_with_abstain"], -dl)); rows.append(per[best])
    x = torch.tensor(parts["val"]["T"][:100]); m.eval(); ts = []
    with torch.no_grad():
        m(x)
        for _ in range(5): t = time.perf_counter(); m(x); ts.append(time.perf_counter() - t)
    g = lambda w, k: float(np.mean([r[w][k] for r in rows]))
    return {"val_hit1": g("val", "hit1"), "val_acc": g("val", "acc_with_abstain"), "test_hit1": g("test", "hit1"), "test_acc": g("test", "acc_with_abstain"), "params": sum(p.numel() for p in m.parameters()), "ms100": 1000 * float(np.median(ts))}


def drop_parts(sp_, cols):
    P = {}
    for k, v in sp_["parts"].items():
        T = v["T"].copy(); T[:, :, cols] = 0; P[k] = dict(v, T=T)
    return dict(sp_, parts=P)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=2); ap.add_argument("--out", default=os.path.join(arms.D, "shrink-20261029.json")); ap.add_argument("--only", default=""); ap.add_argument("--merge", action="store_true"); ap.add_argument("--md", default=os.path.join(HERE, "notes", "shrink-20261029.md")); a = ap.parse_args()
    t0 = time.time()
    if not a.merge:
        S, names, tokf = setup(a); din = S["time"]["parts"]["train"]["T"].shape[2]; d3 = (din - 12) // 3; print("setup", round(time.time() - t0), "s din", din, "emb", d3, flush=True)
        s0 = 3 * d3; G = {"query_emb": range(0, d3), "cand_emb": range(d3, 2 * d3), "query*cand": range(2 * d3, 3 * d3), "knn(cos,energy,rank)": [s0, s0 + 1, s0 + 2], "prev-skill(is_prev,in_prev_used)": [s0 + 3, s0 + 4],
             "bm25(raw,rel,logrank,top1,top3)": [s0 + 6, s0 + 7, s0 + 8, s0 + 9, s0 + 10], "prior": [s0 + 11], "is_none_flag": [s0 + 5]}
        want = [x for x in a.only.split("|") if x]; R = {}
        for d in (256, 128, 64, 32, 16):
            for b in (2, 1, 0):
                nm = f"d{d}b{b}"
                if want and nm not in want and "grid" not in want: continue
                R[nm] = {sp: evaluate(S[sp], names, tokf, a.seeds, lambda din, d=d, b=b: SHead(din, d, b)) for sp in S}; print(nm, {sp: {k: round(v, 3) for k, v in r.items()} for sp, r in R[nm].items()}, round(time.time() - t0), flush=True)
        for g, cols in G.items():
            nm = "drop:" + g
            if want and nm not in want and "drops" not in want: continue
            R[nm] = {sp: evaluate(drop_parts(S[sp], list(cols)), names, tokf, a.seeds, lambda din: SHead(din, 256, 2)) for sp in S}; print(nm, {sp: round(r["val_hit1"], 3) for sp, r in R[nm].items()}, round(time.time() - t0), flush=True)
        json.dump(R, open(a.out, "w"), indent=1); print("part done", round(time.time() - t0), "s"); return
    import glob; R = {}
    for f in sorted(glob.glob(os.path.join(arms.D, "shrink-part-*.json"))): R.update(json.load(open(f)))
    S = {"time": 0, "session": 0}; base = R["d256b2"]
    # session VAL has only 3 rows in arms.py's split (no labelled rows -> NaN): rule is applied on the splits with a defined val hit1 (time only if session is NaN)
    use = [sp for sp in S if not np.isnan(base[sp]["val_hit1"])]
    ok = [k for k in R if not k.startswith("drop:") and all(R[k][sp]["val_hit1"] >= base[sp]["val_hit1"] - 0.005 for sp in use)]; ch = min(ok, key=lambda k: R[k]["time"]["params"])
    pr = [k for k in R if k.startswith("drop:") and all(base[sp]["val_hit1"] - R[k][sp]["val_hit1"] < 0.005 for sp in use)]
    a.seeds = a.seeds
    L = ["# shrink 2026-10-29: smallest AVB head keeping held-out quality", "", f"Index: chosen {ch} ({R[ch]['time']['params']} params vs {base['time']['params']} baseline); prunable groups: {', '.join(p[5:] for p in pr) or 'none'}; seeds {a.seeds}, B1 path via arms.py, CPU; runtime {round(time.time() - t0)} s.", "",
         "Splits used for the rule: " + "+".join(use) + ". Note: session VAL has only 3 rows in arms.py (no labelled rows, hit1 NaN), so the rule is evaluated on time only; session TEST shown for reference. Rule (pre-registered): smallest params with VAL hit1 >= baseline VAL hit1 - 0.005 on BOTH splits. Prunable = val hit1 drop < 0.005 on both splits. Numbers = mean over seeds (delta picked on VAL per seed).", "",
         "| variant | params | ms/100rows | time val h1 | time test h1 | time test acc | sess val h1 | sess test h1 | sess test acc |", "|---|---|---|---|---|---|---|---|---|"]
    for k, r in R.items(): t, s = r["time"], r["session"]; L.append(f"| {k} | {t['params']} | {t['ms100']:.1f} | {t['val_hit1']:.3f} | {t['test_hit1']:.3f} | {t['test_acc']:.3f} | {s['val_hit1']:.3f} | {s['test_hit1']:.3f} | {s['test_acc']:.3f} |")
    L += ["", f"## Chosen: {ch}", ""] + [f"- {sp}: test hit1 {R[ch][sp]['test_hit1']:.3f} vs baseline {base[sp]['test_hit1']:.3f}; test acc_with_abstain {R[ch][sp]['test_acc']:.3f} vs {base[sp]['test_acc']:.3f}" for sp in S]
    L += ["", "## Prunable feature groups", ""] + ([f"- {p[5:]}: val hit1 drop time {base['time']['val_hit1'] - R[p]['time']['val_hit1']:+.3f}, session {base['session']['val_hit1'] - R[p]['session']['val_hit1']:+.3f}" for p in pr] or ["- none"])
    os.makedirs(os.path.dirname(a.md), exist_ok=True); open(a.md, "w", encoding="utf-8").write("\n".join(L) + "\n"); print("\n".join(L[:6])); print("done", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
