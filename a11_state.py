"""a11_state.py: ledger A11 drill. J0 prompt-only (=B1 d16b0), J1 + prev same-sid prompt, J2 + compact STATE (task/active/frontier/unresolved/transition) instead of prev prompt. Causal: only earlier same-sid rows.
  C:/llm/qlora/venv/Scripts/python.exe a11_state.py [--seeds 3] [--out data/a11-20261029.json] [--md notes/a11-state-20261029.md]   (CPU)"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse, hashlib, json, random, sys, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import arms, shrink, skilljepa as sj, avb_feats as af, avb_rerank as ar
torch.set_num_threads(int(os.environ.get("SHRINK_THREADS", "3")))
ARMS = {"J0": [], "J1": ["prevq"], "J2": ["task", "active", "frontier", "unres", "trans"]}
ABL = {"J2-task": ["active", "frontier", "unres", "trans"], "J2-active": ["task", "frontier", "unres", "trans"], "J2-frontier": ["task", "active", "unres", "trans"],
       "J2-unres": ["task", "active", "frontier", "trans"], "J2-trans": ["task", "active", "frontier", "unres"]}


def build(a):
    X, Y, lab, allp, sid, names, plab = sj.data("prevskill", "all"); n = len(lab); K = 16; d = X.shape[1]
    pairs = [json.loads(l) for l in open(os.path.join(arms.D, "pairs.jsonl"), encoding="utf-8")]; B, Pr = af.lex(pairs, names)
    pos, cnt = np.zeros(n), {}
    for i, c in enumerate(sid): pos[i] = cnt.get(c, 0); cnt[c] = pos[i] + 1
    tot = np.array([cnt[c] for c in sid]); allp_prev = [allp[i - 1] if i and sid[i - 1] == sid[i] else set() for i in range(n)]
    h = lambda s, salt: int(hashlib.md5((salt + s).encode()).hexdigest()[:8], 16) % 100
    te_s = np.array([h(s, "test") < 30 for s in sid]); va_s = np.array([(not t) and h(s, "val") < 20 for s, t in zip(sid, te_s)]); tr_s = ~te_s & ~va_s
    tr_t, sub = pos < 0.7 * tot, pos < 0.49 * tot
    splits = {"time": (np.where(sub)[0], np.where(tr_t & ~sub)[0], np.where(~tr_t)[0], np.where(tr_t)[0]), "session": (np.where(tr_s)[0], np.where(va_s)[0], np.where(te_s)[0], np.where(~te_s)[0])}
    a0 = arms.A0(frozen=True); t0 = time.time(); pool = {i: a0.score(pairs[i]) for i in range(n)}; print("A0 all rows", round(time.time() - t0), "s", flush=True)
    nm_ = lambda s_: {names[j] for j in s_}
    def R_of(idx, rank=None, top=None):
        rk = rank if rank is not None else [[nm for nm, _ in pool[i][:arms.K_SHOW]] for i in idx]; tp = top if top is not None else np.array([pool[i][0][1] if pool[i] else -9.0 for i in idx])
        return {"top": tp, "rank": rk, "gold": [nm_(allp[i]) for i in idx], "none": lab[idx] < 0}
    nidx = {nm: j for j, nm in enumerate(names)}
    Xp = np.zeros((n, d), np.float32); X3 = np.zeros((n, d), np.float32); has_prev = np.zeros(n, np.float32); front = np.zeros((n, len(names)), np.float32); unres = np.zeros(n, np.float32); has2 = np.zeros(n, np.float32); chg = np.zeros(n, np.float32)
    for i in range(n):   # causal: earlier same-sid rows only; never row i, never row i's gold
        j = []
        for k in range(i - 1, max(i - 4, -1), -1):
            if sid[k] != sid[i]: break
            j.append(k)
        if j:
            Xp[i] = X[j[0]]; has_prev[i] = 1; X3[i] = X[j].mean(0); p = pool[j[0]]
            for nm, _ in p[:arms.POOL]: front[i, nidx[nm]] = 1
            unres[i] = float((not p) or p[0][1] < arms.NOMATCH)
            if len(j) > 1: has2[i] = 1; chg[i] = float(allp[j[0]] != allp[j[1]])
    def extras(idx, cs, use):
        nb = len(idx); c = np.array(cs); Kk = c.shape[1]; blocks = []; g = lambda M: np.take_along_axis(M, c, 1)
        def pad(arr):
            o = np.zeros((nb, Kk + 1) + arr.shape[2:], np.float32); o[:, :Kk] = arr; return o
        Yc = Y[c]
        if "prevq" in use:
            blocks += [pad(Xp[idx][:, None] * Yc), pad(np.repeat(np.einsum("nd,nd->n", Xp[idx], X[idx])[:, None, None], Kk, 1)), pad(np.repeat(has_prev[idx][:, None, None], Kk, 1))]
        if "task" in use:
            blocks += [pad(X3[idx][:, None] * Yc), pad(np.einsum("nd,nkd->nk", X3[idx], Yc)[..., None])]
        if "active" in use: blocks.append(pad(np.array([[s in allp_prev[i] for s in row] for row, i in zip(c, idx)], np.float32)[..., None]))
        if "frontier" in use: blocks.append(pad(g(front[idx])[..., None]))
        if "unres" in use: blocks.append(np.repeat(unres[idx][:, None, None], Kk + 1, 1))
        if "trans" in use: blocks.append(np.repeat(np.stack([has2[idx], chg[idx]], 1)[:, None, :], Kk + 1, 1))
        return np.concatenate(blocks, 2).astype(np.float32) if blocks else np.zeros((nb, Kk + 1, 0), np.float32)
    def loo(si, k=10):
        Xs, ls = X[si], lab[si]; lt = ls[ls >= 0]; Sm = Xs @ Xs[ls >= 0].T; Sm[np.arange(len(si))[:, None] == np.where(ls >= 0)[0][None]] = -9
        nn_ = np.argsort(-Sm, 1)[:, :k]; E = np.ones((len(si), len(Y)), np.float32)
        for r in range(len(si)): np.add.at(E[r], lt[nn_[r]], -Sm[r, nn_[r]] / k)
        return E
    def part(i, E, shuf=None, fit=False):
        cs = af.cands(E, B[i], K, True)
        if shuf: [shuf.shuffle(c) for c in cs]
        y, out = af.target(lab[i], [allp[j] for j in i], cs, False); T = af.feats(X[i], Y, E, B[i], Pr[i], cs, plab[i], [allp_prev[j] for j in i], True); k = (y >= 0) if fit else np.ones(len(y), bool)
        return {"T": T, "k": k, "y": y[k], "idx": i, "cs": cs, "lab": lab[i], "cands": cs, "allp": [allp[j] for j in i], "n": len(i)}
    S = {}
    for sp, (si, vi, ei, ti) in splits.items():
        parts = {"train": part(si, loo(si), random.Random(3), True), "val": part(vi, sj.knn_energies(X[si], lab[si], X[vi], len(Y))), "test": part(ei, sj.knn_energies(X[ti], lab[ti], X[ei], len(Y)))}
        S[sp] = dict(parts=parts, pv=[pool[i][:arms.POOL] for i in vi], pt=[pool[i][:arms.POOL] for i in ei], vi=vi, ei=ei, R_of=R_of)
    return S, names, a0.brief_tok, extras


HEAD = (16, 0)


def run_arm(sp_, names, tokf, extras, use):
    parts = {}
    for k, v in sp_["parts"].items():
        T = np.concatenate([v["T"], extras(v["idx"], v["cs"], use)], 2) if use else v["T"]; parts[k] = dict(v, T=T[v["k"]])
    return shrink.evaluate(dict(sp_, parts=parts), names, tokf, 1, lambda din: shrink.SHead(din, *HEAD))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--head", default="d16b0"); ap.add_argument("--noabl", action="store_true"); ap.add_argument("--out", default=os.path.join(arms.D, "a11-20261029.json")); ap.add_argument("--md", default=os.path.join(HERE, "notes", "a11-state-20261029.md")); a = ap.parse_args()
    global HEAD
    if a.head == "d256b2": HEAD = (256, 2)
    t0 = time.time(); S, names, tokf, extras = build(a); R = {}; orig = ar.run
    for nm, use in list(ARMS.items()) + ([] if a.noabl else list(ABL.items())):
        R[nm] = {}
        for sp in (["time", "session"] if nm in ARMS else ["time"]):
            runs = []
            for sd in range(a.seeds):
                ar.run = lambda s, *args, _sd=sd, **kw: orig(_sd, *args, **kw); runs.append(run_arm(S[sp], names, tokf, extras, use))
            ar.run = orig; tp = S[sp]["parts"]["test"]
            R[nm][sp] = {k: [float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs]))] for k in ("test_hit1", "test_acc", "val_hit1")} | {"din_extra": int(extras(tp["idx"][:1], tp["cs"][:1], use).shape[2]), "raw": [r["test_hit1"] for r in runs]}
        print(nm, {sp: [round(x, 3) for x in v["test_hit1"]] for sp, v in R[nm].items()}, round(time.time() - t0), flush=True)
    n = {sp: {"test": int(S[sp]["parts"]["test"]["n"]), "test_labelled": int((S[sp]["parts"]["test"]["lab"] >= 0).sum())} for sp in S}
    json.dump({"R": R, "n": n}, open(a.out, "w"), indent=1)
    f = lambda v: f"{v[0]:.3f}+-{v[1]:.3f}"
    L = ["# A11 compact state J0/J1/J2 (2026-10-29)", "", "Hypothesis (scans-2026-10-29 sec 4 Stage 3): J2 (compact STATE+prompt) > J1 (prev prompt+prompt) > J0 (prompt only). Pre-registered: J2 must beat J1 by > sd and > .01 on BOTH splits, else NOT SUPPORTED.", "",
         f"Setup: a11_state.py, arms.py machinery (A0 frozen a0_index, B1 path, pairs.jsonl 4185 rows via sj.data, not holdout.jsonl), head {a.head} (SHead{HEAD}), K=16, delta picked on VAL, {a.seeds} seeds, mean+-sd, CPU. Context causal: earlier same-sid rows only; active_skill = prev-row gold (as allp_prev), frontier = prev-row A0 top-{arms.POOL} multi-hot, unresolved = prev-row A0 empty/top<{arms.NOMATCH}, transition = (has prev-prev, changed bit), task = mean emb of last <=3 earlier prompts. J1 adds prevq*cand emb + cos + has_prev; J2 adds task*cand emb + cos + 5 scalars. n: {n}", "",
         "| arm | extra cols | time test hit1 | time test acc | sess test hit1 | sess test acc |", "|---|---|---|---|---|---|"]
    for nm in ARMS: r = R[nm]; L.append(f"| {nm} | {r['time']['din_extra']} | {f(r['time']['test_hit1'])} | {f(r['time']['test_acc'])} | {f(r['session']['test_hit1'])} | {f(r['session']['test_acc'])} |")
    L += ["", "Ablation on time split (J2 minus one component):", "", "| arm | time test hit1 | delta vs J2 |", "|---|---|---|"]
    for nm in (ABL if not a.noabl else []): L.append(f"| {nm} | {f(R[nm]['time']['test_hit1'])} | {R[nm]['time']['test_hit1'][0] - R['J2']['time']['test_hit1'][0]:+.3f} |")
    ok = all(R["J2"][sp]["test_hit1"][0] - R["J1"][sp]["test_hit1"][0] > max(R["J1"][sp]["test_hit1"][1], R["J2"][sp]["test_hit1"][1], 0.01) for sp in ("time", "session"))
    d_ = lambda x, y, sp: R[x][sp]["test_hit1"][0] - R[y][sp]["test_hit1"][0]
    L += ["", "## Verdict", "", "SUPPORTED" if ok else "NOT SUPPORTED", f"(J2-J1 time {d_('J2','J1','time'):+.3f}, session {d_('J2','J1','session'):+.3f}; J1-J0 time {d_('J1','J0','time'):+.3f}, session {d_('J1','J0','session'):+.3f}). Session VAL is 3 rows in the arms.py split, so delta/early-stop on session are VAL-starved and session test is noisy. Not committed."]
    open(a.md, "w", encoding="utf-8").write("\n".join(L) + "\n"); print("\n".join(L)); print("done", round(time.time() - t0))


if __name__ == "__main__":
    main()
