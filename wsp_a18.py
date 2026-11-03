"""wsp_a18.py: A18 weight-space programming test (Takehi et al. 2026 "Sparks of Weight-Space Programming"; harness idea from github.com/leofmarciano/weightspace, not its code).
An LLM (Claude in this chat) edits a COPY of data/avb_head.npz by reasoning about the weights, no gradients. Probes = VAL split only (positions [0.49,0.70), knn pool = rows < 0.49); budget 25 probes.
Frozen holdout scored ONCE per final candidate via holdout_check.score. Prod head untouched.
  py -3.13 -I wsp_a18.py --cache                 build VAL tensors -> data/wsp_val.npz (+ .json cands/allp/lab)
  py -3.13 -I wsp_a18.py --probe HEAD.npz        VAL metrics (af.metrics at head thr + argmax) and count of weights changed vs prod; logs to data/wsp_a18.log"""
import json, os, sys, time
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(_v, "1")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "data"); sys.path.insert(0, HERE)
VC, VJ, LOG, PROD, BUDGET = os.path.join(D, "wsp_val.npz"), os.path.join(D, "wsp_val.json"), os.path.join(D, "wsp_a18.log"), os.path.join(D, "avb_head.npz"), 25


def cache():
    import avb_feats as af, policy_eval as pe, avb_np, holdout_check as hc
    sj, X, Y, lab, allp, sid, names, plab, te = hc._base(); n = len(lab); pos, cnt = np.zeros(n), {}
    for i, c in enumerate(sid): pos[i] = cnt.get(c, 0); cnt[c] = pos[i] + 1
    tot = np.array([cnt[c] for c in sid]); sub = pos < pe.VAL_LO * tot; vi = np.where(~te & ~sub)[0]
    pairs = [json.loads(l) for l in open(os.path.join(D, "pairs.jsonl"), encoding="utf-8")]; B, Pr = af.lex([pairs[i] for i in vi], names)
    allp_prev = [allp[i - 1] if i and sid[i - 1] == sid[i] else set() for i in range(n)]; K = avb_np.load_head(PROD)["__meta__"]["K"]
    E = sj.knn_energies(X[sub], lab[sub], X[vi], len(Y)); cs = af.cands(E, B, K, True); T = af.feats(X[vi], Y, E, B, Pr, cs, plab[vi], [allp_prev[j] for j in vi], True)
    np.savez_compressed(VC, T=T); json.dump({"cands": cs, "allp": [sorted(allp[j]) for j in vi], "lab": lab[vi].tolist(), "i": vi.tolist()}, open(VJ, "w"))
    print(f"VAL rows {len(vi)} labelled {int((lab[vi] >= 0).sum())} T {T.shape}")


def load_val():
    j = json.load(open(VJ)); return np.load(VC)["T"], {"cands": j["cands"], "allp": [set(a) for a in j["allp"]], "lab": np.array(j["lab"])}


def changed(W):
    P0 = np.load(PROD, allow_pickle=True); tot = ch = 0
    for k in P0.files:
        if k == "__meta__": continue
        tot += P0[k].size; ch += int((P0[k] != W[k]).sum())
    return ch, tot


def probe(path):
    import avb_feats as af, avb_np
    used = sum(1 for l in open(LOG, encoding="utf-8") if '"probe"' in l) if os.path.exists(LOG) else 0
    if used >= BUDGET: sys.exit(f"probe budget {BUDGET} spent")
    T, P = load_val(); W = avb_np.load_head(path); Z = avb_np.forward(W, T); ch, tot = changed(W)
    r = {"probe": used + 1, "head": os.path.basename(path), "t": time.strftime("%F %T"), "thr": af.metrics(Z, P, W["__meta__"]["thr"]), "argmax": af.metrics(Z, P), "changed": ch, "frac": round(ch / tot, 6)}
    open(LOG, "a", encoding="utf-8").write(json.dumps(r) + "\n"); print(json.dumps(r))


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--cache" in a: cache()
    elif "--probe" in a: probe(a[a.index("--probe") + 1])
    else: print(__doc__)
