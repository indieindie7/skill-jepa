"""Skill-JEPA prototype: route a prompt to an agent skill by predicting in a frozen embedding space.

JEPA shape, text edition:
  context encoder  = frozen EmbeddingGemma-300M on the prompt          -> x (768-d, unit)
  target encoder   = the same frozen model on "name: description"     -> y_s for every skill s
  predictor        = small residual MLP g, trained so g(x) lands on y_s of the skill the agent really used
  energy           = 1 - cos(g(x), y_s); route = argmin energy; abstain when the best energy > tau
Nothing is generated and the encoders never train (a frozen target encoder is the EMA limit), so training
is seconds of numpy on CPU. Skills with no training pairs still route through the residual path.
The skill-target negatives in the InfoNCE term play the anti-collapse role VICReg plays in image JEPAs.

  py -3.13 skilljepa.py embed            embed prompts + skills into data/emb.npz (needs the embedding server)
  py -3.13 skilljepa.py eval [--split session|prompt] cross-validation vs the zero-shot cosine baseline
  py -3.13 skilljepa.py train            fit on all pairs, write data/model.npz
  py -3.13 skilljepa.py route "prompt"   top-3 skills + energies (or abstain)
"""
import argparse, json, os, sys, urllib.request
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")   # 24 BLAS threads on a busy CPU made a 64x768 step take 0.4 s
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")
EMB_URL = os.environ.get("SKILLJEPA_EMB_URL", "http://127.0.0.1:8084/v1/embeddings")
Q_PRE, S_PRE = "task: search result | query: ", "title: {} | text: "   # EmbeddingGemma prompt formats


def embed(texts, bs=8):
    out = []
    for i in range(0, len(texts), bs):
        body = json.dumps({"input": [t[:1500] for t in texts[i:i + bs]]}).encode()
        req = urllib.request.Request(EMB_URL, body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = sorted(json.load(r)["data"], key=lambda e: e["index"])
        out += [e["embedding"] for e in data]
        print(f"\r  {min(i + bs, len(texts))}/{len(texts)}", end="", file=sys.stderr)
    print(file=sys.stderr)
    m = np.array(out, np.float32)
    return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)


def load():
    pairs = [json.loads(l) for l in open(os.path.join(D, "pairs.jsonl"), encoding="utf-8")]
    skills = json.load(open(os.path.join(D, "skills.json"), encoding="utf-8"))
    return pairs, sorted(skills), skills


def cmd_embed():
    pairs, names, skills = load()
    np.savez(os.path.join(D, "emb.npz"), X=embed([Q_PRE + p["q"] for p in pairs]),
             Y=embed([S_PRE.format(n) + skills[n] for n in names]), names=np.array(names))


def data():
    pairs, names, _ = load()
    z = np.load(os.path.join(D, "emb.npz"))
    assert list(z["names"]) == names and len(z["X"]) == len(pairs), "data changed: rerun embed"
    idx = {n: i for i, n in enumerate(names)}
    lab = np.array([idx.get(p["skills"][0], -1) for p in pairs])      # -1 = "none" (should abstain)
    allp = [set(idx[s] for s in p["skills"] if s in idx) for p in pairs]
    return z["X"], z["Y"], lab, allp, np.array([p["sid"] for p in pairs]), names


class Predictor:
    def __init__(s, d, h=512, seed=0):
        r = np.random.default_rng(seed)
        s.W1 = r.normal(0, d ** -0.5, (d, h)).astype(np.float32); s.b1 = np.zeros(h, np.float32)
        s.W2 = np.zeros((h, d), np.float32)                      # starts as identity: g(x) = x = zero-shot baseline
        s.ps = [s.W1, s.b1, s.W2]; s.m = [np.zeros_like(p) for p in s.ps]; s.v = [np.zeros_like(p) for p in s.ps]; s.t = 0

    def fwd(s, x):
        a = x @ s.W1 + s.b1; h = np.maximum(a, 0); p = x + h @ s.W2
        n = np.linalg.norm(p, axis=1, keepdims=True); return p / n, (x, a, h, n)

    def step(s, x, y_idx, Y, alpha=0.5, temp=0.05, lr=1e-3, wd=1e-4):
        ph, (x, a, h, n) = s.fwd(x); B = len(x); Yt = Y[y_idx]
        reg = 1 - (ph * Yt).sum(1)                                # latent regression (the JEPA loss)
        lg = ph @ Y.T / temp; lg -= lg.max(1, keepdims=True); pr = np.exp(lg); pr /= pr.sum(1, keepdims=True)
        nce = -np.log(pr[np.arange(B), y_idx] + 1e-9)
        g = -Yt / B                                               # d reg / d ph
        oh = np.zeros_like(pr); oh[np.arange(B), y_idx] = 1
        g += alpha * (pr - oh) @ Y / temp / B                     # d nce / d ph
        gp = (g - (g * ph).sum(1, keepdims=True) * ph) / n        # back through the unit-norm
        gW2 = h.T @ gp; ga = (gp @ s.W2.T) * (a > 0); gW1 = x.T @ ga; gb1 = ga.sum(0)
        s.t += 1
        for i, (p, gr) in enumerate(zip(s.ps, [gW1, gb1, gW2])):  # AdamW
            s.m[i] = 0.9 * s.m[i] + 0.1 * gr; s.v[i] = 0.999 * s.v[i] + 0.001 * gr * gr
            mh = s.m[i] / (1 - 0.9 ** s.t); vh = s.v[i] / (1 - 0.999 ** s.t)
            p -= lr * (mh / (np.sqrt(vh) + 1e-8) + wd * p)
        return float(reg.mean()), float(nce.mean())


def fit(X, Y, lab, epochs=40, bs=64, seed=0, **kw):
    P = Predictor(X.shape[1], seed=seed); r = np.random.default_rng(seed); tr = np.where(lab >= 0)[0]
    for _ in range(epochs):
        r.shuffle(tr)
        for i in range(0, len(tr), bs): P.step(X[tr[i:i + bs]], lab[tr[i:i + bs]], Y, **kw)
    return P


def energies(P, X, Y):
    return 1 - (P.fwd(X)[0] if P else X) @ Y.T


def knn_energies(Xtr, labtr, X, nY, k=10):
    """Baseline that ignores skill descriptions: vote of the k nearest labelled training prompts."""
    L = labtr >= 0; S = X @ Xtr[L].T; lt = labtr[L]; nn = np.argsort(-S, 1)[:, :k]; E = np.ones((len(X), nY), np.float32)
    for i in range(len(X)):
        for j in nn[i]: E[i, lt[j]] -= S[i, j] / k
    return E


def tune_tau(E, lab, allp):
    """tau maximising routing accuracy: correct = top-1 in the used set, or abstain on a 'none' prompt."""
    best, top, emin = (-1, 1.0), E.argmin(1), E.min(1)
    hit = np.array([t in s for t, s in zip(top, allp)])
    for tau in np.quantile(emin, np.linspace(0, 1, 101)):
        best = max(best, (np.where(emin > tau, lab < 0, hit).mean(), float(tau)))
    return best[1]


def report(E, lab, allp, tau):
    L = lab >= 0; rk = E.argsort(1); top = rk[:, 0]; emin = E.min(1)
    hit = np.array([t in s for t, s in zip(top, allp)])
    route = emin <= tau
    return {"hit1": hit[L].mean(), "hit3": np.mean([bool(set(rk[i, :3]) & allp[i]) for i in np.where(L)[0]]),
            "acc_with_abstain": np.where(~route, lab < 0, hit).mean(),
            "false_route_on_none": route[~L].mean() if (~L).any() else 0.0, "abstain_on_labelled": (~route[L]).mean()}


def cmd_eval(folds, split="session", **kw):
    X, Y, lab, allp, sid, _ = data()
    if split == "prompt": sid = np.arange(len(sid)).astype(str)   # projects seen in training (ongoing chats)
    us = np.unique(sid); r = np.random.default_rng(0); r.shuffle(us); grp = {s: i % folds for i, s in enumerate(us)}
    f = np.array([grp[s] for s in sid]); res = {"zero_shot_cosine": [], "knn_prompts": [], "skill_jepa": []}
    for k in range(folds):
        tr, te = f != k, f == k
        if not (lab[te] >= 0).any(): continue
        P = fit(X[tr], Y, lab[tr], **kw)
        atr, ate = [allp[i] for i in np.where(tr)[0]], [allp[i] for i in np.where(te)[0]]
        for name, model in (("zero_shot_cosine", None), ("knn_prompts", "knn"), ("skill_jepa", P)):
            if model == "knn":   # tau tuned on the train fold leave-one-out would cost more; reuse train energies with k+1
                tau = tune_tau(knn_energies(X[tr], lab[tr], X[tr], len(Y)), lab[tr], atr)
                rep = report(knn_energies(X[tr], lab[tr], X[te], len(Y)), lab[te], ate, tau)
            else:
                tau = tune_tau(energies(model, X[tr], Y), lab[tr], atr)
                rep = report(energies(model, X[te], Y), lab[te], ate, tau)
            res[name].append((rep, int(te.sum())))
    out = {}
    for name, rows in res.items():
        w = np.array([n for _, n in rows], float)
        out[name] = {m: round(float(np.average([r[m] for r, _ in rows], weights=w)), 3) for m in rows[0][0]}
    out["always_abstain_acc"] = round(float((lab < 0).mean()), 3)
    out["split"] = split
    out["n"] = {"prompts": len(lab), "labelled": int((lab >= 0).sum()), "skills": len(Y), "sessions": len(us), "folds": folds}
    print(json.dumps(out, indent=1)); json.dump(out, open(os.path.join(D, f"eval-{split}.json"), "w"), indent=1)


def cmd_train(**kw):
    X, Y, lab, allp, _, names = data(); P = fit(X, Y, lab, **kw)
    tau = tune_tau(energies(P, X, Y), lab, allp)
    np.savez(os.path.join(D, "model.npz"), W1=P.W1, b1=P.b1, W2=P.W2, Y=Y, names=np.array(names), tau=tau)
    print("model.npz written, tau", round(tau, 3))


def cmd_route(q):
    z = np.load(os.path.join(D, "model.npz")); P = Predictor(z["W1"].shape[0]); P.W1, P.b1, P.W2 = z["W1"], z["b1"], z["W2"]
    e = energies(P, embed([Q_PRE + q]), z["Y"])[0]; o = e.argsort()[:3]
    print(json.dumps({"route": None if e[o[0]] > z["tau"] else str(z["names"][o[0]]),
                      "top3": [[str(z["names"][i]), round(float(e[i]), 3)] for i in o], "tau": round(float(z["tau"]), 3)}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["embed", "eval", "train", "route"]); ap.add_argument("q", nargs="?")
    ap.add_argument("--folds", type=int, default=5); ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=2.0); ap.add_argument("--split", choices=["session", "prompt"], default="session")
    a = ap.parse_args()
    kw = {"epochs": a.epochs, "alpha": a.alpha}
    {"embed": cmd_embed, "eval": lambda: cmd_eval(a.folds, a.split, **kw), "train": lambda: cmd_train(**kw),
     "route": lambda: cmd_route(a.q)}[a.cmd]()
