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
  py -3.13 skilljepa.py embedctx         embed "previous prompt + prompt" texts into data/emb_ctx.npz (for --ctx cat)
  py -3.13 skilljepa.py eval [--split session|prompt] cross-validation vs the zero-shot cosine baseline
  py -3.13 skilljepa.py train            fit on all pairs, write data/model.npz
  py -3.13 skilljepa.py route "prompt"   top-3 skills + energies (or abstain); --prev "text" --prev-skill name --hist N
--split time tests on the last 30% of every chat (each chat brings its own past). The gate routers use zero-shot energies
for a chat with little labelled history and the trained predictor once it has GATE_K prompts (soft: blend by n/(n+k)).
--ctx picks the context: none (prompt only), prev (+ previous prompt), prevskill (+ previous prompt and the skill
the previous turn used), cat (one embedding of "previous | now" text). Follow-ups like "ok, do it" carry no skill
signal on their own; the previous turn does.
"""
import argparse, json, os, sys, urllib.request
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")   # 24 BLAS threads on a busy CPU made a 64x768 step take 0.4 s
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")
EMB_URL = os.environ.get("SKILLJEPA_EMB_URL", "http://127.0.0.1:8084/v1/embeddings")
GATE_K = 5   # labelled prompts a chat needs before the gate trusts the trained predictor
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


def prev_index(pairs):
    """Index of the previous prompt in the same chat (pairs.jsonl keeps transcript order), -1 for a chat's first."""
    return np.array([i - 1 if i and pairs[i - 1]["sid"] == p["sid"] else -1 for i, p in enumerate(pairs)])


def cmd_embedctx():
    pairs, _, _ = load(); prv = prev_index(pairs)
    texts = [Q_PRE + (f"previous: {pairs[j]['q'][:600]} | now: " if j >= 0 else "") + p["q"] for p, j in zip(pairs, prv)]
    np.savez(os.path.join(D, "emb_ctx.npz"), X=embed(texts))


def with_ctx(X, Y, prv, plab, ctx, b=0.5, c=0.5):
    """Mix the previous turn into the prompt embedding (unit vectors in, unit vectors out)."""
    if ctx == "none": return X
    has = (prv >= 0)[:, None]
    Z = X + b * np.where(has, X[np.maximum(prv, 0)], 0)
    if ctx == "prevskill": Z = Z + c * np.where((plab >= 0)[:, None], Y[np.maximum(plab, 0)], 0)
    return Z / np.linalg.norm(Z, axis=1, keepdims=True)


def sep_input(X, Y, prv, plab):
    """ctx sep: the prevskill mix as the residual path, plus the previous skill's target as extra predictor input
    (zeros when the previous turn used none), so the hidden layer can learn "follow-up = same skill" explicitly."""
    S = np.where((plab >= 0)[:, None], Y[np.maximum(plab, 0)], 0).astype(np.float32)
    return np.hstack([with_ctx(X, Y, prv, plab, "prevskill"), S])


def data(ctx="none"):
    pairs, names, _ = load()
    z = np.load(os.path.join(D, "emb.npz"))
    assert list(z["names"]) == names and len(z["X"]) == len(pairs), "data changed: rerun embed"
    idx = {n: i for i, n in enumerate(names)}
    lab = np.array([idx.get(p["skills"][0], -1) for p in pairs])      # -1 = "none" (should abstain)
    allp = [set(idx[s] for s in p["skills"] if s in idx) for p in pairs]
    prv = prev_index(pairs); plab = np.where(prv >= 0, lab[np.maximum(prv, 0)], -1)
    X = np.load(os.path.join(D, "emb_ctx.npz"))["X"] if ctx == "cat" else with_ctx(z["X"], z["Y"], prv, plab, ctx)
    if ctx == "sep": X = sep_input(z["X"], z["Y"], prv, plab)
    return X, z["Y"], lab, allp, np.array([p["sid"] for p in pairs]), names, plab


class Predictor:
    def __init__(s, d, h=512, seed=0, dout=None):
        r = np.random.default_rng(seed); dout = dout or d       # d > dout: extra input columns (ctx sep: previous skill)
        s.W1 = r.normal(0, dout ** -0.5, (d, h)).astype(np.float32); s.b1 = np.zeros(h, np.float32)
        s.W2 = np.zeros((h, dout), np.float32)                   # starts as identity: g(x) = x = zero-shot baseline
        s.ps = [s.W1, s.b1, s.W2]; s.m = [np.zeros_like(p) for p in s.ps]; s.v = [np.zeros_like(p) for p in s.ps]; s.t = 0

    def fwd(s, x):
        a = x @ s.W1 + s.b1; h = np.maximum(a, 0); p = x[:, :s.W2.shape[1]] + h @ s.W2
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
    P = Predictor(X.shape[1], seed=seed, dout=Y.shape[1]); r = np.random.default_rng(seed); tr = np.where(lab >= 0)[0]
    for _ in range(epochs):
        r.shuffle(tr)
        for i in range(0, len(tr), bs): P.step(X[tr[i:i + bs]], lab[tr[i:i + bs]], Y, **kw)
    return P


def energies(P, X, Y):
    return 1 - (P.fwd(X)[0] if P else X[:, :Y.shape[1]]) @ Y.T


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


def gate(Ez, Ej, hist, k=GATE_K, soft=False, taus=None):
    """Router gate: zero-shot energies for a chat with little labelled history, the trained predictor once it has some.
    hist = labelled prompts this chat already had in training; soft blends with weight hist / (hist + k).
    taus = (tau_zero_shot, tau_predictor), each tuned on its own branch, blended with the same weight per prompt."""
    w = (hist / (hist + k) if soft else (hist >= k).astype(np.float32))
    E = (1 - w[:, None]) * Ez + w[:, None] * Ej
    return (E, (1 - w) * taus[0] + w * taus[1]) if taus else E


def cmd_eval(folds, split="session", ctx="none", **kw):
    X, Y, lab, allp, sid, _, plab = data(ctx); chat = sid.copy()
    if split == "prompt": sid = np.arange(len(sid)).astype(str)   # projects seen in training (ongoing chats)
    us = np.unique(sid); r = np.random.default_rng(0); r.shuffle(us); grp = {s: i % folds for i, s in enumerate(us)}
    f = np.array([grp[s] for s in sid])
    if split == "time":   # one fold: the last 30% of every chat is the test set, so chats arrive with their own past
        pos = np.zeros(len(chat)); n = {}
        for i, c in enumerate(chat): pos[i] = n.get(c, 0); n[c] = pos[i] + 1
        f = (pos < 0.7 * np.array([n[c] for c in chat])).astype(int); folds = 1
    res = {"zero_shot_cosine": [], "knn_prompts": [], "skill_jepa": [], "repeat_prev_skill": [], "gate_hard": [], "gate_soft": []}
    for k in range(folds):
        tr, te = f != k, f == k
        if not (lab[te] >= 0).any(): continue
        P = fit(X[tr], Y, lab[tr], **kw)
        atr, ate = [allp[i] for i in np.where(tr)[0]], [allp[i] for i in np.where(te)[0]]
        cnt = {}
        for c in chat[tr & (lab >= 0)]: cnt[c] = cnt.get(c, 0) + 1
        hist = np.array([cnt.get(c, 0) for c in chat], np.float32) - (tr & (lab >= 0))   # a train prompt is not its own history
        for name, model in (("zero_shot_cosine", None), ("knn_prompts", "knn"), ("skill_jepa", P), ("repeat_prev_skill", "rep"),
                            ("gate_hard", "gh"), ("gate_soft", "gs")):
            if model in ("gh", "gs"):
                Ez, Ej = energies(None, X, Y), energies(P, X, Y)
                taus = (tune_tau(Ez[tr], lab[tr], atr), tune_tau(Ej[tr], lab[tr], atr))
                Eg, tau = gate(Ez, Ej, hist, soft=model == "gs", taus=taus)
                rep = report(Eg[te], lab[te], ate, tau[te])
            elif model == "rep":   # baseline: route to whatever skill the previous turn used, abstain if it used none
                E = np.ones((len(lab), len(Y)), np.float32); E[plab >= 0, plab[plab >= 0]] = 0
                rep = report(E[te], lab[te], ate, 0.5)
            elif model == "knn":   # tau tuned on the train fold leave-one-out would cost more; reuse train energies with k+1
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
    out["split"], out["ctx"] = split, ctx
    out["n"] = {"prompts": len(lab), "labelled": int((lab >= 0).sum()), "skills": len(Y), "sessions": len(np.unique(chat)), "folds": folds}
    print(json.dumps(out, indent=1)); json.dump(out, open(os.path.join(D, f"eval-{split}-{ctx}.json"), "w"), indent=1)


def cmd_train(ctx="none", **kw):
    X, Y, lab, allp, _, names, _ = data(ctx); P = fit(X, Y, lab, **kw)
    tau, tau_z = tune_tau(energies(P, X, Y), lab, allp), tune_tau(energies(None, X, Y), lab, allp)
    np.savez(os.path.join(D, "model.npz"), W1=P.W1, b1=P.b1, W2=P.W2, Y=Y, names=np.array(names), tau=tau, tau_z=tau_z, ctx=ctx)
    print("model.npz written, tau", round(tau, 3))


def cmd_route(q, prev=None, prev_skill=None, hist=None):
    z = np.load(os.path.join(D, "model.npz")); P = Predictor(z["W1"].shape[0], dout=z["W2"].shape[1]); P.W1, P.b1, P.W2 = z["W1"], z["b1"], z["W2"]
    ctx, names = str(z["ctx"]) if "ctx" in z else "none", list(z["names"])
    if ctx == "cat":
        x = embed([Q_PRE + (f"previous: {prev[:600]} | now: " if prev else "") + q])
    else:
        x = embed([Q_PRE + q] + ([Q_PRE + prev] if prev else []))
        X = np.vstack([x[1:], x[:1]]) if prev else x; prv = np.array([-1, 0] if prev else [-1])
        ps = names.index(prev_skill) if prev_skill in names else -1; plab = np.array([-1, ps] if prev else [ps])
        x = (sep_input(X, z["Y"], prv, plab) if ctx == "sep" else with_ctx(X, z["Y"], prv, plab, ctx))[-1:]
    e, tau = energies(P, x, z["Y"]), float(z["tau"])
    if hist is not None:
        e, t = gate(energies(None, x, z["Y"]), e, np.array([hist], np.float32), soft=True,
                    taus=(float(z["tau_z"]) if "tau_z" in z else tau, tau)); tau = float(t[0])
    e = e[0]; o = e.argsort()[:3]
    print(json.dumps({"route": None if e[o[0]] > tau else str(z["names"][o[0]]),
                      "top3": [[str(z["names"][i]), round(float(e[i]), 3)] for i in o], "tau": round(tau, 3)}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["embed", "embedctx", "eval", "train", "route"]); ap.add_argument("q", nargs="?")
    ap.add_argument("--folds", type=int, default=5); ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=2.0); ap.add_argument("--split", choices=["session", "prompt", "time"], default="session")
    ap.add_argument("--ctx", choices=["none", "prev", "prevskill", "cat", "sep"], default="none")
    ap.add_argument("--prev"); ap.add_argument("--prev-skill")
    ap.add_argument("--hist", type=int, help="labelled prompts this chat has had (turns on the soft gate)")
    a = ap.parse_args()
    kw = {"epochs": a.epochs, "alpha": a.alpha}
    {"embed": cmd_embed, "embedctx": cmd_embedctx, "eval": lambda: cmd_eval(a.folds, a.split, a.ctx, **kw),
     "train": lambda: cmd_train(a.ctx, **kw), "route": lambda: cmd_route(a.q, a.prev, a.prev_skill, a.hist)}[a.cmd]()
