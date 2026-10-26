"""avb_feats.py: v2 feature/candidate/metric helpers for avb_rerank.py. BM25 + prev-skill prior lifted from ops/treeroute_eval.py (bm25_root L53, prev_prior L85; tokenizer + stop list = opsd._tok/_STOP):
BM25 over each skill's SKILL.md only (the production 'root' index, but N = the 77 skills of data/skills.json); prior = sum 0.5**(age/1800 s) over the skills this sid read before the prompt (<6 h), from skills/usage.jsonl.
Deviation: a use counts only when ut+60 <= t (usage.jsonl times are minute-truncated, so ut < t could include the read the prompt itself caused = label leak)."""
import collections, datetime, json, math, os, re, time
import numpy as np
SK = os.path.join(os.path.expanduser("~"), ".claude", "skills")
_STOP = set("the a an and or of to in for on with via by from at as is are be this that it its into use using when then than not no yes our we you your their them they what how where which can could should would please lets let's do did does make i me my".split())
tok = lambda s: [w for w in re.findall(r"[a-z][a-z0-9_.+-]{2,}", s.lower()) if w not in _STOP]


def lex(pairs, names, k1=1.2, b=0.75, half_life=1800.0):
    """-> B (n, nS) BM25 score of every skill, P (n, nS) prev_prior of every skill, at each row's t/sid."""
    tf, ln = [], []
    for nm in names:
        try: body = open(os.path.join(SK, nm, "SKILL.md"), encoding="utf-8", errors="replace").read()
        except OSError: body = ""
        c = collections.Counter(tok(body)); tf.append(c); ln.append(sum(c.values()))
    N = len(names); avg = sum(ln) / N; df = collections.Counter()
    for c in tf: df.update(c.keys())
    idf = {w: math.log(1 + (N - d + 0.5) / (d + 0.5)) for w, d in df.items()}; nrm = [k1 * (1 - b + b * l / avg) for l in ln]
    B = np.zeros((len(pairs), N), np.float32); P = np.zeros_like(B); ix = {n: i for i, n in enumerate(names)}; use = collections.defaultdict(list)
    for l in open(os.path.join(SK, "usage.jsonl"), encoding="utf-8"):
        try: j = json.loads(l); t = time.mktime(time.strptime(j["t"], "%Y-%m-%d %H:%M"))
        except (ValueError, KeyError): continue
        if j.get("skill") in ix and j.get("sid"): use[j["sid"]].append((t, ix[j["skill"]]))
    for i, p in enumerate(pairs):
        for w in set(tok(p["q"])):
            if w in idf:
                for j in range(N):
                    f = tf[j].get(w)
                    if f: B[i, j] += idf[w] * f * (k1 + 1) / (f + nrm[j])
        t = datetime.datetime.fromisoformat(p["t"].replace("Z", "+00:00")).timestamp()
        for ut, j in use.get(p["sid"], ()):
            if ut + 60 <= t < ut + 6 * 3600: P[i, j] += 0.5 ** ((t - ut) / half_life)
    return B, P


def cands(E, B, K, union=True):
    """union: interleave knn rank r and BM25 rank r (BM25 score > 0 only), dedupe, until K = union of the knn top-m and BM25 top-m (m adaptive), always filled to K from knn order. else knn top-K."""
    kr = E.argsort(1, kind="stable")
    if not union: return [list(map(int, r[:K])) for r in kr]
    br = (-B).argsort(1, kind="stable"); out = []
    for i in range(len(E)):
        c = []
        for r in range(E.shape[1]):
            for j in (kr[i, r], br[i, r] if B[i, br[i, r]] > 0 else -1):
                if j >= 0 and j not in c and len(c) < K: c.append(int(j))
            if len(c) >= K: break
        out.append(c)
    return out


def feats(Xq, Y, E, B, Pr, cs, plab, allp_prev, lexf=True):
    """-> (n, K+1, 3d+6[+6]); slot K = none token. scalars: cos, knn energy, knn rank in cands/K, cand==prev skill, cand in prev row's used set, is_none | BM25 raw/10, raw/max, log global rank, top1, top3, prior."""
    n, d = Xq.shape; c = np.array(cs); K = c.shape[1]; g = lambda M: np.take_along_axis(M, c, 1)
    e = g(E); cos = np.einsum("nd,nkd->nk", Xq, Y[c]); rank = e.argsort(1).argsort(1) / K; z = np.zeros_like(cos)
    ip = (c == plab[:, None]).astype(np.float32); ia = np.array([[s in pp for s in row] for row, pp in zip(c, allp_prev)], np.float32)
    cols = [cos, e, rank, ip, ia, z]; nz = np.zeros(n); nn = [cos.max(1), e.min(1), nz, (plab < 0).astype(np.float32), nz, np.ones(n)]
    if lexf:
        gr = g((-B).argsort(1).argsort(1)); cols += [g(B) / 10, g(B) / np.maximum(B.max(1, keepdims=True), 1e-6), np.log1p(gr) / math.log(B.shape[1]), gr < 1, gr < 3, g(Pr)]
        nn += [B.max(1) / 10, nz, nz, nz, nz, Pr.sum(1)]
    S = np.stack(cols, 2).astype(np.float32); T = np.zeros((n, K + 1, 3 * d + S.shape[2]), np.float32)
    T[:, :K, :d] = Xq[:, None]; T[:, :K, d:2 * d] = Y[c]; T[:, :K, 2 * d:3 * d] = Xq[:, None] * Y[c]; T[:, :K, 3 * d:] = S; T[:, K, :d] = Xq; T[:, K, 3 * d:] = np.stack(nn, 1); return T


def target(lab, allp, cs, drop=False):
    """label's index among the candidates, else the first used skill among them, else none (= K); drop=True (v1): -1 = row dropped. -> (y, n_out_of_k = labelled rows with no used skill among the candidates)."""
    K = len(cs[0]); t = np.full(len(lab), K); out = 0
    for i, (l, a, c) in enumerate(zip(lab, allp, cs)):
        if l < 0: continue
        if l in c: t[i] = c.index(l)
        else:
            h = [j for j, s in enumerate(c) if s in a]
            if h: t[i] = h[0]
            else: out += 1; t[i] = -1 if drop else K
    return t, out


def pnone(Z):
    z = Z.astype(np.float64); z -= z.max(1, keepdims=True); e = np.exp(z); return e[:, -1] / e.sum(1)


def hits(Z, P):
    rl = np.argsort(-Z[:, :-1], 1); c = P["cands"]; h = lambda j, k: bool(set(c[j][x] for x in rl[j, :k]) & P["allp"][j])
    return np.array([h(j, 1) for j in range(len(c))]), np.array([h(j, 3) for j in range(len(c))])


def metrics(Z, P, thr=None):
    """thr None: route = argmax is a skill slot; else route = P(none) < thr. Same definitions as policy_eval.metrics, none slot = last column."""
    lt = P["lab"]; L = lt >= 0; h1, h3 = hits(Z, P); route = (Z.argmax(1) != Z.shape[1] - 1) if thr is None else (pnone(Z) < thr)
    m = {"hit1": h1[L].mean(), "hit3": h3[L].mean(), "acc_with_abstain": np.where(~route, lt < 0, h1).mean(), "false_route_on_none": route[~L].mean(), "abstain_on_labelled": (~route[L]).mean(), "hit1_strict": (route & h1)[L].mean()}
    return {k: round(float(v), 3) for k, v in m.items()}


def tune_thr(Zv, Pv):
    """thr maximising acc_with_abstain + hit1 on VAL (policy_eval.tune_thr, any K)."""
    pn = pnone(Zv); lt = Pv["lab"]; L = lt >= 0; h1, _ = hits(Zv, Pv); best = (-1.0, 0.0)
    for t in np.r_[0.0, np.quantile(pn, np.linspace(0, 1, 101)), 1.0001]: best = max(best, (np.where(pn < t, h1, lt < 0).mean() + h1[L].mean(), float(t)))
    return best[1]
