"""avb_serve.py: online AVB v2 head scoring for opsd route.log col 8 (10-28, arms-1028). numpy only (no torch), lazy, never raises to the caller.
  s = AvbServe(); s.col(q, sid8, ["skill-a", "skill-b", ...]) -> "avb=skill-a:1.23,skill-b:nan,_none:-0.4" or "" (head/embedder missing; one line in data/avb_serve.log per outage)
Featurization = avb_rerank part(): query = EmbeddingGemma (8084, CPU) of Q_PRE+q mixed like ctx 'prevskill' (prev prompt of the same sid from an in-process cache + the sid's latest skill read from usage.jsonl),
knn energies vs the labelled pool (pairs.jsonl + pairs_trace.jsonl), union(knn, BM25) K candidates, avb_feats.feats, avb_np.forward. BM25 over SKILL.md of the head's names (= avb_feats.lex) and prior from usage.jsonl.
Deviations from training: prev prompt embedding / allp_prev only known in-process (opsd restart = no prev ctx until a sid's 2nd prompt); allp_prev = {latest skill}. Skills outside the head's candidate list log 'nan'."""
import collections, json, math, os, re, sys, threading, time, urllib.request
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(_v, "1")   # tiny matmuls: 1 thread = 4 ms forward, default threads = 100+ ms (only effective when numpy is not imported yet, true in opsd)
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "data")
if HERE not in sys.path: sys.path.insert(0, HERE)
SK = os.path.join(os.path.expanduser("~"), ".claude", "skills")
EMB_URL = os.environ.get("SKILLJEPA_EMB_URL", "http://127.0.0.1:8084/v1/embeddings")
LOG = os.path.join(D, "avb_serve.log")
RETRY = 300.0   # s between load / outage retries


class AvbServe:
    def __init__(self, head=None):
        self.head = head or os.path.join(D, "avb_head.npz"); self.ok = False; self.next_try = 0.0; self.lock = threading.Lock(); self.last = {}; self.use = {"mt": -1, "rows": []}; self.warned = 0.0

    def _log(self, msg):
        try: open(LOG, "a", encoding="utf-8").write(time.strftime("%F %T") + " " + msg + "\n")
        except OSError: pass

    def _load(self):
        import skilljepa as sj, avb_np, avb_feats as af
        self.sj, self.np_, self.af = sj, avb_np, af; self.W = avb_np.load_head(self.head); self.meta = self.W["__meta__"]; self.K = self.meta["K"]
        X, Y, lab, _, _, names, _ = sj.data("prevskill", "all"); self.Y, self.names = Y, names; self.ix = {n: i for i, n in enumerate(names)}
        tp, te = os.path.join(D, "pairs_trace.jsonl"), os.path.join(D, "emb_trace.npz")
        if os.path.exists(tp) and os.path.exists(te):
            rows = [json.loads(l) for l in open(tp, encoding="utf-8") if l.strip()]; X0 = np.load(te)["X"]
            if len(X0) == len(rows) and rows:
                l2 = np.array([self.ix.get(r["skills"][0], -1) for r in rows]); last, prv = {}, np.full(len(rows), -1)
                for i, r in enumerate(rows): prv[i] = last.get(r["sid"], -1); last[r["sid"]] = i
                X = np.vstack([X, sj.with_ctx(X0, Y, prv, np.where(prv >= 0, l2[np.maximum(prv, 0)], -1), "prevskill")]); lab = np.r_[lab, l2]
        self.Xp, self.labp = X, lab
        tf, ln = [], []   # BM25 over SKILL.md, same as avb_feats.lex
        for nm in names:
            try: body = open(os.path.join(SK, nm, "SKILL.md"), encoding="utf-8", errors="replace").read()
            except OSError: body = ""
            c = collections.Counter(af.tok(body)); tf.append(c); ln.append(sum(c.values()))
        N = len(names); avg = sum(ln) / N; df = collections.Counter()
        for c in tf: df.update(c.keys())
        self.tf, self.idf, self.nrm = tf, {w: math.log(1 + (N - d + 0.5) / (d + 0.5)) for w, d in df.items()}, [1.2 * (1 - 0.75 + 0.75 * l / avg) for l in ln]
        self.ok = True

    def _usage(self):
        p = os.path.join(SK, "usage.jsonl")
        try: mt = os.path.getmtime(p)
        except OSError: return self.use["rows"]
        if mt != self.use["mt"]:
            rows = []
            for l in open(p, encoding="utf-8").read().splitlines()[-4000:]:
                try: j = json.loads(l); rows.append((time.mktime(time.strptime(j["t"], "%Y-%m-%d %H:%M")), j.get("skill"), j.get("sid", "")))
                except (ValueError, KeyError): continue
            self.use.update(mt=mt, rows=rows)
        return self.use["rows"]

    def _embed(self, q):
        body = json.dumps({"input": [(self.sj.Q_PRE + q)[:1500]]}).encode()
        with urllib.request.urlopen(urllib.request.Request(EMB_URL, body, {"Content-Type": "application/json"}), timeout=3) as r: e = np.array(json.load(r)["data"][0]["embedding"], np.float32)
        return e / max(float(np.linalg.norm(e)), 1e-9)

    def logits(self, q, sid, now=None, emb=None):
        """-> (cands skill names, logits for them, none logit). Raises on any failure; col() catches."""
        now = now or time.time(); N = len(self.names); af = self.af
        x = emb if emb is not None else self._embed(q); sid = (sid or "")[:8]
        # prev skill / prior from usage.jsonl (uses must be >= 60 s old, as avb_feats.lex)
        Pr = np.zeros((1, N), np.float32); plab = -1; lt = -1.0
        if sid:
            for ut, sk, us in self._usage():
                if us == sid and sk in self.ix and ut + 60 <= now < ut + 6 * 3600:
                    Pr[0, self.ix[sk]] += 0.5 ** ((now - ut) / 1800.0)
                    if ut > lt: lt, plab = ut, self.ix[sk]
        pv = self.last.get(sid) if sid else None
        z = x + (0.5 * pv if pv is not None else 0) + (0.5 * self.Y[plab] if plab >= 0 else 0); xq = (z / np.linalg.norm(z))[None].astype(np.float32)
        if sid: self.last[sid] = x
        B = np.zeros((1, N), np.float32)
        for w in set(af.tok(q)):
            if w in self.idf:
                for j in range(N):
                    f = self.tf[j].get(w)
                    if f: B[0, j] += self.idf[w] * f * 2.2 / (f + self.nrm[j])
        E = self.sj.knn_energies(self.Xp, self.labp, xq, N); cs = af.cands(E, B, self.K, True)
        T = af.feats(xq, self.Y, E, B, Pr, cs, np.array([plab]), [{plab} if plab >= 0 else set()], True); Z = self.np_.forward(self.W, T)[0]
        return [self.names[j] for j in cs[0]], Z[:-1], float(Z[-1])

    def col(self, q, sid, skills):
        """route.log col 8; '' on any problem (head file missing, 8084 down, ...): never raises."""
        try:
            with self.lock:
                now = time.time()
                if not self.ok:
                    if now < self.next_try: return ""
                    try: self._load()
                    except Exception as e:
                        self.next_try = now + RETRY; self._log(f"avb disabled, head/pool not loadable ({type(e).__name__}: {e}); col 8 empty, retry in {int(RETRY)}s"); return ""
                try: c, z, zn = self.logits(q, sid, now)
                except Exception as e:
                    if now - self.warned > RETRY: self.warned = now; self._log(f"avb scoring failed ({type(e).__name__}: {e}); col 8 empty")
                    return ""
                d = dict(zip(c, z))
                return "avb=" + ",".join([f"{s}:{d[s]:.2f}" if s in d else f"{s}:nan" for s in skills] + [f"_none:{zn:.2f}"])
        except Exception: return ""
