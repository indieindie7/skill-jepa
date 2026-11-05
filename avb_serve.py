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
GATE_OFF = 60.0  # 11-05 s the live gate stays off after a scoring failure (dead 8084 = one 3 s timeout per minute, not per route)


FU = re.compile(r"^\W*(ok|okay|yes|yep|sure|go|go on|continue|proceed|do it|do them all|please do them all|all|next|thanks|thx|y)\W*$", re.I)  # = a14_abstain.FU


def gate_feats(q, z, zn):
    """10-31 route.log col 9 'gf=' (shadow, A14 LR abstain gate inputs; SELFIMPROVE_PLAN sec 9): head-side + query-shape features
    as a14_abstain.feats (htop, hmarg, hent over clipped cand logits, znone, log1p chars/words, url, '?', symbol frac, follow-up).
    A0/BM25/c1/qemb/prev features are recomputed offline from q + sid; the gate is fit offline on these logged rows. Never changes routing."""
    try:
        zc = np.clip(np.asarray(z, np.float64), -3.0, 3.0); zs = np.sort(zc)[::-1]; ex = np.exp(zc - zc.max()); pr = ex / ex.sum()
        f = [zs[0], zs[0] - (zs[1] if len(zs) > 1 else 0.0), -(pr * np.log(pr + 1e-9)).sum(), zn, np.log1p(len(q)), np.log1p(len(q.split())),
             float("http" in q or "youtu" in q), float("?" in q), sum(q.count(ch) for ch in "{}[]();=<>_\\/") / max(len(q), 1), float(bool(FU.match(q)) or len(q.split()) <= 3)]
        return "gf=" + ",".join(f"{v:.3f}" for v in f)
    except Exception: return "gf="


class AvbServe:
    def __init__(self, head=None):
        self.head = head or os.path.join(D, "avb_head.npz"); self.ok = False; self.next_try = 0.0; self.lock = threading.Lock(); self.last = {}; self.use = {"mt": -1, "rows": []}; self.warned = 0.0; self.gate_retry = 0.0

    def _log(self, msg):
        try: open(LOG, "a", encoding="utf-8").write(time.strftime("%F %T") + " " + msg + "\n")
        except OSError: pass

    def _load(self):
        import skilljepa as sj, avb_np, avb_feats as af
        self.sj, self.np_, self.af = sj, avb_np, af; self.W = avb_np.load_head(self.head); self.meta = self.W["__meta__"]; self.K = self.meta["K"]
        X, Y, lab, _, _, names, _ = sj.data("prevskill", "all"); self.Y, self.names = Y, names; self.ix = {n: i for i, n in enumerate(names)}
        from a6c_nudger import train_stats; from a17_user_profile import table_probs
        self.tprob = table_probs; self.pg, self.Tg = train_stats(lab, sj.data("prevskill", "all")[6], np.arange(len(lab)), len(names))   # 11-04 A6c shadow col 10
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

    def nud(self, c, z, sid, now, w=0.3, lag=600):
        """11-04 route.log col 10 'nud=<skill>' (shadow only, A6c rank-only nudge, holdout pass SELFIMPROVE_PLAN 11-04): argmax(head logits + w*log p_table(class | this session's skill uses >= lag s old)).
        Live history = usage.jsonl uses only (holdout used labelled turns incl. none): approximation, the shadow day measures it."""
        try:
            N = len(self.names); sid = (sid or "")[:8]; ts = np.zeros((N + 1, N + 1)); cs = np.zeros(N + 1); prev = N
            for ut, sk, us in sorted(self._usage(), key=lambda r: r[0]) if sid else []:
                if us == sid and sk in self.ix and ut + lag <= now: k = self.ix[sk]; ts[prev, k] += 1; cs[k] += 1; prev = k
            pt = self.tprob(prev, ts, cs, self.Tg, self.pg); lp = np.log(np.array([pt[self.ix[n]] for n in c]) + 1e-9)
            return "nud=" + c[int(np.argmax(np.asarray(z) + w * lp))]
        except Exception: return "nud="

    def _ready(self, now):
        """load on first use; False while disabled (retry every RETRY s). Caller holds self.lock."""
        if self.ok: return True
        if now < self.next_try: return False
        try: self._load(); return True
        except Exception as e:
            self.next_try = now + RETRY; self._log(f"avb disabled, head/pool not loadable ({type(e).__name__}: {e}); col 8 empty, retry in {int(RETRY)}s"); return False

    def gate(self, q, sid):
        """11-05 LIVE abstain gate for opsd.route() (SELFIMPROVE_PLAN 11-05 X6, frozen holdout: acc_with_abstain .253 -> .728, false-route-on-none .896 -> .055; ranking untouched).
        -> (abstain, (c, z, zn)) with abstain = P(none) >= head meta thr (0.4485, holdout-tuned), or None = head/8084 unavailable (caller keeps its NOMATCH gate).
        Sync (~130 ms embed); after a scoring failure the gate stays off GATE_OFF s."""
        try:
            with self.lock:
                now = time.time()
                if now < self.gate_retry or not self._ready(now): return None
                try: c, z, zn = self.logits(q, sid, now)
                except Exception as e:
                    self.gate_retry = now + GATE_OFF
                    if now - self.warned > RETRY: self.warned = now; self._log(f"avb scoring failed ({type(e).__name__}: {e}); gate off {int(GATE_OFF)}s")
                    return None
            Z = np.r_[np.asarray(z, np.float64), zn][None]
            return bool(self.af.pnone(Z)[0] >= self.meta["thr"]), (c, z, zn)
        except Exception: return None

    def col(self, q, sid, skills, pre=None):
        """route.log col 8; '' on any problem (head file missing, 8084 down, ...): never raises. pre = (c, z, zn) from gate() reuses its logits (one embed per route)."""
        try:
            with self.lock:
                now = time.time()
                if not self._ready(now): return ""
                if pre is not None: c, z, zn = pre
                else:
                    try: c, z, zn = self.logits(q, sid, now)
                    except Exception as e:
                        if now - self.warned > RETRY: self.warned = now; self._log(f"avb scoring failed ({type(e).__name__}: {e}); col 8 empty")
                        return ""
                d = dict(zip(c, z))
                return "avb=" + ",".join([f"{s}:{d[s]:.2f}" if s in d else f"{s}:nan" for s in skills] + [f"_none:{zn:.2f}"]) + chr(9) + gate_feats(q, z, zn) + chr(9) + self.nud(c, z, sid, now)
        except Exception: return ""
