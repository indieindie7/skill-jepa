"""struct_arms.py: T1 branch routing, T2 external-state prior, T3 short-note lines vs A0 on the frozen holdout (data/holdout.jsonl, hash-verified).
  C:/llm/qlora/venv/Scripts/python.exe struct_arms.py   -> data/struct-arms-20261029.json
A0 = arms.A0 (read-only import). Metrics: hit1 on labelled rows, mean tokens (chars/4). Tuning on VAL only (pairs positions [0.49,0.70) per sid, same as arms.py time split).
T1 GT = transcript tool_use (Read, or Bash/PowerShell cat/head/sed/Get-Content) of ~/.claude/skills/<skill>/tree/<file> in the same sid within WIN s after the query and before the sid next pair. usage.jsonl has no file field (sid,skill,t only)."""
import datetime, glob, json, math, os, re, sys, collections
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "data"); sys.path.insert(0, HERE)
import numpy as np
import holdout_check as hc
import arms
from arms import A0, opsd
SK = os.path.expanduser("~/.claude/skills"); WIN = 1200
ok, msg = hc.verify(); print("holdout", msg); assert ok
ts = lambda s: datetime.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
pairs = [json.loads(l) for l in open(os.path.join(D, "pairs.jsonl"), encoding="utf-8")]
hold = [json.loads(l) for l in open(os.path.join(D, "holdout.jsonl"), encoding="utf-8")]
cnt = collections.Counter(p["sid"] for p in pairs); pos = {}; seen = collections.Counter()
for i, p in enumerate(pairs): pos[i] = seen[p["sid"]] / cnt[p["sid"]]; seen[p["sid"]] += 1
val_idx = [i for i in range(len(pairs)) if 0.49 <= pos[i] < 0.70]; trn_idx = [i for i in range(len(pairs)) if pos[i] < 0.49]
test_idx = [r["i"] for r in hold]
gold = lambda i: set(hold_sk.get(i) or pairs[i]["skills"]); isnone = lambda i: gold(i) <= {"none"}
hold_sk = {r["i"]: r["skills"] for r in hold}
a0 = A0(); print("A0 init", flush=True)
sc0 = {i: a0.score(pairs[i]) for i in set(val_idx) | set(test_idx)}; print("A0 scored", len(sc0), flush=True)
def h1(rank, i): return bool(rank and rank[0] in gold(i))
def tokb(nm): return a0.brief_tok.get(nm, 0.0)
res = {"n_test": len(test_idx), "n_test_labelled": sum(not isnone(i) for i in test_idx)}
Lt = [i for i in test_idx if not isnone(i)]; Lv = [i for i in val_idx if not isnone(i)]
res["A0"] = {"hit1": float(np.mean([h1([k for k, _ in sc0[i]], i) for i in Lt])), "tokens_top1_brief": float(np.mean([tokb(sc0[i][0][0]) if sc0[i] else 0 for i in test_idx])), "tokens_top3_brief": float(np.mean([sum(tokb(k) for k, _ in sc0[i][:3]) for i in test_idx]))}
print("A0", res["A0"], flush=True)
# ---------------- T2: external state prior
trans = collections.defaultdict(collections.Counter); last = {}
for i in trn_idx:                                    # skill->next-skill transitions from TRAIN rows only
    p = pairs[i]; g = [s for s in p["skills"] if s != "none"]
    if p["sid"] in last:
        for s in g:
            for ps in last[p["sid"]]: trans[ps][s] += 1
    if g: last[p["sid"]] = g
def hist(i, k=3):
    t = ts(pairs[i]["t"]); u = sorted((ut, s) for ut, s in a0.use.get(pairs[i]["sid"], ()) if ut + 60 <= t < ut + 6 * 3600)
    out = []
    for ut, s in reversed(u):
        if s not in out: out.append(s)
        if len(out) == k: break
    return out
def t2_rank(i, w_h, w_t, w_p):
    d = dict(sc0[i]); H = hist(i)
    for r, s in enumerate(H):
        if s in a0.skills: d[s] = d.get(s, 0) + w_h * 0.5 ** r / (sum(0.5 ** q for q in range(len(H))) or 1)
    if H and H[0] in a0.skills: d[H[0]] = d.get(H[0], 0) + w_p
    if H and trans.get(H[0]):
        tot = sum(trans[H[0]].values())
        for s, c in trans[H[0]].items():
            if s in a0.skills: d[s] = d.get(s, 0) + w_t * c / tot
    return [k for k, v in sorted(d.items(), key=lambda kv: -kv[1])]
grid = [(a, b, c) for a in (0, .25, .5, 1) for b in (0, .25, .5, 1) for c in (0, .25, .5)]
vs = {g: float(np.mean([h1(t2_rank(i, *g), i) for i in Lv])) for g in grid}; bg = max(grid, key=lambda g: (vs[g], -sum(g)))
res["T2"] = {"feasible": True, "best_w(hist3,trans,prev)": bg, "val_hit1": vs[bg], "val_hit1_A0": vs[(0, 0, 0)], "test_hit1": float(np.mean([h1(t2_rank(i, *bg), i) for i in Lt])), "tokens_delta_vs_A0": 0.0,
             "note": "same pool/briefs as A0 so tokens unchanged; A0 already contains a decayed per-sid prior (mass 1); T2 adds last-3 history + train transition prior"}
print("T2", res["T2"], flush=True)
# ---------------- T1 ground truth from transcripts
R = re.compile(r"skills[/\\]+([a-z0-9-]+)[/\\]+tree[/\\]+([0-9][^\s\"'\\/]*\.md)"); RD = re.compile(r"\b(cat|head|tail|sed|Get-Content|type|less|more)\b"); WR = re.compile(r">>?\s|Out-File|Set-Content|Add-Content|tee\b|<<")
sids = {p["sid"] for p in pairs}; reads = collections.defaultdict(list)
for f in glob.glob(os.path.expanduser("~/.claude/projects/*/*.jsonl")):
    sid = os.path.basename(f)[:8]
    if sid not in sids: continue
    for l in open(f, encoding="utf-8", errors="replace"):
        if "tree" not in l or '"assistant"' not in l[:200]: continue
        try: j = json.loads(l)
        except ValueError: continue
        c0 = j.get("message", {}).get("content")
        if j.get("type") != "assistant" or not isinstance(c0, list): continue
        for c in c0:
            if c.get("type") != "tool_use": continue
            inp = c["input"]; s = json.dumps(inp)
            if c["name"] == "Read": pass
            elif c["name"] in ("Bash", "PowerShell"):
                cmd = inp.get("command", "")
                if not RD.search(cmd) or WR.search(cmd.replace("2>&1", "")): continue
            else: continue
            for m in R.finditer(s): reads[sid].append((ts(j["timestamp"]), m.group(1), m.group(2)))
rows_by_sid = collections.defaultdict(list)
for i, p in enumerate(pairs): rows_by_sid[p["sid"]].append(ts(p["t"]))
def gt(i):
    p = pairs[i]; t = ts(p["t"]); later = [x for x in rows_by_sid[p["sid"]] if x > t]; end = min([t + WIN] + later[:1]); g = gold(i)
    return [(sk, fn) for rt, sk, fn in sorted(reads.get(p["sid"], ())) if t <= rt < end and sk in g]
skills = a0.skills
def branches(nm):
    d = os.path.join(SK, nm, "tree"); out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.md"))):
        try: out[os.path.basename(f)] = open(f, encoding="utf-8", errors="replace").read()
        except OSError: pass
    return out
BR = {nm: branches(nm) for nm in skills}
def bm(qt, texts):
    toks = {k: collections.Counter(opsd._tok(v)) for k, v in texts.items()}; N = len(toks); ln = {k: sum(c.values()) for k, c in toks.items()}; avg = sum(ln.values()) / N or 1; df = collections.Counter()
    for c in toks.values(): df.update(c.keys())
    sc = {}
    for k, c in toks.items():
        r = 0.0
        for w in set(qt):
            f = c.get(w)
            if f: r += math.log(1 + (N - df[w] + 0.5) / (df[w] + 0.5)) * f * 1.5 / (f + 1.2 * (0.25 + 0.75 * ln[k] / avg))
        sc[k] = r
    return sorted(sc, key=lambda k: (-sc[k], k))
def prod_pick(nm, q):   # brief() branch ordering: title/file overlap with q, then highest number
    tree = skills[nm].get("tree") or []
    qw = set(opsd._tok(q)); return [os.path.basename(b["file"]) for b in sorted(tree, key=lambda b: (-len(qw & set(opsd._tok(b["title"] + " " + b["file"]))), -opsd._branch_no(b["file"])))]
rows = []
for i in Lt:
    g = gt(i)
    if g: rows.append((i, g))
print("T1 rows with GT", len(rows), "of", len(Lt), flush=True)
def evalT1(oracle):
    hb, hp, hn, h8, hb3, tb, tf, tn, ncand = [], [], [], [], [], [], [], [], []
    for i, g in rows:
        pk = sc0[i][0][0] if sc0[i] else None
        nm = None
        for sk, fn in g:
            if oracle or sk == pk: nm = sk; break
        if nm is None or not BR.get(nm): continue
        want = {fn for sk, fn in g if sk == nm}; q = pairs[i]["q"]; cand = BR[nm]
        r = bm(opsd._tok(q), cand); rp = prod_pick(nm, q); rn = sorted(cand, key=lambda f: -opsd._branch_no(f))
        hb.append(r[0] in want); hp.append(rp[0] in want if rp else False); h8.append(bool(set(rp[:opsd.BRIEF_BRANCHES]) & want)); hb3.append(bool(set(r[:3]) & want)); hn.append(rn[0] in want); ncand.append(len(cand))
        tb.append(tokb(nm) + len(cand[r[0]]) / 4); tf.append(skills[nm]["tokens"]["skill"] + 0.0); tn.append(tokb(nm))
    m = lambda x: float(np.mean(x)) if x else None
    return {"n": len(hb), "branch_hit1_bm25": m(hb), "branch_hit1_prod_brief_order": m(hp), "branch_listed_in_prod_brief8": m(h8), "branch_hit3_bm25": m(hb3), "branch_hit1_newest": m(hn), "mean_candidates": m(ncand), "tokens_brief_plus_branch": m(tb), "tokens_full_skillmd": m(tf), "tokens_brief_only": m(tn)}
res["T1"] = {"feasible": True, "gt_rows": len(rows), "oracle_skill": evalT1(True), "after_A0_top1_correct": evalT1(False), "gt_source": "transcript Read/cat of tree files within 20 min of query, same sid, skill in gold set (usage.jsonl has no file field)"}
print("T1", json.dumps(res["T1"]), flush=True)
# ---------------- T3 short note: top-3 lines
tl, tb0, hit = [], [], []
for i in test_idx:
    if not sc0[i]: continue
    nm = sc0[i][0][0]; q = pairs[i]["q"]
    try: lt = opsd.lines_text(nm, q, 3)
    except Exception: lt = ""
    tl.append(len(lt) / 4); tb0.append(tokb(nm))
    if lt and not isnone(i) and nm in gold(i):
        want = {fn for sk, fn in gt(i) if sk == nm}
        if want:
            refs = [s_[1] for s_ in opsd.skill_lines(nm, q, 3)]; hit.append(any(any(w in r_ for w in want) for r_ in refs))
res["T3"] = {"feasible": True, "tokens_top3_lines": float(np.mean(tl)), "tokens_brief_top1_A0": float(np.mean(tb0)), "n_routed": len(tl), "proxy_ref_hits_opened_branch": (float(np.mean(hit)) if hit else None), "proxy_n": len(hit),
             "note": "opsd.brief(q=...) already embeds lines_text (k=4) in production; T3 = serve ONLY top-3 lines instead of brief. Proxy = a served line ref is in the tree file the chat later opened (only rows with tree-open GT); no proxy for SKILL.md-only answers"}
print("T3", res["T3"], flush=True)
json.dump(res, open(os.path.join(D, "struct-arms-20261029.json"), "w"), indent=1)
