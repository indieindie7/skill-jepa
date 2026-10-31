"""a17_svts.py: A17 (SVTS tie-in, video-notes tree/369) auto-gold eval proxy. notes/context-nudger-PLAN.md line 36, notes/a17-svts-20261031.md.
  py -3.13 a17_svts.py gen [--n 300]      skill graph -> local-fact lines -> prompts (deterministic template rewrite, see notes) -> data/a17-prompts.json
  C:/llm/qlora/venv/Scripts/python.exe a17_svts.py eval    >= 6 router variants on auto-gold (overall / per hop) + frozen holdout hit1, Spearman -> data/a17-results.json
CPU only (CUDA hidden). Reads data/*, ~/.claude/skills; writes only data/a17-*.json."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import collections, glob, hashlib, json, random, re, sys
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "data"); sys.path.insert(0, HERE)
SK = os.path.join(os.path.expanduser("~"), ".claude", "skills")
F = lambda n: os.path.join(D, n)
TOK = re.compile(r"[a-z0-9][a-z0-9_.\-+]*[a-z0-9]|[a-z0-9]")
STOP = set("the a an and or of to in on for with is are was were be been it this that these those as at by from not no do does did can could should would will may must never always only also than then when what which who how why you your we our they their its if but so too very just into onto over under about via per vs etc any all each both more most some such other own same new old use used using one two see per".split())
FACT = [re.compile(p, re.I) for p in (r"[A-Za-z]:[\\/]", r"~/|\bDocuments/|\.claude/", r"\b[\w\-]+\.(py|md|json|jsonl|exe|ps1|rs|toml|npz|bat|ini|txt|dll|cfg|yaml|yml|js|java|c|cpp)\b", r"\b(127\.0\.0\.1|localhost)\b|:\d{4,5}\b|\bport\s+\d{4}", r"`[^`]*\s[^`]*`", r"\b(?=[0-9a-f]*\d)[0-9a-f]{7,40}\b", r"\bgotcha\b|\bnever\b|\bdo not\b|\bdon't\b|\bmust\b")]


def toks(s): return TOK.findall(s.lower())


def files():
    """-> {skill: [(relpath, kind 'skill'|'tree', [lines])]}"""
    out = {}
    for d in sorted(os.listdir(SK)):
        p = os.path.join(SK, d)
        if not os.path.isdir(p) or not os.path.exists(os.path.join(p, "SKILL.md")): continue
        fs = [(f"{d}/SKILL.md", "skill")] + [(f"{d}/tree/{os.path.basename(t)}", "tree") for t in sorted(glob.glob(os.path.join(p, "tree", "*.md")))]
        out[d] = [(rel, k, open(os.path.join(SK, rel), encoding="utf-8", errors="replace").read().splitlines()) for rel, k in fs]
    return out


def cmd_gen(n=300, seed=17):
    G = files(); names = sorted(json.load(open(F("skills.json"), encoding="utf-8"))); ns = set(names)
    df = collections.defaultdict(set)   # token -> skills whose files contain it
    for sk, fl in G.items():
        for _, _, ls in fl:
            for l in ls: df.update({}) or [df[t].add(sk) for t in toks(l)]
    uniq = lambda t, sk: df[t] == {sk}
    # graph edge A->B when A's files cite B by [[B]] or `B` / skills/B/
    linked_in = collections.Counter()
    for a, fl in G.items():
        txt = "\n".join("\n".join(ls) for _, _, ls in fl)
        for b in G:
            if b != a and (f"[[{b}]]" in txt or f"`{b}`" in txt or f"skills/{b}/" in txt or f"skill {b}" in txt): linked_in[b] += 1
    cand = collections.defaultdict(list)
    for sk, fl in G.items():
        if sk not in ns: continue
        for rel, kind, ls in fl:
            for ln, l in enumerate(ls, 1):
                s = l.strip()
                if not 50 <= len(s) <= 320 or s.startswith(("Reliability:", "<!--", "---", "#", "|")) or "tree/" in s[:12] or "->" in s[:4]: continue
                if not any(p.search(s) for p in FACT): continue
                ts = toks(s); anch = [t for t in dict.fromkeys(ts) if len(t) >= 4 and uniq(t, sk) and (re.search(r"[\d_./\-]", t) or t != t.lower() or True)]
                idl = [t for t in anch if re.search(r"[\d_.\-]", t)] or anch
                if not idl: continue
                # rewrite: keep only words shared with >= 2 skills (no unique token), no owner-name parts, content words only
                nm = set(re.split(r"[-_]", sk)) | {sk}
                w = [t for t in dict.fromkeys(re.findall(r"[a-z]{3,}", s.lower())) if t not in STOP and not uniq(t, sk) and t not in nm and len(df[t]) >= 2]
                if len(w) < 4: continue
                cand[sk].append({"gold": sk, "hop": 1 if kind == "skill" else 2, "src": f"{rel}:{ln}", "line": s, "anchor": idl[0], "words": w[:10], "linked_in": linked_in[sk] > 0})
    rng = random.Random(seed); pool = {1: [], 2: []}
    for sk in sorted(cand):
        for h in (1, 2):
            c = [x for x in cand[sk] if x["hop"] == h]; rng.shuffle(c); pool[h] += c[:5]   # cap 5 lines per skill per hop
    for h in pool: rng.shuffle(pool[h])
    T = ["what was the thing about {w}", "remind me, {w}", "how do i handle {w} again", "quick q on {w}, what was the rule", "{w} gotcha?", "ok so {w}, where is that noted", "hmm {w}", "need the exact setup for {w}"]
    rows, seen = [], set()
    for h, take in ((1, n // 2), (2, n - n // 2)):
        for x in pool[h]:
            if sum(r["hop"] == h for r in rows) >= take: break
            tpl = T[int(hashlib.md5(x["line"].encode()).hexdigest()[:4], 16) % len(T)]; q = tpl.format(w=" ".join(x["words"][:int(6 + int(hashlib.md5(x["src"].encode()).hexdigest()[:2], 16) % 5)]))
            qt = set(toks(q)); sk = x["gold"]
            if any(uniq(t, sk) for t in qt) or qt & (set(re.split(r"[-_]", sk)) | {sk}) or x["anchor"] in qt or q in seen: continue   # leak filter
            seen.add(q); rows.append(dict(x, q=q, id=len(rows)))
    json.dump({"n": len(rows), "by_hop": dict(collections.Counter(r["hop"] for r in rows)), "n_skills": len({r["gold"] for r in rows}), "n_candidates": {h: sum(1 for sk in cand for x in cand[sk] if x["hop"] == h) for h in (1, 2)},
               "rewrite": "deterministic template + leak filter (gpu 1 card not granted: water-gym held it)", "rows": rows}, open(F("a17-prompts.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("prompts", len(rows), dict(collections.Counter(r["hop"] for r in rows)), "skills", len({r['gold'] for r in rows}), "cands", {h: sum(1 for sk in cand for x in cand[sk] if x["hop"] == h) for h in (1, 2)})


def rank(v):
    import numpy as np
    v = np.asarray(v, float); o = np.argsort(v); r = np.empty(len(v)); i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[o[j + 1]] == v[o[i]]: j += 1
        r[o[i:j + 1]] = (i + j) / 2; i = j + 1
    return r


def spearman(a, b):
    import numpy as np
    ra, rb = rank(a), rank(b)
    return float(np.corrcoef(ra, rb)[0, 1]) if ra.std() and rb.std() else float("nan")


def cmd_eval():
    import numpy as np, json as js
    import skilljepa as sj, avb_feats as af, avb_np, holdout_check as hc, arms
    P = js.load(open(F("a17-prompts.json"), encoding="utf-8"))["rows"] if False else json.load(open(F("a17-prompts.json"), encoding="utf-8"))["rows"]
    sj_, X, Y, lab, allp, sid, names, plab, te = hc._base(); ix = {n: i for i, n in enumerate(names)}; P = [r for r in P if r["gold"] in ix]; n = len(P)
    gold = np.array([ix[r["gold"]] for r in P]); hop = np.array([r["hop"] for r in P]); rows = [{"sid": f"a17{i:05d}", "t": "2026-10-31T12:00:00.000Z", "q": r["q"]} for i, r in enumerate(P)]
    import time as _t; _t0 = _t.time(); lg = lambda m: print(m, round(_t.time() - _t0), flush=True)
    Xq = sj.embed([sj.Q_PRE + r["q"] for r in P]); Xq = sj.with_ctx(Xq, Y, np.full(n, -1), np.full(n, -1), "prevskill"); B, Pr = af.lex(rows, names)
    lg("lex done")
    E = sj.knn_energies(X[~te], lab[~te], Xq, len(Y)); pairs = [json.loads(l) for l in open(F("pairs.jsonl"), encoding="utf-8")]; hold_i = np.where(te & (lab >= 0))[0]; lg("knn done")
    V = {}   # name -> dict
    def rec(nm, top1, hold, src):
        ok = np.array([bool(t == g) for t, g in zip(top1, gold)])
        V[nm] = {"auto_hit1": float(ok.mean()), "hop1": float(ok[hop == 1].mean()), "hop2": float(ok[hop == 2].mean()), "holdout_hit1": float(hold), "src": src, "n": n, "n_hop1": int((hop == 1).sum()), "n_hop2": int((hop == 2).sum())}
        print(nm, {k: round(v, 3) for k, v in V[nm].items() if isinstance(v, float)}, flush=True)
    for nm, path in (("prod_d256b2", "avb_head.npz"), ("d16b0", "avb_head_d16b0.npz"), ("d16b0_gate", "a13b-gate-d16b0.npz"), ("distill", "avb_head_distill.npz"), ("distill_stub", "avb_head_distill_stub.npz")) + tuple((f"{a}_s{s}", os.path.join(os.environ.get("TEMP", "/tmp"), f"a16_{a}_{s}.npz")) for a in ("R0", "R1", "R2") for s in range(3)):
        p = path if os.path.isabs(path) else F(path)
        if not os.path.exists(p): print("missing", p); continue
        W = avb_np.load_head(p); K = W["__meta__"]["K"]; cs = af.cands(E, B, K, True); T = af.feats(Xq, Y, E, B, Pr, cs, np.full(n, -1), [set() for _ in range(n)], True); Z = avb_np.forward(W, T)
        top = [cs[j][int(np.argmax(Z[j, :-1]))] for j in range(n)]; rec(nm, top, hc.score(p)["hit1"], os.path.basename(p))
    for fz in (True, False):
        a0 = arms.A0(frozen=fz); top = []
        for r in rows:
            s = a0.score(r); top.append(ix.get(s[0][0], -1) if s else -1)
        hh = [bool((lambda s: s and ix.get(s[0][0]) in allp[i])(a0.score(pairs[i]))) for i in hold_i]; rec("A0_frozen" if fz else "A0_live", top, float(np.mean(hh)), "arms.A0 frozen" if fz else "arms.A0 live")
    ks = list(V); res = {"variants": V, "n_variants": len(ks)}
    for f in ("auto_hit1", "hop1", "hop2"): res[f"rho_{f}_vs_holdout"] = spearman([V[k][f] for k in ks], [V[k]["holdout_hit1"] for k in ks])
    best = max(ks, key=lambda k: V[k]["holdout_hit1"]); bauto = max(ks, key=lambda k: V[k]["auto_hit1"])
    res.update(best_holdout_variant=best, best_auto_variant=bauto, rho=res["rho_auto_hit1_vs_holdout"], best_hop1_gt_hop2=V[bauto]["hop1"] > V[bauto]["hop2"], best_holdout_hop1_gt_hop2=V[best]["hop1"] > V[best]["hop2"],
               pass_rule=(res["rho_auto_hit1_vs_holdout"] >= .7) and V[bauto]["hop1"] > V[bauto]["hop2"])
    json.dump(res, open(F("a17-results.json"), "w", encoding="utf-8"), indent=1); print({k: v for k, v in res.items() if k != "variants"})


if __name__ == "__main__":
    c = sys.argv[1] if len(sys.argv) > 1 else ""
    if c == "gen": cmd_gen(int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 300)
    elif c == "eval": cmd_eval()
    else: print(__doc__)
