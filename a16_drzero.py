"""a16_drzero.py: A16 (DR-Zero tie-in) failure-mined proposer prompts vs unfiltered synthetic prompts as extra head-train rows. notes/context-nudger-PLAN.md line 35.
  C:/llm/qlora/venv/Scripts/python.exe a16_drzero.py gen      proposer (llama-server gpu1, --url) -> data/a16-seeds.json, data/a16-poolA.jsonl, then R0 router + R1/R2 files
  C:/llm/qlora/venv/Scripts/python.exe a16_drzero.py eval     3 arms x 3 seeds x (time, session), holdout gate -> data/a16-results.json
CPU torch only (CUDA hidden). Synthetic rows join the head TRAIN pool only (val/test knn pools unchanged, as avb_rerank --extra); seeds from rows in BOTH training splits (time pos<.7 and session non-test)."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(_v, "6")
import argparse, collections, hashlib, json, random, re, subprocess, sys, tempfile, time, urllib.request
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "data"); sys.path.insert(0, HERE)
import skilljepa as sj, avb_feats as af, avb_rerank as ar, arms
import torch
torch.set_num_threads(6)
URL = os.environ.get("A16_URL", "http://127.0.0.1:8095/v1/chat/completions")
nz = lambda t: " ".join(t.lower().split())
F = lambda n: os.path.join(D, n)


def base():
    X, Y, lab, allp, sid, names, plab = sj.data("prevskill", "all"); n = len(lab)
    pairs = [json.loads(l) for l in open(F("pairs.jsonl"), encoding="utf-8")]
    pos, cnt = np.zeros(n), {}
    for i, c in enumerate(sid): pos[i] = cnt.get(c, 0); cnt[c] = pos[i] + 1
    tot = np.array([cnt[c] for c in sid]); h = lambda s, salt: int(hashlib.md5((salt + s).encode()).hexdigest()[:8], 16) % 100
    te_s = np.array([h(s, "test") < 30 for s in sid]); va_s = np.array([(not t) and h(s, "val") < 20 for s, t in zip(sid, te_s)]); tr_s = ~te_s & ~va_s
    tr_t, sub = pos < 0.7 * tot, pos < 0.49 * tot
    return dict(X=X, Y=Y, lab=lab, allp=allp, sid=sid, names=names, plab=plab, pairs=pairs, pos=pos, tot=tot, te_s=te_s, tr_s=tr_s, tr_t=tr_t, sub=sub)


def chat(prompt, temp=1.0, max_tokens=1200):
    body = json.dumps({"messages": [{"role": "user", "content": prompt}], "temperature": temp, "top_p": 0.95, "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r: return json.load(r)["choices"][0]["message"]["content"]


def brief(nm):
    try: return subprocess.run(["curl.exe", "-s", f"http://127.0.0.1:8765/api/skill/{nm}"], capture_output=True, text=True, timeout=30).stdout[:1500]
    except Exception: return ""


def propose(nm, br, ex, hard=(), k=8, temp=1.0):
    p = (f"You write realistic messages that a developer types to a coding assistant in a chat.\nTarget skill: {nm}\nSkill brief:\n{br}\n\nTwo real messages from this user that needed this skill:\n"
         + "\n".join(f"- {e}" for e in ex)
         + ("\n\nThese earlier attempts were NOT routed to the skill by the router (too ambiguous or off-vocabulary); write messages with the same intent but harder, e.g. less keyword overlap, terse, mid-conversation follow-ups, other phrasing:\n" + "\n".join(f"- {h}" for h in hard) if hard else "")
         + f"\n\nWrite {k} NEW messages in this user's style (lowercase, terse, sometimes pasted paths or errors, varied length 3-40 words) that a person would send when they need this skill. Do not mention the skill name. Output only a JSON array of {k} strings.")
    out = chat(p, temp); m = re.search(r"\[.*\]", out, re.S)
    try: return [s.strip() for s in json.loads(m.group(0)) if isinstance(s, str)]
    except Exception: return []


def mk_rows(qs):
    return [{"sid": f"syn{i:05d}", "t": "2026-10-30T12:00:00.000Z", "q": q} for i, q in enumerate(qs)]


def syn_part(b, qs, E_pool_X, E_pool_lab, targets, names, rng):
    """synthetic prompts -> head-train part dict (T, y) vs a knn pool of real train rows; also returns embeddings."""
    X0 = sj.embed([sj.Q_PRE + q for q in qs]); n = len(qs); Y = b["Y"]; rows = mk_rows(qs)
    prv = np.full(n, -1); plab = np.full(n, -1); Xs = sj.with_ctx(X0, Y, prv, plab, "prevskill"); B, Pr = af.lex(rows, names)
    E = sj.knn_energies(E_pool_X, E_pool_lab, Xs, len(Y)); cs = af.cands(E, B, 16, True); [rng.shuffle(c) for c in cs]
    lab = np.array(targets); allp = [{int(l)} for l in lab]; y, _ = af.target(lab, allp, cs, False)
    T = af.feats(Xs, Y, E, B, Pr, cs, plab, [set() for _ in range(n)], True); return {"T": T, "y": y, "cands": cs, "E": E, "B": B, "Pr": Pr, "Xs": Xs, "rows": rows}


def rank_with(model, delta, a0, rows, part, names, tops=arms.POOL):
    """B1 ranking of synthetic rows (A0 pool + delta*clip(logit))."""
    pool = [a0.score(r)[:tops] for r in rows]
    with torch.no_grad(): Z = model(torch.tensor(part["T"])).numpy()[:, :-1]
    rk, _, _ = arms.combine(pool, part["cands"], Z, names, delta); return rk


def train_split(b, sp, S, extra_T=None, extra_y=None, seed=0):
    parts = dict(S[sp]["parts"]); tr = dict(parts["train"])
    if extra_T is not None and len(extra_T): tr["T"] = np.concatenate([tr["T"], extra_T]); tr["y"] = np.concatenate([tr["y"], extra_y])
    parts["train"] = tr; ms, r = ar.run(seed, parts, True, refit=False); return ms[0], r, parts


def pick_delta(S_sp, parts, m, names, tokf):
    with torch.no_grad(): Zv, Zt = (m(torch.tensor(parts[k]["T"])).numpy()[:, :-1] for k in ("val", "test"))
    per, rks = {}, {}
    for dl in arms.DELTAS:
        rkv, tpv, _ = arms.combine(S_sp["pv"], parts["val"]["cands"], Zv, names, dl); rkt, tpt, _ = arms.combine(S_sp["pt"], parts["test"]["cands"], Zt, names, dl)
        per[dl] = arms.fit_arm(S_sp["R_of"](S_sp["vi"], rkv, tpv), S_sp["R_of"](S_sp["ei"], rkt, tpt), tokf); rks[dl] = rkt
    best = max(arms.DELTAS, key=lambda dl: (per[dl]["val"]["hit1"] + per[dl]["val"]["acc_with_abstain"], -dl)); return best, per[best], rks[best]


def cmd_gen(a):
    b = base(); names, lab, pairs, n = b["names"], b["lab"], b["pairs"], len(b["lab"]); t0 = time.time()
    el = b["tr_t"] & ~b["te_s"] & (lab >= 0) & np.array([len(p["q"]) <= 350 and not p["q"].lstrip().startswith("<") for p in pairs])   # training rows of BOTH splits only
    byk = collections.defaultdict(list)
    for i in np.where(el)[0]: byk[names[lab[i]]].append(int(i))
    skills = sorted([k for k, v in byk.items() if len(v) >= 2], key=lambda k: (-len(byk[k]), k)); q = {k: 9 + (1 if j < 300 - 9 * len(skills) else 0) for j, k in enumerate(skills)}
    rng = random.Random(16); seeds = {k: rng.sample(byk[k], 2) for k in skills}
    json.dump({"skills": skills, "quota": q, "seed_rows": seeds, "eligible_counts": {k: len(byk[k]) for k in skills}, "total": sum(q.values())}, open(F("a16-seeds.json"), "w"), indent=1)
    held = {nz(p["q"]) for p in pairs}; briefs = {k: brief(k) for k in skills}; poolA = []; seen = set()
    from concurrent.futures import ThreadPoolExecutor
    def work(k):
        ex = [pairs[i]["q"][:300] for i in seeds[k]]; out = []; tries = 0
        while len(out) < 3 * q[k] and tries < 12:
            tries += 1
            for s in propose(k, briefs[k], ex):
                z = nz(s)
                if len(z.split()) >= 2 and len(s) <= 400 and z not in held and z not in {nz(o) for o in out}: out.append(s)
        return k, out[:3 * q[k]]
    with ThreadPoolExecutor(2) as ex_:
        for k, out in ex_.map(work, skills):
            poolA += [{"skill": k, "q": s} for s in out]; print(k, len(out), round(time.time() - t0), flush=True)
    open(F("a16-poolA.jsonl"), "w", encoding="utf-8").write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in poolA)); print("poolA", len(poolA))


def cmd_filter(a):
    """R0 router (time split, seed 0, real only) -> R1 = first quota per skill of poolA; R2 = failures (2 rounds)."""
    import shrink
    b = base(); names = b["names"]; ix = {nm: i for i, nm in enumerate(names)}; sd = json.load(open(F("a16-seeds.json"))); quota = sd["quota"]; pairs = b["pairs"]
    S, _, tokf = shrink.setup(argparse.Namespace()); rng = random.Random(7)
    m, r0, parts = train_split(b, "time", S, seed=0); m.eval(); delta, res, _ = pick_delta(S["time"], parts, m, names, tokf); print("R0 time", res["test"], "delta", delta, flush=True)
    a0 = arms.A0(); pool_x, pool_l = b["X"][b["sub"]], b["lab"][b["sub"]]
    poolA = [json.loads(l) for l in open(F("a16-poolA.jsonl"), encoding="utf-8")]
    R1 = []; cnt = collections.Counter()
    for r in poolA:
        if cnt[r["skill"]] < quota[r["skill"]]: R1.append(r); cnt[r["skill"]] += 1
    def fails(rs):
        part = syn_part(b, [r["q"] for r in rs], pool_x, pool_l, [ix[r["skill"]] for r in rs], names, rng); rk = rank_with(m, delta, a0, part["rows"], part, names)
        return [r for r, k in zip(rs, rk) if not k or k[0] != r["skill"]], np.mean([bool(k and k[0] == r["skill"]) for r, k in zip(rs, rk)])
    f1, solved1 = fails(poolA); print("round1 candidates", len(poolA), "router top1 solves", round(float(solved1), 3), "failures", len(f1), flush=True)
    keep = collections.defaultdict(list)
    for r in f1: keep[r["skill"]].append(dict(r, round=1))
    briefs = {k: brief(k) for k in sd["skills"]}; used = {nz(r["q"]) for r in poolA} | {nz(p["q"]) for p in pairs}; stats = {"round1": {"n": len(poolA), "failures": len(f1)}}
    from concurrent.futures import ThreadPoolExecutor
    need = [k for k in sd["skills"] if len(keep[k]) < quota[k]]
    def work(k):
        ex = [pairs[i]["q"][:300] for i in sd["seed_rows"][k]]; hard = [r["q"] for r in keep[k]][:4]; out = []; tries = 0
        while len(out) < 4 * (quota[k] - len(keep[k])) and tries < 10:
            tries += 1
            for s in propose(k, briefs[k], ex, hard, temp=1.0):
                z = nz(s)
                if len(z.split()) >= 2 and len(s) <= 400 and z not in used and z not in {nz(o["q"]) for o in out}: out.append({"skill": k, "q": s})
        return out
    with ThreadPoolExecutor(2) as ex_: outs = list(ex_.map(work, need))
    r2cand = [r for o in outs for r in o]; f2, solved2 = fails(r2cand) if r2cand else ([], 0); print("round2 candidates", len(r2cand), "solved", round(float(solved2), 3), "failures", len(f2), flush=True)
    for r in f2:
        if len(keep[r["skill"]]) < quota[r["skill"]]: keep[r["skill"]].append(dict(r, round=2))
    stats["round2"] = {"n": len(r2cand), "failures": len(f2)}; R2 = [dict(r, round=r.get("round", 1)) for k in sd["skills"] for r in keep[k][:quota[k]]]
    for nm, rs in (("R1", R1), ("R2", R2)): open(F(f"a16-prompts-{nm}.jsonl"), "w", encoding="utf-8").write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rs))
    json.dump({"R0_time_test": res["test"], "delta": delta, "stats": stats, "R1_n": len(R1), "R2_n": len(R2), "R2_per_skill": {k: len(keep[k][:quota[k]]) for k in sd["skills"]}, "quota": quota}, open(F("a16-filter.json"), "w"), indent=1); print("R1", len(R1), "R2", len(R2), stats)


def bucket_stats(a0, b, S_sp, rk_test, names):
    ei = S_sp["ei"]; lab = b["lab"]; out = {"1": [], "2-5": [], ">5": []}; pairs = b["pairs"]
    for j, i in enumerate(ei):
        if lab[i] < 0: continue
        gold = {names[g] for g in b["allp"][i]}; sc = [nm for nm, _ in a0.score(pairs[i])]; rk = min([sc.index(g) + 1 for g in gold if g in sc] or [99])
        out["1" if rk == 1 else "2-5" if rk <= 5 else ">5"].append(bool(rk_test[j] and rk_test[j][0] in gold))
    return {k: [round(float(np.mean(v)), 3) if v else None, len(v)] for k, v in out.items()}


def cmd_eval(a):
    import shrink, avb_np, holdout_check as hc
    b = base(); names = b["names"]; ix = {nm: i for i, nm in enumerate(names)}; pairs = b["pairs"]; rng = random.Random(9)
    S, _, tokf = shrink.setup(argparse.Namespace()); a0 = arms.A0(); out = F("a16-results.json"); R = json.load(open(out)) if os.path.exists(out) else {}
    sets = {"R0": [], "R1": [json.loads(l) for l in open(F("a16-prompts-R1.jsonl"), encoding="utf-8")], "R2": [json.loads(l) for l in open(F("a16-prompts-R2.jsonl"), encoding="utf-8")]}
    knnp = {"time": b["sub"], "session": b["tr_s"]}; embs = {}
    for arm in a.arms.split(","):
        rs = sets[arm]; R.setdefault(arm, {})
        for sp in ("time", "session"):
            if rs:
                pk = knnp[sp]; sp_part = syn_part(b, [r["q"] for r in rs], b["X"][pk], b["lab"][pk], [ix[r["skill"]] for r in rs], names, rng); eT, ey = sp_part["T"][sp_part["y"] >= 0], sp_part["y"][sp_part["y"] >= 0]
            else: eT = ey = None
            for sd in range(a.seeds):
                if len(R[arm].get(sp, [])) > sd: continue
                t0 = time.time(); m, r, parts = train_split(b, sp, S, eT, ey, sd); m.eval(); delta, res, rkt = pick_delta(S[sp], parts, m, names, tokf)
                row = {"test_hit1": res["test"]["hit1"], "test_acc": res["test"]["acc_with_abstain"], "delta": delta, "buckets": bucket_stats(a0, b, S[sp], rkt, names), "syn_rows_used": 0 if eT is None else int(len(eT)), "head_argmax_test": r["argmax"]["hit1"]}
                if sp == "time":
                    hp = os.path.join(tempfile.gettempdir(), f"a16_{arm}_{sd}.npz"); din = int(parts["train"]["T"].shape[2])
                    avb_np.save_head(hp, m.state_dict(), {"thr": r["thr_val"], "K": 16, "din": din, "seed": sd, "val_ce": r["val_ce"], "epochs": r["epochs"], "extra_rows": 0, "created": time.strftime("%F %T"), "src": "a16"})
                    h = hc.score(hp); row["holdout"] = {k: h[k] for k in ("hit1", "hit3", "acc_with_abstain", "n_labelled")}
                R[arm].setdefault(sp, []).append(row); print(arm, sp, sd, round(time.time() - t0), row, flush=True); json.dump(R, open(out, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["gen", "filter", "eval"]); ap.add_argument("--arms", default="R0,R1,R2"); ap.add_argument("--seeds", type=int, default=3); a = ap.parse_args()
    {"gen": cmd_gen, "filter": cmd_filter, "eval": cmd_eval}[a.cmd](a)

