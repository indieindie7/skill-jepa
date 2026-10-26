"""policy_eval.py: SFT policy as a rerank tier on the skilljepa --split time test rows (last 30% of every chat), comparable to skilljepa.cmd_eval.
  C:/llm/qlora/venv/Scripts/python.exe policy_eval.py --lora data/sft-20261026-lora [--base knn|zero] [--order shuffle|rank] [--none-thr auto|<float>] --out data/policy-<date>.json
Candidates = top-8 of the base ranking (knn_energies on the train 70% by default); prompt as es_data; abstain = policy picks I) none.
hit1/hit3 rank the 8 skill letters (as report()); hit1_strict = routed AND top letter right; metrics follow skilljepa.report.
--none-thr: route to the top skill letter unless P(I) (softmax over the 9 letters) >= thr. `auto` tunes thr on VAL = positions [0.49,0.70) of each chat (the last 30% of the
time-split TRAIN part; knn for VAL built from positions < 0.49 only), maximising acc_with_abstain + hit1 (hit1 does not depend on thr), never on the test part.
Importable: build() -> split dict, evaluate(core, W, tok, S, ...) -> metrics (used by sft_warm.py)."""
import argparse, datetime, json, os, random, sys, time
import numpy as np
from es_data import load_data, letter_logits, LET, DATA, HERE
sys.path.insert(0, os.path.join(HERE, "..", "llm"))
import skilljepa as sj
VAL_LO = 0.49   # VAL = train-part positions [0.49, 0.70) of each chat; SUB = positions < 0.49


def _part(idx, E, rng, order, topk, names, plab, pairs, lab, allp):
    rk = E.argsort(1)[:, :topk]; cands, texts = [], []
    for j, i in enumerate(idx):
        c = [int(s) for s in rk[j]]
        if order == "shuffle": rng.shuffle(c)   # the SFT saw shuffled candidate lists
        cands.append(c); prev = names[plab[i]] if plab[i] >= 0 else "none"
        texts.append(f"Previous skill: {prev}\nQuery: {pairs[i]['q'][:400]}\nCandidates: " + " ".join(f"{LET[k]}) {names[s]}" for k, s in enumerate(c)) + " I) none\nAnswer:")
    return {"idx": idx, "cands": cands, "texts": texts, "lab": lab[idx], "allp": [allp[i] for i in idx], "q": [pairs[i]["q"] for i in idx],
            "prev": [names[plab[i]] if plab[i] >= 0 else "none" for i in idx]}


def build(ctx="prevskill", topk=8, order="shuffle", base="knn"):
    """-> dict(test, val, pool, base metrics, ...). test = exactly the old policy_eval rows/candidates; val = held-out slice of train; pool = SUB none rows (SFT none candidates)."""
    X, Y, lab, allp, sid, names, plab = sj.data(ctx, "all"); n = len(lab)
    pos, cnt = np.zeros(n), {}
    for i, c in enumerate(sid): pos[i] = cnt.get(c, 0); cnt[c] = pos[i] + 1
    tot = np.array([cnt[c] for c in sid]); tr = pos < 0.7 * tot; te = ~tr          # identical to cmd_eval split=="time"
    sub = pos < VAL_LO * tot; val = tr & ~sub
    pairs = [json.loads(l) for l in open(os.path.join(DATA, "pairs.jsonl"), encoding="utf-8")]
    E = (lambda m, q: sj.knn_energies(X[m], lab[m], X[q], len(Y))) if base == "knn" else (lambda m, q: sj.energies(None, X[q], Y))
    Etr, Ete = E(tr, tr), E(tr, te)
    atr, ate = [allp[i] for i in np.where(tr)[0]], [allp[i] for i in np.where(te)[0]]
    tau = sj.tune_tau(Etr, lab[tr], atr); bm = {k: round(float(v), 3) for k, v in sj.report(Ete, lab[te], ate, tau).items()}
    args = (order, topk, names, plab, pairs, lab, allp)
    pi = np.where(sub & (lab < 0))[0]; seenq, keep = set(), []
    for i in pi:   # unique none queries only
        if pairs[i]["q"] not in seenq: seenq.add(pairs[i]["q"]); keep.append(i)
    pm = np.zeros(n, bool); pm[keep] = True
    S = {"test": _part(np.where(te)[0], Ete, random.Random(0), *args), "val": _part(np.where(val)[0], E(sub, val), random.Random(1), *args),
         "pool": _part(np.where(pm)[0], E(sub, pm), random.Random(2), *args), "base": bm, "n_train": int(tr.sum())}
    t = S["test"]; S["recall8"] = float(np.mean([bool(set(c) & t["allp"][j]) for j, c in enumerate(t["cands"]) if t["lab"][j] >= 0]))
    return S


def score(core, W, tok, texts, bs=16, dev="cuda"):
    """(n, 9) float32 letter logits, rows in input order."""
    import torch
    ids = [tok(t, add_special_tokens=False).input_ids for t in texts]; order = np.argsort([len(i) for i in ids]); Z = np.zeros((len(ids), 9), np.float32)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for s in range(0, len(ids), bs):
            b = order[s:s + bs]; Z[b] = letter_logits(core, W, [ids[k] for k in b], tok.pad_token_id or 0, dev).float().cpu().numpy()
    return Z


def p_none(Z):
    z = Z.astype(np.float64); z -= z.max(1, keepdims=True); e = np.exp(z); return e[:, 8] / e.sum(1)


def _hits(Z, P):
    rl = np.argsort(-Z[:, :8], 1); ate = P["allp"]; c = P["cands"]
    h = lambda j, k: bool(set(c[j][x] for x in rl[j, :k]) & ate[j])
    return np.array([h(j, 1) for j in range(len(c))]), np.array([h(j, 3) for j in range(len(c))])


def metrics(Z, P, thr=None):
    """thr None: route = argmax is a skill letter; else route = P(I) < thr."""
    lt = P["lab"]; L = lt >= 0; h1, h3 = _hits(Z, P)
    route = (Z.argmax(1) != 8) if thr is None else (p_none(Z) < thr)
    m = {"hit1": h1[L].mean(), "hit3": h3[L].mean(), "acc_with_abstain": np.where(~route, lt < 0, h1).mean(), "false_route_on_none": route[~L].mean(),
         "abstain_on_labelled": (~route[L]).mean(), "hit1_strict": (route & h1)[L].mean()}
    return {k: round(float(v), 3) for k, v in m.items()}, h1, route


def tune_thr(Zv, Pv):
    """thr maximising acc_with_abstain + hit1 on VAL (hit1 constant in thr): route when P(I) < thr; grid = P(I) quantiles + both extremes."""
    pn = p_none(Zv); lt = Pv["lab"]; L = lt >= 0; h1, _ = _hits(Zv, Pv); best = (-1.0, 0.0)
    for t in np.r_[0.0, np.quantile(pn, np.linspace(0, 1, 101)), 1.0001]:
        best = max(best, (np.where(pn < t, h1, lt < 0).mean() + h1[L].mean(), float(t)))
    return best[1]


def evaluate(core, W, tok, S, thr="auto", seen_q=None, bs=16, dev="cuda"):
    """Score test (+ val when thr=='auto') with the model in hand; returns {"policy": argmax metrics, "policy_thr": ..., "thr", "unseen_hit1", ...}."""
    t = S["test"]; t0 = time.time(); Z = score(core, W, tok, t["texts"], bs, dev)
    pol, h1, _ = metrics(Z, t); out = {"policy": pol, "base": S["base"], "recall8": round(S["recall8"], 3), "secs": round(time.time() - t0, 1)}
    L = t["lab"] >= 0; seen = np.array([q in seen_q for q in t["q"]]) if seen_q is not None else np.zeros(len(L), bool); u = L & ~seen
    out["unseen_hit1"] = round(float(h1[u].mean()), 3) if u.any() else None; out["n_unseen_labelled"] = int(u.sum()); out["n_seen_test_queries"] = int(seen.sum())
    if thr is not None:
        v = S["val"]; th = tune_thr(score(core, W, tok, v["texts"], bs, dev), v) if thr == "auto" else float(thr)
        if thr == "auto": out["thr_val"] = {"n": len(v["texts"]), "labelled": int((v["lab"] >= 0).sum())}
        pt, h1t, route = metrics(Z, t, th); out["policy_thr"] = pt; out["thr"] = round(th, 4)
        out["unseen_acc_with_abstain_thr"] = round(float(np.where(~route, t["lab"] < 0, h1t)[~seen].mean()), 3) if (~seen).any() else None
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--lora", required=True); ap.add_argument("--model", default=r"C:\llm\models\qwen35-0.8b-abl")
    ap.add_argument("--split", default="time"); ap.add_argument("--ctx", default="prevskill"); ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--base", default="knn", choices=["zero", "knn"]); ap.add_argument("--order", default="shuffle", choices=["shuffle", "rank"])
    ap.add_argument("--none-thr", default=None, help="auto | float: route unless P(I) >= thr; reported next to argmax")
    ap.add_argument("--gpu", default="auto"); ap.add_argument("--bs", type=int, default=16); ap.add_argument("--out"); a = ap.parse_args()
    assert a.split == "time" and a.topk == 8, "only the time split with 8 candidates (prompt has A-H + I)"
    S = build(a.ctx, a.topk, a.order, a.base)
    import gpu as gpumod
    g = gpumod.pick_gpu(4000) if a.gpu == "auto" else int(a.gpu)
    if g < 0 or (g == 0 and gpumod.game_running()): sys.exit(f"policy_eval: no gpu (pick={g}, game={gpumod.game_running()})")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(g)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True); lid = [tok.encode(" " + c, add_special_tokens=False) for c in LET]; assert all(len(i) == 1 for i in lid)
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=False, local_files_only=True)
    pm = PeftModel.from_pretrained(m, a.lora).eval(); core = pm.base_model.model; W = core.get_output_embeddings().weight[[i[0] for i in lid]].detach()
    tq = os.path.join(a.lora, "train_queries.json")   # written by sft_warm.py: the queries the adapter really trained on
    seen_q = set(json.load(open(tq, encoding="utf-8"))) if os.path.exists(tq) else {r["q"] for r in load_data(os.path.join(DATA, "trace_pairs.jsonl"), os.path.join(DATA, "nomatch.jsonl"), True)[0]}
    torch.cuda.reset_peak_memory_stats()
    res = evaluate(core, W, tok, S, a.none_thr, seen_q, a.bs)
    lt = S["test"]["lab"]
    out = {**res, "base_kind": a.base, "order": a.order, "n": {"test": len(lt), "labelled": int((lt >= 0).sum()), "none": int((lt < 0).sum()), "train": S["n_train"],
           "sft_train_query_overlap": res["n_seen_test_queries"], "labelled_unseen": res["n_unseen_labelled"]}, "always_abstain_acc": round(float((lt < 0).mean()), 3),
           "vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2), "gpu": g, "lora": a.lora}
    print(json.dumps(out, indent=1))
    json.dump(out, open(a.out or os.path.join(DATA, f"policy-{datetime.date.today():%Y%m%d}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
