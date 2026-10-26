"""policy_eval.py: SFT policy as a rerank tier on the skilljepa --split time test rows (last 30% of every chat), comparable to skilljepa.cmd_eval.
  C:/llm/qlora/venv/Scripts/python.exe policy_eval.py --lora data/sft-20261026-lora [--base knn|zero] [--order shuffle|rank] --out data/policy-<date>.json
Candidates = top-8 of the base ranking (knn_energies on the train 70% by default); prompt as es_data; abstain = policy picks I) none.
hit1/hit3 rank the 8 skill letters (as report()); hit1_strict = argmax over all 9 letters; metrics follow skilljepa.report."""
import argparse, datetime, json, os, random, sys, time
import numpy as np
from es_data import load_data, letter_logits, LET, DATA, HERE
sys.path.insert(0, os.path.join(HERE, "..", "llm"))
import skilljepa as sj
ap = argparse.ArgumentParser(); ap.add_argument("--lora", required=True); ap.add_argument("--model", default=r"C:\llm\models\qwen35-0.8b-abl")
ap.add_argument("--split", default="time"); ap.add_argument("--ctx", default="prevskill"); ap.add_argument("--topk", type=int, default=8)
ap.add_argument("--base", default="knn", choices=["zero", "knn"]); ap.add_argument("--order", default="shuffle", choices=["shuffle", "rank"])
ap.add_argument("--gpu", default="auto"); ap.add_argument("--bs", type=int, default=16); ap.add_argument("--out"); a = ap.parse_args()
assert a.split == "time" and a.topk == 8, "only the time split with 8 candidates (prompt has A-H + I)"
X, Y, lab, allp, sid, names, plab = sj.data(a.ctx, "all"); n = len(lab)
pos, cnt = np.zeros(n), {}
for i, c in enumerate(sid): pos[i] = cnt.get(c, 0); cnt[c] = pos[i] + 1
tr = pos < 0.7 * np.array([cnt[c] for c in sid]); te = ~tr           # identical to cmd_eval split=="time"
pairs = [json.loads(l) for l in open(os.path.join(DATA, "pairs.jsonl"), encoding="utf-8")]
atr, ate = [allp[i] for i in np.where(tr)[0]], [allp[i] for i in np.where(te)[0]]
if a.base == "knn": Etr, Ete = sj.knn_energies(X[tr], lab[tr], X[tr], len(Y)), sj.knn_energies(X[tr], lab[tr], X[te], len(Y))
else: Etr, Ete = sj.energies(None, X[tr], Y), sj.energies(None, X[te], Y)
tau = sj.tune_tau(Etr, lab[tr], atr); base = sj.report(Ete, lab[te], ate, tau); base = {k: round(float(v), 3) for k, v in base.items()}
tid = np.where(te)[0]; rk = Ete.argsort(1)[:, :a.topk]; lt = lab[te]; rng = random.Random(0)
cands, texts = [], []
for j, i in enumerate(tid):
    c = [int(s) for s in rk[j]]
    if a.order == "shuffle": rng.shuffle(c)   # the SFT saw shuffled candidate lists
    cands.append(c); prev = names[plab[i]] if plab[i] >= 0 else "none"
    texts.append(f"Previous skill: {prev}\nQuery: {pairs[i]['q'][:400]}\nCandidates: " + " ".join(f"{LET[k]}) {names[s]}" for k, s in enumerate(c)) + " I) none\nAnswer:")
recall8 = float(np.mean([bool(set(c) & ate[j]) for j, c in enumerate(cands) if lt[j] >= 0]))
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
ids = [tok(t, add_special_tokens=False).input_ids for t in texts]; order = np.argsort([len(i) for i in ids]); Z = np.zeros((len(ids), 9), np.float32); t0 = time.time(); torch.cuda.reset_peak_memory_stats()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    for s in range(0, len(ids), a.bs):
        b = order[s:s + a.bs]; Z[b] = letter_logits(core, W, [ids[k] for k in b], tok.pad_token_id or 0, "cuda").float().cpu().numpy()
secs, vram = round(time.time() - t0, 1), round(torch.cuda.max_memory_allocated() / 1e9, 2)
rl = np.argsort(-Z[:, :8], 1); route = Z.argmax(1) != 8; L = lt >= 0
hitk = lambda j, k: bool(set(cands[j][x] for x in rl[j, :k]) & ate[j])
h1 = np.array([hitk(j, 1) for j in range(len(tid))]); h3 = np.array([hitk(j, 3) for j in range(len(tid))])
strict = np.array([bool(route[j] and cands[j][Z[j].argmax()] in ate[j]) for j in range(len(tid))])
pol = {"hit1": h1[L].mean(), "hit3": h3[L].mean(), "acc_with_abstain": np.where(~route, lt < 0, h1).mean(), "false_route_on_none": route[~L].mean(),
       "abstain_on_labelled": (~route[L]).mean(), "hit1_strict": strict[L].mean()}
seen_q = {r["q"] for r in load_data(os.path.join(DATA, "trace_pairs.jsonl"), os.path.join(DATA, "nomatch.jsonl"), True)[0]}
seen = np.array([pairs[i]["q"] in seen_q for i in tid]); u = L & ~seen
out = {"policy": {k: round(float(v), 3) for k, v in pol.items()}, "base": base, "base_kind": a.base, "recall8": round(recall8, 3), "order": a.order,
       "n": {"test": int(te.sum()), "labelled": int(L.sum()), "none": int((~L).sum()), "train": int(tr.sum()), "sft_train_query_overlap": int(seen.sum()), "labelled_unseen": int(u.sum())},
       "always_abstain_acc": round(float((lt < 0).mean()), 3), "unseen_hit1": round(float(h1[u].mean()), 3) if u.any() else None, "secs": secs, "vram_gb": vram, "gpu": g, "lora": a.lora}
print(json.dumps(out, indent=1))
json.dump(out, open(a.out or os.path.join(DATA, f"policy-{datetime.date.today():%Y%m%d}.json"), "w"), indent=1)
