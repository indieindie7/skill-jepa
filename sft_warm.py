"""sft_warm.py: SFT warm start for es_train.py. LoRA r=16, loss = CE over the 9 answer-letter logits at the last position (only the answer token), bf16 autocast.
  C:/llm/qlora/venv/Scripts/python.exe sft_warm.py --exclude-eval [--smoke]   ->  data/sft-<date>-lora/  (feed to es_train.py --init-lora)
  none rebalance (10-26): --none-frac 0.5 --exclude-time-test --exclude-val  ->  data/sft-<date>-bal-lora/ (+ train_queries.json); logs before/after time-split test metrics via policy_eval.evaluate"""
import argparse, json, os, random, sys, time, datetime
from es_data import load_data, LET, DATA, HERE, letter_logits as letter_logits_shared
sys.path.insert(0, os.path.join(HERE, "..", "llm"))
ap = argparse.ArgumentParser()
ap.add_argument("--model", default=r"C:\llm\models\qwen35-0.8b-abl"); ap.add_argument("--rows", default=os.path.join(DATA, "trace_pairs.jsonl"))
ap.add_argument("--nomatch", default=os.path.join(DATA, "nomatch.jsonl")); ap.add_argument("--exclude-eval", action="store_true")
ap.add_argument("--epochs", type=int, default=2); ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--eval-rows", type=int, default=256); ap.add_argument("--gpu", default="auto"); ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--targets", default="q_proj,v_proj"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--smoke", action="store_true")
ap.add_argument("--none-frac", type=float, default=0.0, help="oversample none rows (time-split TRAIN part of pairs.jsonl, unlabelled, knn top-8 candidates) until none ~= this fraction of the train set")
ap.add_argument("--exclude-time-test", action="store_true", help="drop train rows whose query is in the time-split test part")
ap.add_argument("--exclude-val", action="store_true", help="also drop train rows in the val slice (train positions 0.49-0.70) that policy_eval --none-thr auto tunes on")
a = ap.parse_args()
if a.smoke: a.epochs, a.eval_rows = 1, 64
import gpu as gpumod
g = gpumod.pick_gpu(6000) if a.gpu == "auto" else int(a.gpu)
if g < 0 or (g == 0 and gpumod.game_running()): sys.exit(f"sft_warm: no gpu (pick={g}, game={gpumod.game_running()})")
os.environ["CUDA_VISIBLE_DEVICES"] = str(g)
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
dev = "cuda"; random.seed(a.seed); torch.manual_seed(a.seed)
tr, ev = load_data(a.rows, a.nomatch, a.exclude_eval, a.seed); random.Random(1).shuffle(ev); ev = ev[:a.eval_rows]
PS = None
if a.exclude_time_test or a.exclude_val or a.none_frac > 0:
    import policy_eval as pe   # candidates/prompts exactly as the eval builds them
    PS = pe.build(); n0 = len(tr); drop = set()
    if a.exclude_time_test: drop |= set(PS["test"]["q"])
    if a.exclude_val: drop |= set(PS["val"]["q"])
    tr = [r for r in tr if r["q"] not in drop]; print(f"dropped {n0 - len(tr)} train rows (time-test/val queries)", flush=True)
if a.none_frac > 0:
    n_none = sum(r["y"] == 8 for r in tr); n_lab = len(tr) - n_none; need = round(n_lab * a.none_frac / (1 - a.none_frac)) - n_none
    have, P = {r["q"] for r in tr} | set(PS["test"]["q"]), PS["pool"]
    pool = [dict(q=P["q"][j], sid="", top=[], label="none", prev=P["prev"][j], y=8, text=P["texts"][j]) for j in range(len(P["q"])) if P["q"][j] not in have]
    rs = random.Random(a.seed + 7); add = rs.sample(pool, min(need, len(pool))) if need > 0 else []
    if need > len(add): add += [dict(r) for r in rs.choices(pool, k=need - len(add))]   # pool exhausted: oversample with repeats
    print(f"none rebalance: labelled {n_lab} none {n_none} pool {len(pool)} add {len(add)} (repeats {max(0, need - len(pool))})", flush=True); tr += add
if a.smoke: random.Random(0).shuffle(tr); tr = tr[:64]
n_none = sum(r["y"] == 8 for r in tr)
print(f"rows: train {len(tr)} (none {n_none}, {n_none / len(tr):.2f}) eval {len(ev)} gpu {g}", flush=True)
tok = AutoTokenizer.from_pretrained(a.model)
lid = [tok.encode(" " + c, add_special_tokens=False) for c in LET]; assert all(len(i) == 1 for i in lid), lid
base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map=dev, trust_remote_code=False)
pm = get_peft_model(base, LoraConfig(r=a.rank, lora_alpha=a.rank, lora_dropout=0.0, target_modules=a.targets.split(",")))
core = pm.base_model.model; W_let = core.get_output_embeddings().weight[[i[0] for i in lid]].detach()
params = [p for p in pm.parameters() if p.requires_grad]; print(f"trainable {sum(p.numel() for p in params)} dtype {params[0].dtype}", flush=True)
for r in tr + ev: r["ids"] = tok(r["text"], add_special_tokens=False).input_ids

def letter_logits(rows):   # (n, 9) logits at the last real token; shared code in es_data.letter_logits
    return letter_logits_shared(core, W_let, [r["ids"] for r in rows], tok.pad_token_id or 0, dev)

@torch.no_grad()
def evaluate():
    pm.eval(); ok = pc = 0.0
    for s in range(0, len(ev), 32):
        rr = ev[s:s + 32]; y = torch.tensor([r["y"] for r in rr], device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16): z = letter_logits(rr)
        p = F.softmax(z, -1); ok += (p.argmax(-1) == y).sum().item(); pc += p.gather(1, y[:, None]).sum().item()
    return round(ok / len(ev), 4), round(pc / len(ev), 4)

seen_q = {r["q"] for r in tr}
def time_eval():   # policy_eval as a function on the in-memory adapter: argmax + val-tuned P(I) threshold
    pm.eval(); return pe.evaluate(core, W_let, tok, PS, "auto", seen_q, 16, dev)
b_acc, b_p = evaluate(); print(json.dumps({"before": {"acc_eval": b_acc, "p_eval": b_p}}), flush=True)
if PS: b_time = time_eval(); print(json.dumps({"before_time_test": b_time}), flush=True)
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0); nb = (len(tr) + a.batch - 1) // a.batch; total = nb * a.epochs
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / 5) * max(0.0, 1 - i / total)); t0 = time.time(); torch.cuda.reset_peak_memory_stats(); hist = []
for ep in range(a.epochs):
    pm.train(); random.shuffle(tr); tot = 0.0
    for s in range(0, len(tr), a.batch):
        rr = tr[s:s + a.batch]; y = torch.tensor([r["y"] for r in rr], device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16): z = letter_logits(rr)
        loss = F.cross_entropy(z, y); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        tot += loss.item() * len(rr)
    acc, p = evaluate(); hist.append({"epoch": ep + 1, "train_loss": round(tot / len(tr), 4), "acc_eval": acc, "p_eval": p, "secs": round(time.time() - t0)})
    print(json.dumps(hist[-1]), flush=True)
out = os.path.join(DATA, f"sft-{datetime.date.today():%Y%m%d}" + ("-bal" if a.none_frac > 0 else "") + ("-smoke" if a.smoke else "") + "-lora"); pm.save_pretrained(out)
json.dump(sorted(seen_q), open(os.path.join(out, "train_queries.json"), "w", encoding="utf-8"))   # policy_eval: seen/unseen split
vram = round(torch.cuda.max_memory_allocated() / 1e9, 2)
if PS: a_time = time_eval(); print(json.dumps({"after_time_test": a_time}), flush=True)
print(json.dumps({"summary": {"before_acc": b_acc, "after_acc": hist[-1]["acc_eval"], "before_p": b_p, "after_p": hist[-1]["p_eval"], "loss": [h["train_loss"] for h in hist],
      "vram_gb": vram, "train_rows": len(tr), "train_none": n_none, "lora": out}}), flush=True)
