"""es_train.py: ES (OpenES / EGGROLL) fine-tune of a small abliterated Qwen on skill routing. Spec: SELFIMPROVE_PLAN.md 3b.
  C:/llm/qlora/venv/Scripts/python.exe es_train.py --algo eggroll --exclude-eval [--smoke]
Prompt: Previous skill / Query / Candidates A)..H) + I) none / Answer:  fitness = mean P(correct letter) via next-token letter probs.
Trainable: peft LoRA r=16 on q_proj,v_proj (fp32 master, bf16 base). No backward pass: gradient = (1/(P*sigma)) sum z_i eps_i, Adam.
 openes : eps over all LoRA params, candidate i uses theta + sigma*eps_i (in-place swap, sequential forwards).
 eggroll: rank-1 noise E_i = a_i b_i^T on each target's output (hooks, Pc members in one batched forward), then chain-ruled into
          LoRA grads (dB = s g_W A^T, dA = s B^T g_W; g_W never materialised). Antithetic pairs (eps, -eps) in both.
Held-out eval: rows with md5(query)%5==0 + (--exclude-eval) rows whose sid/query is in session-split fold 0 of data/pairs.jsonl
(skilljepa.py eval: np.unique(sid), default_rng(0).shuffle, i%5; eval-*.json files store metrics only, no sids)."""
import argparse, glob, hashlib, json, os, random, sys, time, datetime
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
sys.path.insert(0, os.path.join(HERE, "..", "llm"))
SK_INDEX = r"C:\Users\john\AppData\Local\Temp\claude\C--\a65df56e-768a-48c0-afa4-3bf9b0548cb5\scratchpad\jobs\skills_index.txt"
ap = argparse.ArgumentParser()
ap.add_argument("--algo", choices=["openes", "eggroll"], default="eggroll"); ap.add_argument("--model", default=r"C:\llm\models\qwen35-0.8b-abl")
ap.add_argument("--rows", default=os.path.join(DATA, "trace_pairs.jsonl")); ap.add_argument("--nomatch", default=os.path.join(DATA, "nomatch.jsonl"))
ap.add_argument("--exclude-eval", action="store_true"); ap.add_argument("--pop", type=int, default=32); ap.add_argument("--sigma", type=float, default=0.02)
ap.add_argument("--lr", type=float, default=0.01); ap.add_argument("--gens", type=int, default=30); ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--eval-rows", type=int, default=256); ap.add_argument("--gpu", default="auto"); ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--targets", default="q_proj,v_proj"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--seqs", type=int, default=128, help="max seqs per forward")
ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
if a.smoke: a.gens, a.pop, a.batch, a.eval_rows = 2, 8, 32, 64

# ---- gpu (before torch touches cuda) ----
import gpu as gpumod
g = gpumod.pick_gpu(6000) if a.gpu == "auto" else int(a.gpu)
if g < 0 or (g == 0 and gpumod.game_running()): sys.exit(f"es_train: no gpu (pick={g}, game={gpumod.game_running()})")
os.environ["CUDA_VISIBLE_DEVICES"] = str(g)
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
dev = "cuda"; LET = "ABCDEFGHI"

# ---- data ----
def load_data():
    skills = [l.split(":", 1)[0].strip() for l in open(SK_INDEX, encoding="utf-8") if ":" in l]
    allr = sorted((json.loads(l) for l in open(a.rows, encoding="utf-8")), key=lambda r: r["t"])
    prev, last = [], {}
    for r in allr:   # previous rewarded label in the same sid; unknown sid -> "none"
        prev.append(last.get(r["sid"], "none") if r["sid"] else "none")
        if r["reward"] == 1 and r["label"]: last[r["sid"]] = r["label"]
    rows = [dict(q=r["query"], sid=r["sid"], top=r["route_top"], label=r["label"], prev=p) for r, p in zip(allr, prev)
            if r["reward"] == 1 and r["label"] and r["label"] in skills]
    rows += [dict(q=j["prompt"], sid="", top=[], label="none", prev="none") for j in map(json.loads, open(a.nomatch, encoding="utf-8"))]
    seen, uniq = set(), []
    for r in rows:   # replayed route.log rows overcount: count a (query,label) once
        if (r["q"], r["label"]) not in seen: seen.add((r["q"], r["label"])); uniq.append(r)
    exs, exq = set(), set()
    if a.exclude_eval:
        pr = [json.loads(l) for l in open(os.path.join(DATA, "pairs.jsonl"), encoding="utf-8")]
        us = np.unique([p["sid"] for p in pr]); np.random.default_rng(0).shuffle(us)
        exs = {s for i, s in enumerate(us) if i % 5 == 0}; exq = {p["q"] for p in pr if p["sid"] in exs}
        for f in glob.glob(os.path.join(DATA, "eval-session-*.json")) + glob.glob(os.path.join(DATA, "eval-time-*.json")):
            d = json.load(open(f)); exs |= set(d.get("sids", []) if isinstance(d, dict) else [])   # metrics-only today; future-proof
    h5 = lambda q: int(hashlib.md5(q.encode()).hexdigest(), 16) % 5 == 0
    tr, ev = [], []
    for r in uniq: (ev if h5(r["q"]) or (a.exclude_eval and (r["sid"] in exs or r["q"] in exq)) else tr).append(r)
    rng = random.Random(a.seed)
    for r in tr + ev:   # fixed candidate lists per row: route_top + label + random fillers, shuffled
        c = [s for s in dict.fromkeys(r["top"]) if s in skills][:8]
        if r["label"] != "none" and r["label"] not in c: c = c[:7] + [r["label"]]
        c += rng.sample([s for s in skills if s not in c], 8 - len(c)); rng.shuffle(c)
        r["y"] = 8 if r["label"] == "none" else c.index(r["label"])
        r["text"] = f"Previous skill: {r['prev']}\nQuery: {r['q'][:400]}\nCandidates: " + " ".join(f"{LET[i]}) {s}" for i, s in enumerate(c)) + " I) none\nAnswer:"
    return tr, ev

tr, ev = load_data(); random.Random(1).shuffle(ev); ev = ev[:a.eval_rows]
print(f"rows: train {len(tr)} eval {len(ev)} (none: {sum(r['y']==8 for r in tr)}/{sum(r['y']==8 for r in ev)}) gpu {g}", flush=True)

# ---- model ----
tok = AutoTokenizer.from_pretrained(a.model)
lid = [tok.encode(" " + c, add_special_tokens=False) for c in LET]; assert all(len(i) == 1 for i in lid), lid
base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map=dev, trust_remote_code=False).eval()
pm = get_peft_model(base, LoraConfig(r=a.rank, lora_alpha=a.rank, lora_dropout=0.0, target_modules=a.targets.split(",")))
SC = 1.0   # alpha/r
core = pm.base_model.model
W_let = core.get_output_embeddings().weight[[i[0] for i in lid]].detach()
mods = [(n, m) for n, m in pm.named_modules() if hasattr(m, "lora_A") and "default" in getattr(m, "lora_A", {})]
LA = [m.lora_A["default"].weight for _, m in mods]; LB = [m.lora_B["default"].weight for _, m in mods]   # A (r,in) B (out,r) fp32
params = LA + LB; print(f"lora modules {len(mods)} params {sum(p.numel() for p in params)} dtype {params[0].dtype}", flush=True)
for r in tr + ev: r["ids"] = tok(r["text"], add_special_tokens=False).input_ids
noise = None   # eggroll: per module (a (Pc,out,1), b (Pc,in,1), scale)
def hook(i):
    def h(mod, inp, out):
        if noise is None: return out
        A_, B_, sc = noise[i]; x = inp[0]; Pc = A_.shape[0]
        d = ((x.reshape(Pc, -1, x.shape[-1]) @ B_) @ A_.transpose(1, 2)) * sc
        return out + d.reshape(out.shape).to(out.dtype)
    return h
for i, (_, m) in enumerate(mods): m.register_forward_hook(hook(i))

@torch.no_grad()
def probs(rows, reps=1):
    """(reps, len(rows), 9) letter probs; reps copies of the batch are stacked member-major (for eggroll noise)."""
    out, mb = [], max(1, a.seqs // reps)
    for s in range(0, len(rows), mb):
        rr = rows[s:s + mb]; T = max(len(r["ids"]) for r in rr)
        ids = torch.tensor([r["ids"] + [tok.pad_token_id or 0] * (T - len(r["ids"])) for r in rr], device=dev)
        am = torch.tensor([[1] * len(r["ids"]) + [0] * (T - len(r["ids"])) for r in rr], device=dev)
        h = core.model(input_ids=ids.repeat(reps, 1), attention_mask=am.repeat(reps, 1)).last_hidden_state
        last = am.sum(1).repeat(reps) - 1
        z = h[torch.arange(h.shape[0], device=dev), last] @ W_let.T
        out.append(F.softmax(z.float(), -1).reshape(reps, len(rr), -1))
    return torch.cat(out, 1)

def score(rows, reps=1):
    y = torch.tensor([r["y"] for r in rows], device=dev); p = probs(rows, reps)
    return p.gather(2, y[None, :, None].expand(reps, -1, 1)).squeeze(-1).mean(1), (p.argmax(-1) == y).float().mean(1)

def evaluate():
    f, ac = score(ev); return ac.item(), f.item()

# ---- ES step ----
opt = torch.optim.Adam(params, lr=a.lr); half = a.pop // 2
@torch.no_grad()
def step(batch):
    global noise
    if a.algo == "openes":
        eps = [[torch.randn_like(p) for p in params] for _ in range(half)]; eps += [[-e for e in es] for es in eps]
        theta = [p.detach().clone() for p in params]; fit = []
        for es in eps:
            for p, t, e in zip(params, theta, es): p.copy_(t + a.sigma * e)
            fit.append(score(batch)[0][0])
        for p, t in zip(params, theta): p.copy_(t)
        fit = torch.stack(fit); z = (fit - fit.mean()) / (fit.std() + 1e-8)
        for k, p in enumerate(params):
            p.grad = -sum(z[i] * eps[i][k] for i in range(a.pop)) / (a.pop * a.sigma)
    else:
        nz = []
        for _, m in mods:   # rank-1 noise per target module, scale sigma/sqrt(in): ~sigma*rms(x) relative output change
            o, i_ = m.base_layer.out_features, m.base_layer.in_features
            A_ = torch.randn(half, o, 1, device=dev, dtype=torch.bfloat16); B_ = torch.randn(half, i_, 1, device=dev, dtype=torch.bfloat16)
            nz.append((torch.cat([A_, -A_]), torch.cat([B_, B_]), a.sigma / i_ ** 0.5))
        Pc = max(1, min(a.pop, a.seqs // min(len(batch), 64))); fit = []
        for c in range(0, a.pop, Pc):
            n = min(Pc, a.pop - c); noise = [(A_[c:c + n], B_[c:c + n], sc) for A_, B_, sc in nz]
            fit.append(score(batch, reps=n)[0])
        noise = None; fit = torch.cat(fit); z = (fit - fit.mean()) / (fit.std() + 1e-8)
        for k, (A_, B_, sc) in enumerate(nz):
            A_, B_ = A_.float(), B_.float(); c0 = SC / (a.pop * sc)   # g_W = (1/(P*sc)) sum z_i a_i b_i^T ; W = SC*B_l A_l
            bA = torch.einsum("pie,ri->per", B_, LA[k].float())
            LB[k].grad = -torch.einsum("p,poe,per->or", z, A_, bA) * c0
            Ba = torch.einsum("or,poe->pre", LB[k].float(), A_)
            LA[k].grad = -torch.einsum("p,pre,pie->ri", z, Ba, B_) * c0
    opt.step(); opt.zero_grad(set_to_none=True)
    return fit.mean().item(), fit.max().item()

# ---- run ----
random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.reset_peak_memory_stats()
tag = datetime.date.today().strftime("%Y%m%d") + "-" + a.algo + ("-smoke" if a.smoke else "")
log = open(os.path.join(DATA, f"es-{tag}.jsonl"), "w"); base_acc, base_p = evaluate()
print(json.dumps({"baseline_acc": round(base_acc, 4), "baseline_p": round(base_p, 4)}), flush=True)
best = (base_acc, base_p); best_state = [p.detach().clone() for p in params]; t0 = time.time(); secs = []
for gen in range(a.gens):
    t = time.time(); fm, fb = step(random.sample(tr, min(a.batch, len(tr)))); torch.cuda.synchronize(); acc, ep = evaluate(); secs.append(time.time() - t)
    if (acc, ep) > best: best, best_state = (acc, ep), [p.detach().clone() for p in params]
    rec = {"gen": gen, "fit_mean": round(fm, 4), "fit_best": round(fb, 4), "acc_eval": round(acc, 4), "p_eval": round(ep, 4), "secs": round(secs[-1], 1)}
    log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps(rec), flush=True)
fin_acc = acc
with torch.no_grad():
    for p, b in zip(params, best_state): p.copy_(b)
out = os.path.join(DATA, f"es-{tag}-lora"); pm.save_pretrained(out)
print(json.dumps({"summary": {"baseline_acc": round(base_acc, 4), "final_acc": round(fin_acc, 4), "best_acc": round(best[0], 4), "s_per_gen": round(sum(secs) / len(secs), 1),
      "vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2), "lora": out, "total_s": round(time.time() - t0)}}), flush=True)
