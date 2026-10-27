"""Stable rank of LoRA deltas: per module dW = B@A (x scale), sr = sum(sigma^2)/max(sigma)^2.
  C:/llm/qlora/venv/Scripts/python.exe stable_rank.py data/sft-20261027-r16-lora [--json out.json]"""
import sys, json, glob, os, numpy as np
from safetensors.numpy import load_file
d = sys.argv[1]; f = os.path.join(d, "adapter_model.safetensors")
try: T = load_file(f)
except Exception:
    from safetensors import safe_open
    import torch
    T = {}
    with safe_open(f, "pt") as s:
        for k in s.keys(): T[k] = s.get_tensor(k).float().numpy()
res = {}
for k in sorted(T):
    if "lora_A" in k:
        kb = k.replace("lora_A", "lora_B"); A = T[k].astype(np.float64); B = T[kb].astype(np.float64)
        s = np.linalg.svd(B @ A, compute_uv=False); res[k.split(".lora_A")[0]] = float((s ** 2).sum() / s[0] ** 2)
v = np.array(list(res.values()))
out = {"n_modules": len(v), "min": round(v.min(), 3), "median": round(float(np.median(v)), 3), "max": round(v.max(), 3), "rank_nominal": int(next(iter(T[k] for k in T if "lora_A" in k)).shape[0])}
print(json.dumps(out))
if "--json" in sys.argv: json.dump({**out, "per_module": res}, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
