"""avb_np.py: torch-free forward pass + file format for the AVB v2 head (avb_rerank.Head, attn=True), so opsd / holdout_check run on plain numpy.
Head file = data/avb_head.npz: one array per state_dict key + __meta__ (json: thr, K, din, hit1_val, created, src). Written by save_head (avb_rerank.py --save), read by load_head.
Net: Linear(din,256)+GELU -> 2x pre-norm TransformerEncoderLayer(256, 4 heads, ff 512, relu, LN eps 1e-5) over K+1 slots -> LN, Linear(256,128), GELU, Linear(128,1)."""
import json, math
import numpy as np

_erf = np.frompyfunc(math.erf, 1, 1)


def gelu(x):
    return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)).astype(np.float64)).astype(np.float32)


def ln(x, w, b, eps=1e-5):
    m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True); return (x - m) / np.sqrt(v + eps) * w + b


def save_head(path, state, meta):
    arr = {k: v.detach().cpu().numpy().astype(np.float32) if hasattr(v, "detach") else np.asarray(v, np.float32) for k, v in state.items()}
    np.savez(path, __meta__=np.array(json.dumps(meta)), **arr)


def load_head(path):
    z = np.load(path, allow_pickle=False); W = {k: z[k] for k in z.files if k != "__meta__"}; W["__meta__"] = json.loads(str(z["__meta__"])); return W


def forward(W, T, heads=4):
    """T (n, K+1, din) float32 -> logits (n, K+1). Same math as avb_rerank.Head in eval mode."""
    h = gelu(T @ W["inp.1.weight"].T + W["inp.1.bias"]); n, s, d = h.shape; dh = d // heads
    for i in range(1 + max([int(k.split(".")[2]) for k in W if k.startswith("enc.layers.")], default=-1)):   # 10-29: block count from the weights (d16b0 = 0 blocks)
        p = f"enc.layers.{i}."; x = ln(h, W[p + "norm1.weight"], W[p + "norm1.bias"])
        qkv = x @ W[p + "self_attn.in_proj_weight"].T + W[p + "self_attn.in_proj_bias"]; q, k, v = (qkv[..., j * d:(j + 1) * d].reshape(n, s, heads, dh).transpose(0, 2, 1, 3) for j in range(3))
        a = q @ k.transpose(0, 1, 3, 2) / math.sqrt(dh); a = np.exp(a - a.max(-1, keepdims=True)); a /= a.sum(-1, keepdims=True)
        o = (a @ v).transpose(0, 2, 1, 3).reshape(n, s, d); h = h + o @ W[p + "self_attn.out_proj.weight"].T + W[p + "self_attn.out_proj.bias"]
        x = ln(h, W[p + "norm2.weight"], W[p + "norm2.bias"]); ff = np.maximum(x @ W[p + "linear1.weight"].T + W[p + "linear1.bias"], 0)
        h = h + ff @ W[p + "linear2.weight"].T + W[p + "linear2.bias"]
    x = ln(h, W["out.0.weight"], W["out.0.bias"]); x = gelu(x @ W["out.1.weight"].T + W["out.1.bias"])
    return (x @ W["out.3.weight"].T + W["out.3.bias"])[..., 0]
