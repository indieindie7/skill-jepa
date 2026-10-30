"""a15_test.py: causality + param count + CPU smoke for the Grassmann mixer (avb_rerank.Head mixer='grassmann'). C:/llm/qlora/venv/Scripts/python.exe a15_test.py"""
import numpy as np, torch
import avb_rerank as ar
torch.manual_seed(0); din = 3 * 384 + 12; ar.ARCH.update(mixer="grassmann", lexf=True)
gd, tgt = ar.match_d(din); n_g = sum(p.numel() for p in ar.Head(din, gd, mixer="grassmann").parameters()); n_p = sum(p.numel() for p in ar.Head(din, 256, mixer="attn").parameters())
print("params prod d256b2", n_p, "grassmann d", gd, n_g, "ratio", round(n_g / n_p, 3)); assert abs(n_g / n_p - 1) < 0.05
# causality: output at the token ranked i must not change when tokens ranked > i are perturbed (rank key = knn rank + BM25 rank - prior)
m = ar.Head(din, gd, mixer="grassmann"); m.eval(); K = 16; T = torch.randn(4, K + 1, din)
T[:, :K, -10] = torch.rand(4, K) * 0.5; T[:, :K, -4] = torch.rand(4, K) * 0.5; T[:, :K, -1] = 0
key = T[:, :K, -10] + T[:, :K, -4] - T[:, :K, -1]; order = key.argsort(1)
with torch.no_grad(): z0 = m(T)
bad = 0
for i in range(K):
    T2 = T.clone()
    for b in range(4):
        later = order[b, i + 1:]; T2[b, later, :din - 12] = torch.randn(len(later), din - 12)   # change content, keep rank scalars so the order is stable
    with torch.no_grad(): z1 = m(T2)
    for b in range(4):
        keep = order[b, :i + 1]
        if not torch.allclose(z0[b, keep], z1[b, keep], atol=1e-5): bad += 1
print("causality violations", bad); assert bad == 0
assert not torch.isnan(z0).any(); print("CAUSALITY OK")
