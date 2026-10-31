import json, glob, numpy as np, sys
fs = sorted(glob.glob("data/a13b-s[0-9].json")); R = [json.load(open(f)) for f in fs]; arms = [k for k in R[0] if "drop:" not in k]
m = lambda k, sp, f: float(np.mean([r[k][sp][f] for r in R if k in r])); sd = lambda k, sp, f: float(np.std([r[k][sp][f] for r in R if k in r]))
base = {sp: m("d256b2", sp, "cv_hit1") for sp in ("time", "session")}; out = {"seeds": len(R), "arms": {}, "base_cv": base}
for k in arms:
    out["arms"][k] = {"params": R[0][k]["time"]["params"], **{f"{sp}_{f}": m(k, sp, f) for sp in base for f in ("cv_hit1", "test_hit1", "test_acc")}, **{f"{sp}_cv_sd": sd(k, sp, "cv_hit1") for sp in base}, **{f"{sp}_cv_perseed": [r[k][sp]["cv_hit1"] for r in R] for sp in base}}
    out["arms"][k]["holds"] = all(out["arms"][k][f"{sp}_cv_hit1"] >= base[sp] - .005 for sp in base)
ok = [k for k in arms if k != "d256b2" and out["arms"][k]["params"] <= 168000 and out["arms"][k]["holds"]]; out["pick"] = min(ok, key=lambda k: out["arms"][k]["params"]) if ok else None
dr = [k for k in R[0] if "drop:" in k]; out["drops"] = {k: {sp: m(k, sp, "cv_hit1") for sp in base} for k in dr}
json.dump(out, open("data/a13b-20261030.json", "w"), indent=1)
print("| arm | params | CV time | CV sess | test time h1/acc | test sess h1/acc | holds |\n|---|---|---|---|---|---|---|")
for k, v in out["arms"].items(): print(f"| {k} | {v['params']} | {v['time_cv_hit1']:.3f}+-{v['time_cv_sd']:.3f} | {v['session_cv_hit1']:.3f}+-{v['session_cv_sd']:.3f} | {v['time_test_hit1']:.3f}/{v['time_test_acc']:.3f} | {v['session_test_hit1']:.3f}/{v['session_test_acc']:.3f} | {v['holds']} |")
print("base", base, "pick", out["pick"]); [print(k, {s: round(x, 3) for s, x in v.items()}) for k, v in out["drops"].items()]
