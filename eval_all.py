"""eval_all.py: one-shot router eval. Production router (ops/treeroute_eval.py) + skilljepa splits (data/eval-*.json).
  py -3.13 -I skilljepa/eval_all.py [--rerun] [--no-tree] [--compare OLD.json] [--out PATH]
Default reads the existing data/eval-<split>[-ctx[-labels]].json files (written by `skilljepa.py eval`).
--lora DIR adds sft_policy_rerank + base_knn + recall8 (policy_eval.py, ~1 min, needs a free GPU) to the time-prevskill-all stem.
--rerun re-runs skilljepa.py eval for session/prompt none+all and time prevskill+all (~70 s each, needs numpy + data/emb.npz).
Writes data/eval-<YYYYMMDD>.json = {"tree": {router: {hit1,hit3,n}}, "jepa": {stem: {model: {hit1,hit3,abstain_acc,...}}}}."""
import argparse, datetime, glob, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
TREE = os.path.join(HERE, "..", "ops", "treeroute_eval.py")
MODELS = ["zero_shot_cosine", "skill_jepa", "gate_soft", "gate_soft_followup", "base_zero", "base_knn", "recall8", "sft_policy_rerank", "sft_policy_rerank_bal", "sft_policy_rerank_bal_thr", "avb_head", "avb_head_thr", "avb_noattn", "avb_noattn_thr"]   # full set stays in the json
NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def run(cmd, timeout):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, creationflags=NOWIN)
    except subprocess.TimeoutExpired:
        print(f"skip: timeout {cmd[2:4]}", file=sys.stderr); return None
    if r.returncode:
        print(f"skip: rc={r.returncode} {(r.stderr or r.stdout).strip()[-200:]}", file=sys.stderr); return None
    return r.stdout


def tree_eval():
    """treeroute_eval.py has no CLI; parse rows `name  Hit@1  Hit@3  n  top1 dist` (print at L144-146)."""
    if not os.path.exists(TREE): print("skip tree: script missing", file=sys.stderr); return {}
    out = run([sys.executable, "-I", TREE], 600)
    res = {}
    for ln in (out or "").splitlines():
        m = re.match(r"^(.+?)\s+(\d\.\d{3})\s+(\d\.\d{3})\s+(\d+)\s+\[", ln)
        if m: res[m[1].strip()] = {"hit1": float(m[2]), "hit3": float(m[3]), "n": int(m[4])}
    return res


def rerun_jepa():
    try:
        import numpy  # noqa: F401  (-I env: venv-less py needs numpy installed)
    except ImportError:
        print("skip --rerun: numpy missing (skilljepa eval needs numpy + data/emb.npz)", file=sys.stderr); return
    if not os.path.exists(os.path.join(DATA, "emb.npz")):
        print("skip --rerun: data/emb.npz missing (run skilljepa.py embed, needs embed server 8084)", file=sys.stderr); return
    for sp, cx, lb in [("session", "none", "all"), ("prompt", "none", "all"), ("time", "prevskill", "all")] + [("session", "prevskill", "clean")]:
        print(f"rerun {sp}/{cx}/{lb} ...", file=sys.stderr)
        run([sys.executable, "-I", os.path.join(HERE, "skilljepa.py"), "eval", "--split", sp, "--ctx", cx, "--labels", lb], 900)


def jepa_load():
    res = {}
    for f in sorted(glob.glob(os.path.join(DATA, "eval-*.json"))):
        stem = os.path.basename(f)[5:-5]
        if re.fullmatch(r"\d{8}[a-z]?", stem): continue          # earlier eval_all outputs
        try: d = json.load(open(f, encoding="utf-8"))
        except (OSError, ValueError): continue
        row = {}
        for m, v in d.items():
            if isinstance(v, dict) and "hit1" in v:
                row[m] = {"hit1": v["hit1"], "hit3": v["hit3"], "abstain_acc": v.get("acc_with_abstain"),
                          "false_route": v.get("false_route_on_none"), "abstain_on_labelled": v.get("abstain_on_labelled")}
        if row: res[stem] = {"models": row, "n": d.get("n"), "always_abstain_acc": d.get("always_abstain_acc")}
    return res


QLORA_PY = "C:/llm/qlora/venv/Scripts/python.exe"


def policy_rows(lora, base="knn"):
    """Run policy_eval.py (time split, prevskill ctx, --none-thr auto) and return model rows for the time-prevskill-all stem.
    Names: sft_policy_rerank (+ _thr) for plain adapters, sft_policy_rerank_bal (+ _thr) when the adapter dir name has 'bal'."""
    tag = os.path.basename(lora.rstrip("/\\")); out = os.path.join(DATA, f"policy-{datetime.date.today():%Y%m%d}-{tag}.json")   # never overwrites policy-<date>.json
    if run([QLORA_PY, os.path.join(HERE, "policy_eval.py"), "--lora", lora, "--base", base, "--none-thr", "auto", "--out", out], 1800) is None: return {}
    p = json.load(open(out, encoding="utf-8")); m = lambda d: {"hit1": d["hit1"], "hit3": d["hit3"], "abstain_acc": d["acc_with_abstain"],
        "false_route": d["false_route_on_none"], "abstain_on_labelled": d["abstain_on_labelled"]}
    nm = "sft_policy_rerank_bal" if "bal" in tag else "sft_policy_rerank"
    rows_ = {nm: m(p["policy"]), f"base_{base}": m(p["base"]), "recall8": {"hit1": p["recall8"], "hit3": p["recall8"], "abstain_acc": None, "false_route": None, "abstain_on_labelled": None}}
    if "policy_thr" in p: rows_[nm + "_thr"] = {**m(p["policy_thr"]), "thr": p["thr"]}
    return rows_


def rows(res):
    out = [(f"tree:{k}", v["hit1"], v["hit3"], None) for k, v in res.get("tree", {}).items()]
    for stem, d in res.get("jepa", {}).items():
        for m in MODELS:
            if m in d["models"]:
                v = d["models"][m]; out.append((f"{stem}:{m}", v["hit1"], v["hit3"], v["abstain_acc"]))
    return out


def table(res, old=None):
    f = lambda x: "-" if x is None else f"{x:.3f}"
    oldr = {r[0]: r for r in rows(old)} if old else {}
    print("| run | Hit@1 | Hit@3 | abstain-acc |" + (" d Hit@1 | d Hit@3 | d abst |" if old else ""))
    print("|---|---|---|---|" + ("---|---|---|" if old else ""))
    for k, h1, h3, ab in rows(res):
        line = f"| {k} | {f(h1)} | {f(h3)} | {f(ab)} |"
        if old:
            o = oldr.get(k)
            d = lambda a, b: "-" if o is None or a is None or b is None else f"{a - b:+.3f}"
            line += f" {d(h1, o[1] if o else None)} | {d(h3, o[2] if o else None)} | {d(ab, o[3] if o else None)} |"
        print(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rerun", action="store_true"); ap.add_argument("--no-tree", action="store_true")
    ap.add_argument("--compare"); ap.add_argument("--out"); ap.add_argument("--avb", help="avb_rerank.py json: merge avb_head / avb_head_thr / avb_noattn rows (mean over seeds) into time-prevskill-all")
    ap.add_argument("--lora", help="SFT LoRA dir: add the policy rerank tier to time-prevskill-all")
    a = ap.parse_args()
    if a.rerun: rerun_jepa()
    res = {"date": datetime.date.today().isoformat(), "tree": {} if a.no_tree else tree_eval(), "jepa": jepa_load()}
    if a.lora:
        pr = policy_rows(os.path.abspath(a.lora))
        if pr: res["jepa"].setdefault("time-prevskill-all", {"models": {}})["models"].update(pr)
    if a.avb:
        d = json.load(open(a.avb, encoding="utf-8")); mm = lambda r, thr=False: {"hit1": r["hit1"][0], "hit3": r["hit3"][0], "abstain_acc": r["acc_with_abstain"][0], "false_route": r["false_route_on_none"][0],
            "abstain_on_labelled": r["abstain_on_labelled"][0], "std": {k: v[1] for k, v in r.items()}}
        rw = {k: mm(d[k]) for k in ("avb_head", "avb_head_thr", "avb_noattn", "avb_noattn_thr") if k in d}
        res["jepa"].setdefault("time-prevskill-all", {"models": {}})["models"].update(rw)
    path = a.out or os.path.join(DATA, f"eval-{datetime.date.today():%Y%m%d}.json")
    json.dump(res, open(path, "w", encoding="utf-8"), indent=1)
    old = None
    if a.compare:
        try: old = json.load(open(a.compare, encoding="utf-8"))
        except (OSError, ValueError) as e: print(f"compare skipped: {e}", file=sys.stderr)
        if old is not None and "jepa" not in old and "tree" not in old: print("compare skipped: not an eval_all json", file=sys.stderr); old = None
    table(res, old); print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
