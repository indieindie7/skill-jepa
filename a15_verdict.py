"""a15_verdict.py: PRE-REGISTERED A15 (SELFIMPROVE_PLAN 10-30): Grassmann/Pluecker causal mixer vs production attention in the AVB head. Do not change the rule.
  C:/llm/qlora/venv/Scripts/python.exe a15_verdict.py [--idea ID] [--seeds 3]      (CPU only; ~1 h: 2 x avb_rerank + 2 x arms.py)
Steps per mixer in (attn d256b2, grassmann d tuned to +-5% params): (1) avb_rerank --extra <first 213 pairs_trace rows = production setup> --no-refit, 3 seeds, val-fit head of lowest-val-CE seed + its val thr;
(2) that head scored on the FROZEN holdout (holdout_check.verify + the same pipeline as holdout_check.score, torch forward instead of avb_np); (3) arms.py (frozen A0 index, B1, time + session splits, 3 seeds).
Accept iff grassmann holdout hit1 >= .600 AND grassmann session B1 hit1 >= .325 (prod, SELFIMPROVE_PLAN). gain = session B1 hit1 (grassmann - prod, both measured here) if accepted, else min(diff, 0) - 0.001.
Writes data/a15-grassmann-<date>.json and llm/autorun/logs/idea<ID>.verdict.json."""
import argparse, json, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "data"); sys.path.insert(0, HERE)
PY = "C:/llm/qlora/venv/Scripts/python.exe"; VERDICT_DIR = "C:/Users/john/Documents/Tools/llm/autorun/logs"; IDEA = 131; NX = 213; HOLD_MIN, SESS_MIN = 0.600, 0.325
NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def sh(args, tag):
    t = time.time(); r = subprocess.run([PY] + args, cwd=HERE, creationflags=NOWIN, capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": "1"})  # gpu1 only: never gpu0 (user games on it)
    open(os.path.join(D, f"a15-{tag}.log"), "w", encoding="utf-8").write((r.stdout or "") + "\n" + (r.stderr or "")); print(tag, "rc", r.returncode, round(time.time() - t), "s", flush=True)
    if r.returncode: raise SystemExit(f"{tag} failed rc {r.returncode}: {(r.stderr or r.stdout)[-400:]}")


def extra_dir():
    import numpy as np
    d = os.path.join(D, "a15_extra"); os.makedirs(d, exist_ok=True); rows = open(os.path.join(D, "pairs_trace.jsonl"), encoding="utf-8").read().splitlines(True)[:NX]
    open(os.path.join(d, "pairs_trace.jsonl"), "w", encoding="utf-8", newline="").writelines(rows); np.savez(os.path.join(d, "emb_trace.npz"), X=np.load(os.path.join(D, "emb_trace.npz"))["X"][:NX]); return os.path.join(d, "pairs_trace.jsonl")


def holdout_torch(pt_path, js_path, mixer, d):
    """frozen-holdout metrics of the best-val-CE seed's val-fit head (torch forward; pipeline = holdout_check.score)."""
    import numpy as np, torch, holdout_check as hc, avb_feats as af, avb_rerank as ar
    ok, vmsg = hc.verify(); assert ok, vmsg
    ar.ARCH.update(d=d, mixer=mixer, lexf=True); ck = torch.load(pt_path, weights_only=False); runs = json.load(open(js_path))["avb2_runs"]; b = min(range(len(runs)), key=lambda i: runs[i]["val_ce"])
    m = ar.Head(ck["din"]); m.load_state_dict(ck["state"][b][0]); m.eval(); thr = runs[b]["thr_val"]; K = ck["K"]
    sj, X, Y, lab, allp, sid, names, plab, te = hc._base(); rows = [json.loads(l) for l in open(hc.HOLD, encoding="utf-8")]; ei = np.array([r["i"] for r in rows]); assert (ei == np.where(te)[0]).all()
    B = np.array([r["B"] for r in rows], np.float32); Pr = np.zeros_like(B)
    for k, r in enumerate(rows):
        for j, v in r["Pr"].items(): Pr[k, int(j)] = v
    n = len(lab); ap = [allp[i - 1] if i and sid[i - 1] == sid[i] else set() for i in range(n)]
    E = sj.knn_energies(X[~te], lab[~te], X[ei], len(Y)); cs = af.cands(E, B, K, True); T = af.feats(X[ei], Y, E, B, Pr, cs, plab[ei], [ap[j] for j in ei], True)
    with torch.no_grad(): Z = m(torch.tensor(T)).numpy()
    P = {"cands": cs, "allp": [allp[j] for j in ei], "lab": lab[ei]}
    return {k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in (af.metrics(Z, P, thr) | {"seed": b, "thr": thr, "val_ce": runs[b]["val_ce"]}).items()}


def main():
    global IDEA
    ap = argparse.ArgumentParser(); ap.add_argument("--idea", type=int, default=IDEA); ap.add_argument("--seeds", type=int, default=3); a = ap.parse_args(); date = time.strftime("%Y%m%d"); ex = extra_dir()
    import skilljepa as sj, avb_rerank as ar
    X = sj.data("prevskill", "all")[0]; din = 3 * X.shape[1] + 12; gd, tgt = ar.match_d(din); ar.ARCH.update(mixer="grassmann", lexf=True); pg = sum(p.numel() for p in ar.Head(din, gd).parameters())
    out = {"date": date, "rule": f"accept iff holdout hit1 >= {HOLD_MIN} and session B1 hit1 >= {SESS_MIN}", "params": {"prod_attn_d256b2": tgt, "grassmann": pg, "grassmann_d": gd, "ratio": round(pg / tgt, 4)}, "arms": {}}
    assert abs(pg / tgt - 1) <= 0.05, out["params"]
    for mixer, d in (("attn", 256), ("grassmann", gd)):
        tag = f"{mixer}-{date}"; js = os.path.join(D, f"avb-a15-{tag}.json"); sh(["avb_rerank.py", "--mixer", mixer, "--d", str(d), "--seeds", str(a.seeds), "--extra", ex, "--no-refit", "--out", js], "train-" + mixer)
        ho = holdout_torch(js.replace(".json", ".pt"), js, mixer, d); print(mixer, "holdout", ho, flush=True)
        ao = os.path.join(D, f"arms-a15-{tag}.json"); sh(["arms.py", "--mixer", mixer, "--d", str(d), "--seeds", str(a.seeds), "--out", ao], "arms-" + mixer); r = json.load(open(ao))
        out["arms"][mixer] = {"d": d, "holdout": ho, "A0_hit1": {s: r[s]["A0"]["test"]["hit1"] for s in ("time", "session")}, "B1_hit1": {s: r[s]["B1"]["hit1"] for s in ("time", "session")}, "B1_acc_abstain": {s: r[s]["B1"]["acc_with_abstain"] for s in ("time", "session")}, "arms_file": os.path.basename(ao)}
    g, p = out["arms"]["grassmann"], out["arms"]["attn"]; diff = round(g["B1_hit1"]["session"][0] - p["B1_hit1"]["session"][0], 4); h = g["holdout"]["hit1"]; s_ = g["B1_hit1"]["session"][0]
    acc = bool(h >= HOLD_MIN and s_ >= SESS_MIN); gain = diff if acc else min(diff, 0.0) - 0.001
    out["verdict"] = {"accepted": acc, "holdout_hit1": h, "session_B1_grassmann": s_, "session_B1_prod": p["B1_hit1"]["session"][0], "session_diff": diff, "time_B1_diff": round(g["B1_hit1"]["time"][0] - p["B1_hit1"]["time"][0], 4), "gain": gain,
                      "text": "SUPPORTED" if acc else "NOT SUPPORTED"}
    json.dump(out, open(os.path.join(D, f"a15-grassmann-{date}.json"), "w", encoding="utf-8"), indent=1)
    note = f"A15 {out['verdict']['text']}: holdout hit1 {h} (gate {HOLD_MIN}), session B1 {s_} vs prod {p['B1_hit1']['session'][0]} (gate {SESS_MIN}), time diff {out['verdict']['time_B1_diff']}; params {pg} vs {tgt}"
    if a.idea: json.dump({"gain": gain, "note": note}, open(os.path.join(VERDICT_DIR, f"idea{a.idea}.verdict.json"), "w", encoding="utf-8"))
    print(note)


if __name__ == "__main__": main()
