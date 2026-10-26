"""es_data.py: shared data loader + prompt builder for es_train.py / sft_warm.py (no torch). Split: md5(query)%5==0 + (exclude_eval) session fold 0."""
import glob, hashlib, json, os, random
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
SK_INDEX = r"C:\Users\john\AppData\Local\Temp\claude\C--\a65df56e-768a-48c0-afa4-3bf9b0548cb5\scratchpad\jobs\skills_index.txt"
LET = "ABCDEFGHI"

def load_data(rows_path, nomatch_path, exclude_eval, seed=0):
    skills = [l.split(":", 1)[0].strip() for l in open(SK_INDEX, encoding="utf-8") if ":" in l]
    allr = sorted((json.loads(l) for l in open(rows_path, encoding="utf-8")), key=lambda r: r["t"])
    prev, last = [], {}
    for r in allr:   # previous rewarded label in the same sid; unknown sid -> "none"
        prev.append(last.get(r["sid"], "none") if r["sid"] else "none")
        if r["reward"] == 1 and r["label"]: last[r["sid"]] = r["label"]
    rows = [dict(q=r["query"], sid=r["sid"], top=r["route_top"], label=r["label"], prev=p) for r, p in zip(allr, prev)
            if r["reward"] == 1 and r["label"] and r["label"] in skills]
    rows += [dict(q=j["prompt"], sid="", top=[], label="none", prev="none") for j in map(json.loads, open(nomatch_path, encoding="utf-8"))]
    seen, uniq = set(), []
    for r in rows:   # replayed route.log rows overcount: count a (query,label) once
        if (r["q"], r["label"]) not in seen: seen.add((r["q"], r["label"])); uniq.append(r)
    exs, exq = set(), set()
    if exclude_eval:
        pr = [json.loads(l) for l in open(os.path.join(DATA, "pairs.jsonl"), encoding="utf-8")]
        us = np.unique([p["sid"] for p in pr]); np.random.default_rng(0).shuffle(us)
        exs = {s for i, s in enumerate(us) if i % 5 == 0}; exq = {p["q"] for p in pr if p["sid"] in exs}
        for f in glob.glob(os.path.join(DATA, "eval-session-*.json")) + glob.glob(os.path.join(DATA, "eval-time-*.json")):
            d = json.load(open(f)); exs |= set(d.get("sids", []) if isinstance(d, dict) else [])   # metrics-only today; future-proof
    h5 = lambda q: int(hashlib.md5(q.encode()).hexdigest(), 16) % 5 == 0
    tr, ev = [], []
    for r in uniq: (ev if h5(r["q"]) or (exclude_eval and (r["sid"] in exs or r["q"] in exq)) else tr).append(r)
    rng = random.Random(seed)
    for r in tr + ev:   # fixed candidate lists per row: route_top + label + random fillers, shuffled
        c = [s for s in dict.fromkeys(r["top"]) if s in skills][:8]
        if r["label"] != "none" and r["label"] not in c: c = c[:7] + [r["label"]]
        c += rng.sample([s for s in skills if s not in c], 8 - len(c)); rng.shuffle(c)
        r["y"] = 8 if r["label"] == "none" else c.index(r["label"])
        r["text"] = f"Previous skill: {r['prev']}\nQuery: {r['q'][:400]}\nCandidates: " + " ".join(f"{LET[i]}) {s}" for i, s in enumerate(c)) + " I) none\nAnswer:"
    return tr, ev

def letter_logits(core, W_let, ids_rows, pad, dev):
    """(n, 9) logits of the 9 letters at the last real token (right-padded + mask). core = causal LM with .model; W_let = lm_head rows of LET."""
    import torch
    T = max(len(i) for i in ids_rows)
    ids = torch.tensor([i + [pad] * (T - len(i)) for i in ids_rows], device=dev)
    am = torch.tensor([[1] * len(i) + [0] * (T - len(i)) for i in ids_rows], device=dev)
    h = core.model(input_ids=ids, attention_mask=am).last_hidden_state
    return h[torch.arange(len(ids_rows), device=dev), am.sum(1) - 1].float() @ W_let.float().T
