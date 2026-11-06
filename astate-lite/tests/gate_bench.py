"""gate_bench.py: the deterministic gate (SKILL.md section 3) against the frozen skill-jepa holdout.
Prompts + gold labels: skilljepa/data/holdout.jsonl (1237 rows, never regenerated). 'Stated in the user's message'
is matched with the frozen per-skill trigger table skilljepa/data/a0_index.json (2026-10-29 snapshot, the same
table the production hand router uses), so no table is tuned on the holdout.
'Confirmed claim in STATE' is simulated per chat: after each labelled prompt the chat's STATE gains the claim
"project uses <skill>" (what a real Close would log), in time order, so a prompt only sees earlier rows of its chat.
Metrics follow ablate_20261105.py: hit1 on labelled rows, acc_abst over all rows, fr_none = false-load rate on none rows.
Writes results/gate_bench.json."""
import json, os, sys, re, collections

HERE = os.path.dirname(os.path.abspath(__file__))
SJ = r"C:\Users\john\Documents\Tools\skilljepa\data"
rows = [json.loads(l) for l in open(os.path.join(SJ, "holdout.jsonl"), encoding="utf-8")]
idx = json.load(open(os.path.join(SJ, "a0_index.json"), encoding="utf-8"))
skills = idx["skills"]
rows.sort(key=lambda r: (r["sid"], r["t"]))


def hits(q, name):
    """Trigger phrases of one skill present in the message, with weights."""
    ql = q.lower()
    qw = set(re.findall(r"[a-z][a-z0-9_.+-]{2,}", ql))
    tr = skills[name].get("triggers", {})
    return {w: wt for w, wt in tr.items() if (w in qw) or (" " in w and w in ql)}


def name_stated(q, name):
    ql = q.lower()
    return name in ql or name.replace("-", " ") in ql or any(a in ql for a in skills[name].get("aliases", []))


def candidates(q, state_skill, variant):
    """Which skills pass the gate for this prompt under a variant; score = trigger weight sum (tie rule, GAP-9)."""
    out = {}
    for name in skills:
        h = hits(q, name)
        strong = any(wt >= 0.8 for wt in h.values())
        in_state = state_skill == name
        ok = {"name": name_stated(q, name),
              "strong": strong,
              "any": bool(h),
              "strong_or_state": strong or (in_state and bool(h)),
              "state_and_any": in_state and bool(h),
              "name_or_strong": name_stated(q, name) or strong,
              "two_items": len(h) >= 2 and strong}[variant]
        if ok:
            out[name] = sum(h.values()) + (1.0 if name_stated(q, name) else 0.0)
    return out


def _score(variant):
    """Per row, in chat time order: (labelled, loaded a skill, loaded the gold skill)."""
    state, out = {}, []
    for r in rows:
        gold = set(r["skills"]) - {"none"}
        cand = candidates(r["q"], state.get(r["sid"]), variant)
        top = max(cand, key=cand.get) if cand else None
        out.append((bool(gold), top is not None, bool(gold) and top in gold))
        if gold:
            state[r["sid"]] = sorted(gold)[0]
    return out


def _ci(k, n, z=1.96):
    """Wilson 95% interval of k/n (s8 item 10, 11-06); None when n == 0."""
    if not n:
        return None
    p, zz = k / n, z * z
    c = (p + zz / (2 * n)) / (1 + zz / n)
    h = z * (p * (1 - p) / n + zz / (4 * n * n)) ** 0.5 / (1 + zz / n)
    return [round(c - h, 3), round(c + h, 3)]


def _metrics(sc):
    n = len(sc)
    nl = hits = loads = wrong = none_loads = 0
    for lab, rt, h in sc:
        nl += lab
        hits += h
        loads += rt
        wrong += rt and not h
        none_loads += rt and not lab
    correct = hits + (n - nl - none_loads)        # loaded-and-right + abstained-on-none (ablate_20261105 acc_abst)
    return {"hit1": round(hits / nl, 3), "hit1_ci": _ci(hits, nl), "acc_abst": round(correct / n, 3), "acc_abst_ci": _ci(correct, n),
            "fr_none": round(none_loads / (n - nl), 3), "fr_none_ci": _ci(none_loads, n - nl), "loads": loads,
            "precision": round((loads - wrong) / loads, 3) if loads else None, "precision_ci": _ci(loads - wrong, loads)}


def run(variant):
    return _metrics(_score(variant))


def main():
    res = {"n": len(rows), "n_lab": sum(1 for r in rows if set(r["skills"]) - {"none"}), "always_abstain": {"hit1": 0.0, "acc_abst": round(sum(1 for r in rows if r["skills"] == ["none"]) / len(rows), 3), "fr_none": 0.0}}
    n, nl, nn = res["n"], res["n_lab"], sum(1 for r in rows if r["skills"] == ["none"])
    res["base_rates"] = {"labelled": round(nl / n, 3), "labelled_ci": _ci(nl, n), "none": round(nn / n, 3), "n_none": nn,
                         "state": "STATE in _score holds only claims from EARLIER labelled rows of the same chat (gold = confirmed), never the row itself"}
    res["always_abstain"]["acc_abst_ci"] = _ci(nn, n)
    for v in ("name", "strong", "two_items", "any", "name_or_strong", "strong_or_state", "state_and_any"):
        res[v] = run(v)
        print(v, res[v], flush=True)
    ab = os.path.join(SJ, "ablate-20261105.json")
    if os.path.exists(ab):
        a = json.load(open(ab, encoding="utf-8"))
        res["A0_hand_router_at_NOMATCH"] = a.get("X0") or a.get("results", {}).get("X0")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "results", "gate_bench.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
