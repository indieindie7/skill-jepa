"""effect_veto_bench.py (2026-11-09, integration item 5): the gate as a VETO for effectful skills only.
Data: frozen skill-jepa holdout (skilljepa/data/holdout.jsonl, 1237 rows) + frozen trigger table a0_index.json.
Router stand-in: top-1 by trigger weight sum (same table gate_bench.py uses; not the production route()).
For each effectful skill: rows the stand-in routes to it; false = gold labels do not contain it.
Veto = the contract's `requires` (read from PROPOSED below, or the live SKILL.md front matter with --live, via astate_lite.parse_contract) is unmet by
the message (no STATE claims: worst case for the veto). Reports false loads blocked and true loads blocked.
Writes results/effect_veto_bench.json. Usage: py -3.13 -I effect_veto_bench.py [--show]"""
import ast, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import astate_lite as al

V22 = "--v22" in sys.argv
SJ = os.path.join(os.path.dirname(os.path.dirname(HERE)), "data")          # inside the skill-jepa repo
if not os.path.isdir(SJ):
    SJ = os.path.expanduser("~/Documents/Tools/skilljepa/data")
SK = os.path.expanduser("~/.claude/skills")
rows = [json.loads(l) for l in open(os.path.join(SJ, "holdout.jsonl"), encoding="utf-8")]
skills = json.load(open(os.path.join(SJ, "a0_index.json"), encoding="utf-8"))["skills"]


def score(q, name):
    ql = q.lower()
    qw = set(re.findall(r"[a-z][a-z0-9_.+-]{2,}", ql))
    return sum(wt for w, wt in skills[name].get("triggers", {}).items() if (w in qw) or (" " in w and w in ql))


def top1(q):
    best = max(skills, key=lambda n: score(q, n))
    return best if score(q, best) > 0 else None


PROPOSED = {  # written before any holdout row was looked at: one action word the user must say
    "mod-release": ("release", "git push to a public repo, Nexus upload"),
    "advent-test-launch": ("launch", "starts the game on the user's desktop"),
    "wake-chats": ("message", "sends messages to other sessions"),
    "convert-movie": ("convert", "re-encodes and replaces video files"),
    "pc-cleanup": ("clean", "deletes files, moves games, disables services"),
}


def fm_text(name, req, eff):
    lines = ["---", f"name: {name}", "description: x", "metadata:", f"  requires: [{req}]", "  reads: []",
             "  produces: []", f"  effects: [{eff}]", "  conflicts: []", "---", ""]
    return chr(10).join(lines)


def contracts(live):
    out = {}
    for name in (sorted(os.listdir(SK)) if live else PROPOSED):
        try:
            text = open(os.path.join(SK, name, "SKILL.md"), encoding="utf-8-sig").read() if live else fm_text(name, *PROPOSED[name])
            c = al.parse_contract(text)
        except Exception:
            continue
        if c["metadata"]["effects"]:
            out[name] = c
    return out


def bench_one(name, c):
    r = dict(requires=c["metadata"]["requires"], routed=0, false=0, false_blocked=0, true=0, true_blocked=0, examples=[])
    for row in rows:
        if top1(row["q"]) != name:
            continue
        gold = ast.literal_eval(row["skills"]) if isinstance(row["skills"], str) else row["skills"]
        if V22:   # v2.2 gate, mechanical part only: the user named the skill; its effects taken as authorised (11-10)
            load, _ = al.gate(c, [], row["q"], authorised=c["metadata"]["effects"])
        else:
            load, _ = al.gate_v1(c, [], row["q"])
        ok = name in gold
        r["routed"] += 1
        r["true" if ok else "false"] += 1
        if not load:
            r["true_blocked" if ok else "false_blocked"] += 1
        r["examples"].append(("T" if ok else "F") + ("-" if not load else "+") + " " + row["q"][:110].replace(chr(10), " "))
    return r


def main():
    cs = contracts("--live" in sys.argv)
    res = {"n_rows": len(rows), "skills": {n: bench_one(n, c) for n, c in cs.items() if n in skills}}
    keys = ("routed", "false", "false_blocked", "true", "true_blocked")
    res["total"] = {k: sum(r[k] for r in res["skills"].values()) for k in keys}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "results", "effect_veto_bench_v22.json" if V22 else "effect_veto_bench.json"), "w", encoding="utf-8"), indent=1)
    for name, r in res["skills"].items():
        print(name, {k: v for k, v in r.items() if k != "examples"})
        for e in (r["examples"] if "--show" in sys.argv else []):
            print("   ", e)
    print("TOTAL", res["total"])


if __name__ == "__main__":
    main()
