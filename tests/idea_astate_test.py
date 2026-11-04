"""Idea test: ASTATE scoped evidence spans vs compact_check-style 'value appears anywhere' for compaction exact values.
Real claims = (section, value) pairs from SELFIMPROVE_PLAN.md (section = '##' header id, values = 7-hex commits + .ddd metrics).
Mutants: (a) invented values, (b) swapped = real value of ANOTHER arm (exists in file, wrong attribution - the failure compact_check cannot see).
gain = (astate catch - baseline catch) on all mutants - astate false-reject rate on real claims. Writes verdict json."""
import json, os, random, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import astate as A
P = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "SELFIMPROVE_PLAN.md")
st = {"version": 1, "sources": [A.src(P)], "claims": []}; b = st["sources"][0]["_b"]
heads = [(m.start(), m.group(1).decode()) for m in re.finditer(rb"^##+ (\S+)", b, re.M)]   # subject = section id
secs = {}
for i, (a, nm) in enumerate(heads): secs.setdefault(nm, []).append((a, heads[i + 1][0] if i + 1 < len(heads) else len(b)))
near = lambda bb, s: secs.get(s, [])
VAL = rb"(?<![0-9a-zA-Z])([0-9a-f]{7}(?![0-9a-zA-Z])|\.\d{3}(?!\d))"
real = sorted({(nm, m.group(1).decode()) for nm, ws in secs.items() for a, z in ws for m in re.finditer(VAL, b[a:z]) if re.search(rb"[a-f]", m.group(1)) or m.group(1).startswith(b".")})
for s, o in real: st["claims"].append(A.claim(s, "value", o, A.support(st, s, "value", o, near)))
assert not A.validate(st), A.validate(st)[:3]
rnd = random.Random(7); vals = sorted({o for _, o in real}); arms = sorted(secs)
inv = [(s, "".join(rnd.choice("0123456789abcdef") for _ in range(7)) if len(o) == 7 else "." + str(rnd.randrange(1000)).zfill(3)) for s, o in rnd.sample(real, min(60, len(real)))]
inv = [(s, o) for s, o in inv if o.encode() not in b]
own = {s: {o for t, o in real if t == s} for s in arms}
swp = [(s, o) for s, o in ((rnd.choice(arms), rnd.choice(vals)) for _ in range(400)) if o not in own[s]][:60]
base = lambda s, o: o.encode() in b; ast = lambda s, o: bool(A.support(st, s, "value", o, near))
mut = inv + swp
bc = sum(not base(*x) for x in mut) / len(mut); ac = sum(not ast(*x) for x in mut) / len(mut); fr = sum(not ast(*x) for x in real) / len(real)
m2 = A.merge(st, {"version": 1, "sources": st["sources"], "claims": st["claims"][:10]})
r = {"real": len(real), "invented": len(inv), "swapped": len(swp), "base_catch": round(bc, 3), "astate_catch": round(ac, 3), "astate_false_reject": round(fr, 3),
     "merge_dedup_ok": len(m2["claims"]) == len(st["claims"]), "conflicts_same_arm_multi_value": len(m2["conflicts"])}
r["gain"] = round(ac - bc - fr, 3); print(json.dumps(r))
out = sys.argv[1] if len(sys.argv) > 1 else None
if out: json.dump({"gain": r["gain"], "note": json.dumps(r)}, open(out, "w"))
print("VERDICT " + json.dumps({"gain": r["gain"], "note": f"astate catch {r['astate_catch']} vs anywhere-check {r['base_catch']}, false reject {r['astate_false_reject']}, n real {r['real']}"}))
