"""A6c shadow judge: nud=<skill> (rank-only nudge, shadow) vs prod avb argmax vs A0 top1, on route.log rows with nud.
Join rule copied from ops/route_trace.py: same sid (8 chars) when both sides have one, read within 0..600 s after route,
usage.jsonl minute-resolution; rows with query starting '<' or len<8 skipped. Promote iff nud_hit >= avb_hit on labelled rows."""
import os, json, time, collections
OPS = r"C:\Users\john\Documents\Tools\ops"
WINDOW = 600
def ts(s, fmt):
    try: return time.mktime(time.strptime(s, fmt))
    except ValueError: return None
use = []
for l in open(os.path.expanduser("~/.claude/skills/usage.jsonl"), encoding="utf-8"):
    try: j = json.loads(l); t = ts(j["t"], "%Y-%m-%d %H:%M")
    except (ValueError, KeyError): continue
    if t: use.append((t, j["skill"], j.get("sid", "")))
N = dict(n_nud_rows=0, n_used=0, n_labelled=0, n_none=0, nud_hit=0, avb_hit=0, a0_hit=0, nud_eq_avb=0, nud_only_hit=0, avb_only_hit=0)
per = collections.defaultdict(lambda: [0, 0, 0])
for l in open(os.path.join(OPS, "route.log"), encoding="utf-8"):
    p = l.rstrip("\n").split("\t")
    if len(p) < 4: continue
    nud = next((c[4:] for c in p[5:] if c.startswith("nud=")), "")
    if not nud: continue
    N["n_nud_rows"] += 1
    t = ts(p[0], "%Y-%m-%d %H:%M:%S")
    if not t or p[3].startswith("<") or len(p[3]) < 8: continue
    N["n_used"] += 1
    sid = p[4] if len(p) > 4 else ""
    read = {sk for ut, sk, us in use if 0 <= ut - t <= WINDOW and (not sid or not us or us == sid) and len(sk) > 2}
    if not read: N["n_none"] += 1; continue
    N["n_labelled"] += 1
    avb = next((c[4:] for c in p[5:] if c.startswith("avb=")), "")
    best, bl = "", -1e18
    for it in avb.split(","):
        k, _, v = it.rpartition(":")
        if not k or k == "_none": continue
        try: v = float(v)
        except ValueError: continue
        if v != v: continue
        if v > bl: best, bl = k, v
    a0 = p[2].split(",")[0].split(":")[0] if p[2] else ""
    nh, ah = nud in read, best in read
    N["nud_hit"] += nh; N["avb_hit"] += ah; N["a0_hit"] += a0 in read
    N["nud_eq_avb"] += nud == best; N["nud_only_hit"] += nh and not ah; N["avb_only_hit"] += ah and not nh
    s = per[sid or "?"]; s[0] += 1; s[1] += nh; s[2] += ah
N["per_sid"] = dict(sorted(per.items(), key=lambda kv: -kv[1][0])[:8])
N["promote"] = N["nud_hit"] >= N["avb_hit"]
json.dump(N, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "a6c-shadow-20261105.json"), "w"), indent=1)
print(json.dumps(N))
