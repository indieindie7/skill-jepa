"""trace_pairs.py: data/trace_pairs.jsonl = one row per routed prompt (join of ops/route.log + skills/usage.jsonl).
Join copied from ops/route_trace.py (that file imports opsd at import time, so not importable): same sid, 0..600 s window.
reward: +1 brief_only/full_read, -1 miss, 0 none/none+read.
10-27 exploration slot: route.log col 6 'explore=<skill>' -> class explore_hit (skill read in the window; label = that skill, reward +1)
or explore_miss (reward 0); the normal class of the real top candidate is kept as class_main (same as ops/route_trace.py cls_main).
  py -3.13 -I trace_pairs.py [--days N] [--out PATH]"""
import os, sys, json, time, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROUTE_LOG = os.path.join(HERE, "..", "ops", "route.log")
USAGE = os.path.expanduser("~/.claude/skills/usage.jsonl")
WINDOW = 600
REWARD = {"brief_only": 1, "full_read": 1, "miss": -1, "none": 0, "none+read": 0, "explore_hit": 1, "explore_miss": 0}


def ts(s, fmt):
    try: return time.mktime(time.strptime(s, fmt))
    except ValueError: return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "data", "trace_pairs.jsonl"))
    a = ap.parse_args()
    t_min = time.time() - a.days * 86400 if a.days else 0
    use = []
    if os.path.exists(USAGE):
        for l in open(USAGE, encoding="utf-8"):
            try: j = json.loads(l); t = ts(j["t"], "%Y-%m-%d %H:%M")
            except (ValueError, KeyError): continue
            if t: use.append((t, j["skill"], j.get("sid", "")))
    rows, seen = [], set()
    for l in open(ROUTE_LOG, encoding="utf-8"):
        p = l.rstrip("\n").split("\t")
        if len(p) < 4: continue
        t = ts(p[0], "%Y-%m-%d %H:%M:%S")
        if not t or t < t_min or p[3].startswith("<") or len(p[3]) < 8: continue
        conf = p[1]; cands = [c.split(":")[0] for c in p[2].split(",") if c]
        sid = p[4] if len(p) > 4 else ""
        ex = p[5][8:] if len(p) > 5 and p[5].startswith("explore=") else ""
        if (sid, p[0], p[3]) in seen: continue
        seen.add((sid, p[0], p[3]))
        read = sorted({sk for ut, sk, us in use if 0 <= ut - t <= WINDOW and (not sid or not us or us == sid) and len(sk) > 2})
        top = cands[0] if cands else ""
        if conf == "none": cls = "none+read" if read else "none"
        elif not read: cls = "brief_only"
        elif top in read: cls = "full_read"
        else: cls = "miss"
        rs = top if top in read else (read[0] if read else "")
        label = top if cls == "brief_only" else rs
        cls_main = cls
        if ex:
            cls = "explore_hit" if ex in read else "explore_miss"
            label = ex if cls == "explore_hit" else ""
        rows.append({"query": p[3], "sid": sid, "t": p[0], "route_top": cands[:3], "conf": conf, "class": cls, "class_main": cls_main, "explore": ex,
                     "read_skill": rs, "label": label, "reward": REWARD[cls] if label else 0})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    by = {}
    for r in rows: by[r["class"]] = by.get(r["class"], 0) + 1
    n = sum(1 for r in rows if r["reward"])
    print(f"rows {len(rows)} " + " ".join(f"{k} {by.get(k, 0)}" for k in REWARD) + f" rewarded {n} -> {a.out}")
    if n < 300: print(f"STOP partial: {n} of 300")


if __name__ == "__main__":
    main()
