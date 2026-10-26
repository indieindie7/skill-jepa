[part 1, lines 1-123]
```python
#!/usr/bin/env python3
"""skilljepa/trace_pairs.py — build data/trace_pairs.jsonl from ops/route_trace join logic."""
import os, sys, json, time, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
OPS_DIR = os.path.join(HERE, "..", "ops")
sys.path.insert(0, OPS_DIR)

WINDOW = 600


def ts(s, fmt):
    try:
        return time.mktime(time.strptime(s, fmt))
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=0)
    ap.add_argument("--out", type=str, default=os.path.join(HERE, "data", "trace_pairs.jsonl"))
    args = ap.parse_args()

    t_min = time.time() - args.days * 86400 if args.days else 0

    # Load usage.jsonl (same as route_trace.py L28-L32)
    use_path = os.path.expanduser("~/.claude/skills/usage.jsonl")
    use = []
    try:
        with open(use_path, encoding="utf-8") as f:
            for l in f:
                try:
                    j = json.loads(l)
                    t = ts(j["t"], "%Y-%m-%d %H:%M")
                except (ValueError, KeyError):
                    continue
                if t:
                    use.append((t, j["skill"], j.get("sid", "")))
    except FileNotFoundError:
        print(f"WARNING: {use_path} not found; no read info available.")

    # Load route.log (same as route_trace.py L37-L51)
    log_path = os.path.join(OPS_DIR, "route.log")
    rows = []
    try:
        with open(log_path, encoding="utf-8") as f:
            for l in f:
                p = l.rstrip("\n").split("\t")
                if len(p) < 4:
                    continue
                t = ts(p[0], "%Y-%m-%d %H:%M:%S")
                if not t or t < t_min or p[3].startswith("<") or len(p[3]) < 8:
                    continue
                conf = p[1]
                cands = [c.split(":")[0] for c in p[2].split(",") if c]
                sid = p[4] if len(p) > 4 else ""
                read = {sk for ut, sk, usid in use if 0 <= ut - t <= WINDOW and (not sid or not usid or usid == sid) and len(sk) > 2}
                top = cands[0] if cands else ""

                # Classify (same as route_trace.py L46-L49)
                if conf == "none":
                    cls = "none+read" if read else "none"
                elif not read:
                    cls = "brief_only"
                elif top in read:
                    cls = "full_read"
                else:
                    cls = "miss"

                # Determine label and reward per spec (L101-L102)
                if cls == "brief_only":
                    label = top  # route_top[0] for brief_only
                    reward = +1
                elif cls == "full_read":
                    label = top  # skill actually read is top
                    reward = +1
                elif cls == "miss":
                    label = sorted(read)[0] if read else ""  # skill actually read
                    reward = -1
                elif cls == "none+read":
                    label = sorted(read)[0] if read else ""  # skill actually read
                    reward = -1
                elif cls == "none":
                    label = ""
                    reward = 0

                rows.append({
                    "query": p[3][:80],
                    "sid": sid,
                    "t": p[0],
                    "route_top": cands[:3],
                    "conf": conf,
                    "class": cls,
                    "read_skill": sorted(read)[0] if read else "",
                    "label": label,
                    "reward": reward,
                })
    except FileNotFoundError:
        print(f"ERROR: {log_path} not found.")
        sys.exit(1)

    # Write JSONL (deduplicate by sid+t to avoid duplicate keys per spec L39-L40)
    seen = set()
    out_rows = []
    for r in rows:
        key = (r["sid"], r["t"])
        if key not in seen:
            seen.add(key)
            out_rows.append(r)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Count per class and rewarded rows (spec L103-L104, L110-L111)
    by_cls = {}
    rewarded = 0
    for r in out_rows:
        by_cls[r["class"]] = by_cls.get(r["class"], 0) + 1
        if r["reward"] != 0:
            rewarded += 1

    print(f"Total rows: {len(out_rows)}")
    for k in ("brief_only", "full_read", "miss", "none+read", "none"):
        print(f"  {k}: {by_cls.get(k, 0)}")
    print(f"Rewarded (reward != 0): {rewarded}")

    if rewarded < 300:
        print(f"STOP partial: {rewarded} of 300")


if __name__ == "__main__":
    main()
```

-- localread: trace_inputs.txt, lines 1-123 of 123, 1 part(s), 96s, Strata qwen3.8-flash-next-iq3_xxs on port 8090. Verify cited lines before relying on them.
