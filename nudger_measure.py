"""nudger_measure.py (chat "jepa-context work chat", 10-27): read-only measurements for notes/context-nudger-PLAN.md
A3 exploration slot: live explore rows + retro upper bound (reads outside top-k, least-routed-skill hit rate).
A4 refute templates: live refuted rows + retro template-vs-class agreement (templates are a function of conf+class?).
A6 baseline: daily top-1 entropy + novel-choice rate.
Reads ops/route.log + ~/.claude/skills/usage.jsonl only; writes the markdown given by --out. Join logic copied from trace_pairs.py (600 s window)."""
import os, sys, json, time, math, argparse, collections

HERE = os.path.dirname(os.path.abspath(__file__))
ROUTE_LOG = os.path.join(HERE, "..", "ops", "route.log")
USAGE = os.path.expanduser("~/.claude/skills/usage.jsonl")
SKILLS = os.path.expanduser("~/.claude/skills")
WINDOW = 600

def ts(s, fmt):
    try: return time.mktime(time.strptime(s, fmt))
    except ValueError: return None

def template(conf, top):
    if conf in ("high", "medium"): return "reads other skill than " + top
    if conf == "low": return "no skill read within 2 turns"
    return "reads any skill within 2 turns"

def refuted(ref, read):
    if ref.startswith("reads other skill than "): return bool(set(read) - {ref[len("reads other skill than "):]})
    if ref == "no skill read within 2 turns": return not read
    if ref == "reads any skill within 2 turns": return bool(read)
    return None

def entropy(c):
    n = sum(c.values()); return -sum(v / n * math.log2(v / n) for v in c.values()) if n else 0.0

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(HERE, "notes", "nudger-measure.md")); a = ap.parse_args()
    use = []
    for l in open(USAGE, encoding="utf-8"):
        try: j = json.loads(l); t = ts(j["t"], "%Y-%m-%d %H:%M")
        except (ValueError, KeyError): continue
        if t: use.append((t, j["skill"], j.get("sid", "")))
    skills = sorted(d for d in os.listdir(SKILLS) if os.path.exists(os.path.join(SKILLS, d, "SKILL.md")))
    rows, seen = [], set()
    for l in open(ROUTE_LOG, encoding="utf-8"):
        p = l.rstrip("\n").split("\t")
        if len(p) < 4: continue
        t = ts(p[0], "%Y-%m-%d %H:%M:%S")
        if not t or p[3].startswith("<") or len(p[3]) < 8: continue
        conf = p[1]; cands = [c.split(":")[0] for c in p[2].split(",") if c]
        sid = p[4] if len(p) > 4 else ""
        ex = p[5][8:] if len(p) > 5 and p[5].startswith("explore=") else ""
        ref = p[6][7:] if len(p) > 6 and p[6].startswith("refute=") else ""
        if (sid, p[0], p[3]) in seen: continue
        seen.add((sid, p[0], p[3]))
        read = sorted({sk for ut, sk, us in use if 0 <= ut - t <= WINDOW and (not sid or not us or us == sid) and len(sk) > 2})
        top = cands[0] if cands else ""
        if conf == "none": cls = "none+read" if read else "none"
        elif not read: cls = "brief_only"
        elif top in read: cls = "full_read"
        else: cls = "miss"
        rows.append(dict(t=t, day=p[0][:10], sid=sid, conf=conf, cands=cands, top=top, read=read, cls=cls, explore=ex, refute=ref))
    out = [f"# nudger measurements {time.strftime('%Y-%m-%d %H:%M')} (nudger_measure.py, read-only over route.log {len(rows)} rows, usage.jsonl {len(use)} reads, {len(skills)} skills)", ""]
    # A4
    live = [r for r in rows if r["refute"]]
    out += ["## A4 refute templates", f"- live rows with col 7: {len(live)}; refuted: {sum(1 for r in live if refuted(r['refute'], r['read']))}"]
    retro = [(r, refuted(template(r["conf"], r["top"]), r["read"])) for r in rows]
    tab = collections.Counter((r["conf"], r["cls"], rf) for r, rf in retro)
    out.append("- retro (template applied to every historical row), conf x class -> refuted count:")
    out.append("  | conf | class | refuted | n |"); out.append("  |---|---|---|---|")
    for (cf, cl, rf), n in sorted(tab.items()): out.append(f"  | {cf} | {cl} | {rf} | {n} |")
    # is refuted a function of (conf, class)? count (conf,class) cells with both True and False
    cells = collections.defaultdict(set)
    for (cf, cl, rf), n in tab.items(): cells[(cf, cl)].add(rf)
    mixed = [k for k, v in cells.items() if len(v) > 1]
    multi = sum(1 for r in rows if r["cls"] == "full_read" and len(r["read"]) > 1)
    out.append(f"- cells where refuted is NOT determined by (conf,class): {mixed if mixed else 'none'}; full_read rows with >1 skill read (only place a template could add info): {multi}/{sum(1 for r in rows if r['cls']=='full_read')}")
    hm = [(r, rf) for r, rf in retro if r["conf"] in ("high", "medium")]
    agree = sum(1 for r, rf in hm if rf == (r["cls"] == "miss")) / max(1, len(hm))
    out.append(f"- high/medium rows: refuted == (class==miss) on {agree:.3f} of {len(hm)} rows")
    # A3
    ex_rows = [r for r in rows if r["explore"]]
    out += ["", "## A3 exploration slot", f"- live explore rows: {len(ex_rows)}; explore_hit: {sum(1 for r in ex_rows if r['explore'] in r['read'])}"]
    with_read = [r for r in rows if r["read"]]
    for k in (1, 3, 5):
        outside = sum(1 for r in with_read if any(s not in r["cands"][:k] for s in r["read"]))
        out.append(f"- rows with a read where some read skill is outside top-{k}: {outside}/{len(with_read)} = {outside/max(1,len(with_read)):.3f}")
    # least-routed-14d policy, retro: would the least-routed skill have been read?
    hits = shown = 0; hist = collections.deque()
    for r in rows:
        while hist and r["t"] - hist[0][0] > 14 * 86400: hist.popleft()
        cnt = collections.Counter(s for _, s in hist)
        lr = min(skills, key=lambda s: (cnt.get(s, 0), s)) if skills else ""
        shown += 1; hits += lr in r["read"]
        hist.append((r["t"], r["top"]))
    out.append(f"- retro least-routed-14d candidate read in window: {hits}/{shown} = {hits/max(1,shown):.4f}")
    rand_hits = 0
    for r in rows:
        if r["read"]: rand_hits += len(r["read"]) / max(1, len(skills))
    out.append(f"- expected hit rate of a uniform-random explore candidate: {rand_hits/max(1,len(rows)):.4f}")
    # usual-suspects policies (causal, 14 d): candidate = skill most often read OUTSIDE top-3 recently, not in this row's top-3
    META = {"ops-manager", "local-gpu-ai"}
    def suspects(key, skip=frozenset()):
        hist = collections.deque(); hits = shown = 0; picks = collections.Counter()
        for r in rows:
            while hist and r["t"] - hist[0][0] > 14 * 86400: hist.popleft()
            cnt = collections.Counter(s for _, k, s in hist if key(r) == k)
            cand = next((s for s, _ in cnt.most_common() if s not in r["cands"][:3] and s not in skip), "")
            if cand:
                shown += 1; hits += cand in r["read"]; picks[cand] += 1
            for s in r["read"]:
                if s not in r["cands"][:3]: hist.append((r["t"], key(r), s))
        return hits, shown, picks.most_common(3)
    for name, key, skip in (("global", lambda r: 0, frozenset()), ("per-chat (sid)", lambda r: r["sid"], frozenset()),
                            ("global, no meta skills", lambda r: 0, META), ("per-chat, no meta skills", lambda r: r["sid"], META)):
        h, n, top = suspects(key, skip)
        out.append(f"- usual-suspects {name}: candidate read in window {h}/{n} = {h/max(1,n):.4f}; most picked: " + ", ".join(f"{s} {c}" for s, c in top))
    out.append("- caveat: a usual-suspect hit is a systematic router miss (exploitation), not exploration; meta skills (ops-manager, local-gpu-ai) are read by every chat regardless of route")
    # reads not in top-3 ranked by frequency = what exploration must find
    miss_sk = collections.Counter(s for r in with_read for s in r["read"] if s not in r["cands"][:3])
    out.append("- skills read but outside top-3 (top 10): " + ", ".join(f"{s} {n}" for s, n in miss_sk.most_common(10)))
    # A6 baseline
    out += ["", "## A6 autonomy baseline (no nudger)"]
    byday = collections.defaultdict(collections.Counter)
    for r in rows: byday[r["day"]][r["top"]] += 1
    days = sorted(byday)[-14:]
    out.append("  | day | routes | distinct top-1 | entropy bits |"); out.append("  |---|---|---|---|")
    for d in days: out.append(f"  | {d} | {sum(byday[d].values())} | {len(byday[d])} | {entropy(byday[d]):.2f} |")
    seen_sk, novel, tot = set(), 0, 0
    for r in rows:
        tot += 1; novel += r["top"] not in seen_sk; seen_sk.add(r["top"])
    out.append(f"- novel top-1 (never routed before) over all rows: {novel}/{tot}; skills ever top-1: {len(seen_sk)}/{len(skills)}")
    allc = collections.Counter(r["top"] for r in rows)
    out.append("- top-1 share, top 5: " + ", ".join(f"{s} {n/len(rows):.2f}" for s, n in allc.most_common(5)))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    open(a.out, "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
    print("\n".join(out))

if __name__ == "__main__": main()
