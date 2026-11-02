"""qstd.py: code-rot meter + regression gate (SlopCodeBench metrics; video-notes tree/442, SELFIMPROVE_PLAN sec 10). Pure stdlib (ast).
Metrics over a set of .py files:
  erosion = share of total fn weight in fns with CC>10, weight = cyclomatic complexity * sqrt(LOC)
  dup     = share of normalized code lines inside a WIN-line window that occurs >= 2x (across the whole set)
  cc10    = count of fns with CC >= 10
  py -3.13 -I qstd.py scan [PATH|REPO ...]                  metrics of files/dirs (working tree)
  py -3.13 -I qstd.py diff REPO A [B]                         metrics at rev A vs rev B (B default WT = working tree) + fns that crossed CC 10
  py -3.13 -I qstd.py gate REPO [--test CMD] [--base REV] [--accept] [--log]
       PASS only if the FULL test suite passes AND erosion/dup rise <= .005 vs last accepted (REPO/.git/qstd_accepted.json, else HEAD)
       AND no fn crossed CC 10 (--allow-cc to waive). Suite: --test, else REPO/.qstd_test (one command line). No suite -> UNTESTED.
       Exit 0 PASS, 1 FAIL, 2 UNTESTED. One verdict line, then JSON. --accept stores B=HEAD as new base on PASS. --log appends the reward row.
  py -3.13 -I qstd.py replay REPO [--n N] [--top K]           metrics per first-parent commit (oldest first) + biggest rises
  py -3.13 -I qstd.py hook                                    PostToolUse (Edit|Write) stdin JSON: edited .py vs its HEAD blob; prints a warning only (opt-in, see skill quality-standard)
Reward rows (--log): data/qstd_rewards.jsonl (untracked, data/* ignored): {ts, repo, base, rev, verdict, reward, d_erosion, d_dup, crossed, tests}."""
import ast, hashlib, json, math, os, re, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
REWARDS = os.path.join(HERE, "data", "qstd_rewards.jsonl")
WIN, CC_HI, THR = 6, 10, .005   # dup window (lines, CPD-style), "complex" fn line, max allowed rise of erosion/dup share
SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", "site-packages", "build", "dist"}
BRANCH = (ast.If, ast.IfExp, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler, ast.Assert, ast.comprehension, ast.match_case)
SCOPE = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


def cc(fn):
    """McCabe: 1 + branches + extra bool operands; nested defs/classes/lambdas are their own scope."""
    c, st = 1, list(ast.iter_child_nodes(fn))
    while st:
        n = st.pop()
        if isinstance(n, SCOPE): continue
        if isinstance(n, BRANCH): c += 1 + (len(n.ifs) if isinstance(n, ast.comprehension) else 0)
        elif isinstance(n, ast.BoolOp): c += len(n.values) - 1
        st.extend(ast.iter_child_nodes(n))
    return c


def fns(src):
    """-> [(qualname, cc, loc)]; loc = nonblank lines of the def minus nested defs. None on syntax error."""
    try: tree = ast.parse(src)
    except (SyntaxError, ValueError): return None
    lines, out = src.splitlines(), []
    def walk(node, pre):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)):
                own = set(range(ch.lineno, ch.end_lineno + 1))
                for sub in ast.walk(ch):
                    if sub is not ch and isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)): own -= set(range(sub.lineno, sub.end_lineno + 1))
                out.append((pre + ch.name, cc(ch), sum(1 for i in own if i <= len(lines) and lines[i - 1].strip())))
                walk(ch, pre + ch.name + ".")
            elif isinstance(ch, ast.ClassDef): walk(ch, pre + ch.name + ".")
            else: walk(ch, pre)
    walk(tree, "")
    return out


def norm_lines(src):
    """Code lines for duplication: stripped, whitespace collapsed; drops blank, comment-only and trivial (<= 4 chars: 'else:', ')', 'try:')."""
    out = []
    for l in src.splitlines():
        s = " ".join(l.split())
        if len(s) > 4 and not s.startswith("#"): out.append(s)
    return out


_CACHE = {}
def blob_stats(src):
    """Per-file stats cached by content hash: fns + window hashes."""
    k = hashlib.sha1(src.encode("utf-8", "replace")).hexdigest()
    if k not in _CACHE:
        nl = norm_lines(src)
        _CACHE[k] = (fns(src), len(nl), [hash("\n".join(nl[i:i + WIN])) for i in range(len(nl) - WIN + 1)])
    return _CACHE[k]


def dup_lines(stats):
    """{path: blob_stats} -> set of (path, normalized line idx) inside a window seen >= 2x anywhere in the set."""
    wins, dupl = {}, set()
    for p, (_, _, hs) in stats.items():
        for i, h in enumerate(hs): wins.setdefault(h, []).append((p, i))
    for p, i in (o for occ in wins.values() if len(occ) > 1 for o in occ): dupl.update((p, j) for j in range(i, i + WIN))
    return dupl


def fn_weights(stats):
    """-> ({path::qualname: cc}, total weight, weight in fns CC>10); weight = cc * sqrt(loc)."""
    fcc, W, Whi = {}, 0.0, 0.0
    for p, (F, _, _) in stats.items():
        for q, c, loc in F or ():
            w = c * math.sqrt(max(loc, 1)); fcc[f"{p}::{q}"] = c; W += w; Whi += w * (c > CC_HI)
    return fcc, W, Whi


def metrics(files):
    """files: {path: source} -> dict of set metrics + per-fn CC map."""
    stats = {p: blob_stats(src) for p, src in files.items()}; nlines = sum(s[1] for s in stats.values())
    fcc, W, Whi = fn_weights(stats); bad, dupl = sum(1 for F, _, _ in stats.values() if F is None), dup_lines(stats)
    return {"files": len(files), "parse_err": bad, "fns": len(fcc), "lines": nlines,
            "erosion": round(Whi / W, 4) if W else 0.0, "dup": round(len(dupl) / nlines, 4) if nlines else 0.0,
            "cc10": sum(1 for c in fcc.values() if c >= CC_HI), "cc_max": max(fcc.values(), default=0), "_fcc": fcc}


def read(p):
    with open(p, encoding="utf-8", errors="replace") as f: return f.read()


def git(repo, *a, inp=None):
    r = subprocess.run(["git", "-C", repo, *a], input=inp, capture_output=True)
    if r.returncode: raise SystemExit(f"git {' '.join(a)}: {r.stderr.decode(errors='replace').strip()}")
    return r.stdout


def keep(path):
    return path.endswith(".py") and not (set(re.split(r"[\\/]", path)[:-1]) & SKIP_DIRS)


def wt_files(repo):
    """Working tree: tracked + untracked-not-ignored .py files that still exist."""
    names = git(repo, "ls-files", "-z", "--cached", "--others", "--exclude-standard").decode().split("\0")
    paths = [n for n in names if keep(n) and os.path.isfile(os.path.join(repo, n))]
    return {n: read(os.path.join(repo, n)) for n in paths}


def files_at(repo, rev):
    """{path: source} of tracked .py files at rev ('WT' = working tree)."""
    if rev == "WT": return wt_files(repo)
    ents = [l.split("\t", 1) for l in git(repo, "ls-tree", "-r", "-z", rev).decode().split("\0") if "\t" in l]
    ents = [(e[0].split()[2], e[1]) for e in ents if e[0].split()[1] == "blob" and keep(e[1])]
    raw, out, pos = git(repo, "cat-file", "--batch", inp="".join(s + "\n" for s, _ in ents).encode()), {}, 0
    for _, path in ents:   # "<sha> blob <size>\n<bytes>\n" per entry, in order
        nl = raw.index(b"\n", pos); size = int(raw[pos:nl].split()[2]); out[path] = raw[nl + 1:nl + 1 + size].decode("utf-8", "replace"); pos = nl + 2 + size
    return out


def compare(a, b):
    """-> deltas + fns that crossed CC 10 (new fn CC>10 or old fn rose from <=10 to >10)."""
    crossed = sorted(f"{k} {a['_fcc'].get(k, '-')}->{c}" for k, c in b["_fcc"].items() if c > CC_HI and a["_fcc"].get(k, 0) <= CC_HI)
    return {"d_erosion": round(b["erosion"] - a["erosion"], 4), "d_dup": round(b["dup"] - a["dup"], 4), "d_cc10": b["cc10"] - a["cc10"], "crossed": crossed}


def pub(m): return {k: v for k, v in m.items() if not k.startswith("_")}


def scan(paths):
    files = {}
    for p in paths or ["."]:
        if os.path.isfile(p): files[p] = read(p); continue
        for root, ds, fs in os.walk(p):
            ds[:] = [d for d in ds if d not in SKIP_DIRS]
            for f in fs:
                if f.endswith(".py"): q = os.path.join(root, f); files[q] = read(q)
    m = metrics(files); top = sorted(m["_fcc"].items(), key=lambda kv: -kv[1])[:10]
    print(json.dumps({**pub(m), "top_cc": [f"{k} {c}" for k, c in top]}, indent=1))


def run_suite(repo, cmd, timeout=1800):
    t = time.time()
    try: r = subprocess.run(cmd, shell=True, cwd=repo, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired: return {"cmd": cmd, "ok": False, "sec": timeout, "tail": "TIMEOUT"}
    tail = (r.stdout + r.stderr).decode(errors="replace").strip().splitlines()[-6:]
    return {"cmd": cmd, "ok": r.returncode == 0, "rc": r.returncode, "sec": round(time.time() - t, 1), "tail": " | ".join(tail)[-600:]}


def judge(d, suite, allow_cc=False):
    """-> (verdict, why, reward). FAIL on broken suite, erosion/dup rise > THR, or a fn crossing CC 10; PASS needs a suite."""
    why = ([f"suite rc={suite.get('rc')}"] if suite and not suite["ok"] else []) + [f"{k[2:]} +{d[k]}" for k in ("d_erosion", "d_dup") if d[k] > THR]
    if d["crossed"] and not allow_cc: why.append(f"{len(d['crossed'])} fn crossed CC{CC_HI}")
    verdict = "FAIL" if why else ("PASS" if suite else "UNTESTED")
    reward = {"PASS": 1.0, "UNTESTED": 0.0, "FAIL": -1.0}[verdict] - 10 * (max(d["d_erosion"], 0) + max(d["d_dup"], 0))
    return verdict, why, round(reward, 4)


def accepted_base(accf):
    try: return json.loads(read(accf))["rev"]
    except (OSError, ValueError, KeyError): return "HEAD"


def log_reward(out):
    os.makedirs(os.path.dirname(REWARDS), exist_ok=True)
    row = {k: out[k] for k in ("repo", "base", "verdict", "reward", "d_erosion", "d_dup")}
    row.update(ts=time.strftime("%Y-%m-%d %H:%M:%S"), rev=out["head"] if out["rev"] == "WT" else out["rev"], crossed=len(out["crossed"]), tests=(out["tests"] or {}).get("ok"))
    with open(REWARDS, "a", encoding="utf-8") as f: f.write(json.dumps(row) + "\n")


def gate(repo, test=None, base=None, accept=False, allow_cc=False, log=False, rev="WT"):
    accf = os.path.join(git(repo, "rev-parse", "--absolute-git-dir").decode().strip(), "qstd_accepted.json"); base = base or accepted_base(accf)
    a, b = metrics(files_at(repo, base)), metrics(files_at(repo, rev)); d = compare(a, b)
    tf = os.path.join(repo, ".qstd_test"); test = test or (read(tf).strip() if os.path.isfile(tf) else None)
    suite = run_suite(repo, test) if test else None
    verdict, why, reward = judge(d, suite, allow_cc); head = git(repo, "rev-parse", "HEAD").decode().strip()
    out = {"verdict": verdict, "why": why, "reward": reward, "repo": os.path.abspath(repo), "base": base, "rev": rev, "head": head, "before": pub(a), "after": pub(b), **d, "tests": suite}
    note = "; ".join(why) or ("suite ok" if suite else "no test suite (set --test or .qstd_test)")
    print(f"qstd {verdict}: {note} | erosion {a['erosion']}->{b['erosion']} dup {a['dup']}->{b['dup']} cc10 {a['cc10']}->{b['cc10']}")
    print(json.dumps(out, indent=1))
    if accept and verdict == "PASS":
        with open(accf, "w") as f: json.dump({"rev": head, "ts": time.strftime("%Y-%m-%d %H:%M"), "metrics": pub(b)}, f)
    if log: log_reward(out)
    return {"PASS": 0, "FAIL": 1, "UNTESTED": 2}[verdict]


def replay(repo, n=0, top=5):
    revs = git(repo, "rev-list", "--first-parent", "--reverse", "HEAD").decode().split()
    rows, prev = [], metrics({})
    for r in revs[-n:] if n else revs:
        m = metrics(files_at(repo, r)); subj = git(repo, "log", "-1", "--format=%s", r).decode().strip()[:70]
        d = compare(prev, m) if rows else {"d_erosion": 0, "d_dup": 0, "d_cc10": 0, "crossed": []}
        rows.append({"rev": r[:7], **pub(m), **{k: v for k, v in d.items() if k != "crossed"}, "crossed": len(d["crossed"]), "subj": subj}); prev = m
    print_replay(rows, top); return rows


def print_replay(rows, top):
    print(f"{'rev':7} {'files':>5} {'fns':>5} {'eros':>6} {'dup':>6} {'cc10':>4} {'dEro':>7} {'dDup':>7} {'x':>2}  subject")
    for r in rows: print(f"{r['rev']:7} {r['files']:5} {r['fns']:5} {r['erosion']:6.3f} {r['dup']:6.3f} {r['cc10']:4} {r['d_erosion']:+7.3f} {r['d_dup']:+7.3f} {r['crossed']:2}  {r['subj']}")
    flag = [r for r in rows if r["d_erosion"] > THR or r["d_dup"] > THR or r["crossed"]]
    print(f"\n{len(flag)}/{len(rows)} commits would FAIL the metric gate (rise > {THR} or fn crossed CC{CC_HI}); worst by d_erosion+d_dup:")
    for r in sorted(rows, key=lambda r: -(r["d_erosion"] + r["d_dup"]))[:top]: print(f"  {r['rev']} dEro {r['d_erosion']:+.3f} dDup {r['d_dup']:+.3f} crossed {r['crossed']}  {r['subj']}")


def head_src(p):
    """HEAD blob of file p, '' when not in git or new."""
    try:
        top = git(os.path.dirname(os.path.abspath(p)), "rev-parse", "--show-toplevel").decode().strip()
        return git(top, "show", "HEAD:" + os.path.relpath(os.path.abspath(p), top).replace("\\", "/")).decode("utf-8", "replace")
    except SystemExit: return ""


def hook():
    """Cheap per-file check after Edit/Write: prints a one-line warning to stderr (exit 2 feeds it back to Claude); silent otherwise."""
    try: p = (json.load(sys.stdin).get("tool_input") or {}).get("file_path", "")
    except ValueError: return 0
    if not p.endswith(".py") or not os.path.isfile(p): return 0
    old = head_src(p); a, b = metrics({p: old} if old else {}), metrics({p: read(p)}); d = compare(a, b)
    msg = [f"fn crossed CC{CC_HI}: {', '.join(x.split('::')[1] for x in d['crossed'][:4])} -> split into helpers"] if d["crossed"] else []
    if d["d_dup"] > 0.02: msg.append(f"dup {a['dup']}->{b['dup']} (copy-paste? reuse/delete instead)")
    if msg: print(f"qstd {os.path.basename(p)}: " + "; ".join(msg), file=sys.stderr); return 2
    return 0


def take(rest, name, default=None):
    """Pop '--name value' from rest."""
    if name not in rest: return default
    i = rest.index(name); v = rest[i + 1]; del rest[i:i + 2]; return v


def take_flag(rest, name):
    if name not in rest: return False
    rest.remove(name); return True


def cmd_diff(rest):
    a, b = metrics(files_at(rest[0], rest[1])), metrics(files_at(rest[0], rest[2] if len(rest) > 2 else "WT"))
    print(json.dumps({"before": pub(a), "after": pub(b), **compare(a, b)}, indent=1)); return 0


def cmd_gate(rest):
    test, base, rev = take(rest, "--test"), take(rest, "--base"), take(rest, "--rev", "WT")
    acc, allow, log = (take_flag(rest, f) for f in ("--accept", "--allow-cc", "--log"))
    return gate(rest[0] if rest else ".", test, base, acc, allow, log, rev)


def cmd_replay(rest):
    n, top = int(take(rest, "--n", 0)), int(take(rest, "--top", 5)); replay(rest[0] if rest else ".", n, top); return 0


CMDS = {"scan": lambda r: scan(r) or 0, "diff": cmd_diff, "gate": cmd_gate, "replay": cmd_replay, "hook": lambda r: hook()}


def main(av):
    if not av or av[0] not in CMDS: print(__doc__); return 0 if not av or av[0] in ("-h", "--help") else 1
    return CMDS[av[0]](av[1:])


if __name__ == "__main__": sys.exit(main(sys.argv[1:]))
