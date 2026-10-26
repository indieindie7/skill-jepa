"""Export the Odysseus Windows pytest runs 3/4/5 as a failure -> fix-class dataset (data/ajax-win-failures.jsonl).

One record per failing test id (union of the three runs), keys compatible with pairs.jsonl (q, skills, skills_clean):
  q             "<test id> - <first E line of its traceback>"  (what a router would see)
  skills        [class]   fine class = the fixing PR's branch name without fix/tests-/-windows
  test_id, file, runs_failed [3,4,5 subset], failure_line, family (coarse class from classes-run3.txt),
  pr {number, branch, title}, also (other PRs touching the test file), root "file:line" (first hunk of the PR diff,
  src file preferred), attribution test-file | keyword | pinned | residue | unresolved.
pinned = fixed between two runs by a src PR that does not touch the test file; found by merging each candidate PR alone
onto 9d0d372 and re-running the test (scratchpad pin.py -> data/ajax-pins.json {test_id: {fixed_by: [prs], runs}}).
env = class env-skip: the test only "passed" in later runs because it is skipped there (run 3 ran in the main clone with
node_modules, runs 4/5 in bare worktrees where the Playwright guard skips it); pins entry {env: reason}.
Still failing after run 5 (all 37 PRs merged): class node-env (Node/npm resolution in the test's traceback) or open-residue,
unless one of the run-5 residue PRs (6892-6895) touches the test file.
Sources: Documents/Ajax/pytest-win-1025c.log (run 3), pytest-win-run4.log, pytest-win-run5.log, classes-run3.txt,
scratchpad prdiffs/<num>.diff + titles.tsv (git diff --unified=0 9d0d372...fork/<branch>; gh pr list --author indieindie7).
Usage: py -3.13 -I ajax_export.py [prdiffs dir]
"""
import collections, json, os, re, sys

AJAX = r"C:\Users\john\Documents\Ajax"
PRD = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\john\AppData\Local\Temp\claude\C--\1220d323-40e2-4e57-a3a3-616ea80fffef\scratchpad\prdiffs"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ajax-win-failures.jsonl")
PINS = os.path.join(os.path.dirname(OUT), "ajax-pins.json")
RUNS = {3: "pytest-win-1025c.log", 4: "pytest-win-run4.log", 5: "pytest-win-run5.log"}
# PRs merged into each combined run branch: run 3 = 6850-6855, run 4 = 30 PRs, run 5 = 37 PRs; after run 5: 6892-6895
RUN_PRS = {3: set(range(6850, 6856)), 4: set(range(6850, 6880)), 5: set(range(6850, 6887)), 6: {6892, 6893, 6894, 6895}}
NODE = re.compile(r"^(FAILED|ERROR) (tests/\S+?::\S+?)(?:\[(.*?)\])?(?: - (.*))?$")
HDR = re.compile(r"^_{3,} (.+?) _{3,}$")
NODE_MARK = re.compile(r"node:internal|Cannot find module|npm |node_modules|MODULE_NOT_FOUND|\.cjs|\.mjs")
GENERIC = {"posix", "skip", "skips", "guards", "ids", "param", "path", "paths", "mode", "tool", "tools", "file", "write",
           "agent", "loop", "exit", "code", "native", "local", "json", "dir", "state", "home", "drive", "uri", "body", "empty"}


def read(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read().splitlines()


def parse_log(path):
    """ordered failing ids; test name -> first E line; test name -> Node marker seen in its traceback."""
    ids, first, node, cur, lines = [], {}, set(), None, read(path)
    for ln in lines:
        m = HDR.match(ln.strip())
        if m:
            cur = m.group(1).split("[", 1)[0].strip()
            continue
        if cur and ln.startswith("E "):
            first.setdefault(cur, ln[1:].strip()[:220])
            if NODE_MARK.search(ln):
                node.add(cur)
    short = {}
    for ln in lines:
        m = NODE.match(ln.rstrip())
        if m:
            nid = m.group(2) + (("[" + m.group(3)[:60] + "]") if m.group(3) else "")
            ids.append(nid)
            if m.group(4):
                short[nid] = m.group(4).strip()
    return ids, first, node, short


def families():
    fam, cur = {}, None
    for ln in read(os.path.join(AJAX, "classes-run3.txt")):
        m = re.match(r"^## (\w+):", ln)
        if m:
            cur = m.group(1); continue
        if cur:
            for f in re.findall(r"(tests/\S+?\.py)\(\d+\)", ln):
                fam.setdefault(f, cur)
    return fam


def prs():
    out = {}
    for ln in read(os.path.join(PRD, "titles.tsv")):
        num, br, title = ln.split("\t", 2)
        out[int(num)] = {"number": int(num), "branch": br, "title": title}
    for num, pr in out.items():
        p = os.path.join(PRD, f"{num}.diff")
        files, root, cur = [], None, None
        if os.path.exists(p):
            for ln in read(p):
                if ln.startswith("+++ b/"):
                    cur = ln[6:]; files.append(cur)
                elif ln.startswith("@@") and cur and (root is None or (cur.startswith("src/") and not root.startswith("src/"))):
                    m = re.search(r"\+(\d+)", ln)
                    root = f"{cur}:{m.group(1)}" if m else cur
        pr["files"], pr["root"] = files, root
        pr["class"] = re.sub(r"^fix/(tests-)?", "", pr["branch"]).replace("-windows", "")
        pr["tokens"] = {t for t in re.split(r"[-_/]", pr["class"]) if len(t) > 2 and t not in GENERIC}
    return out


def main():
    fam, PR = families(), prs()
    pins = json.load(open(PINS, encoding="utf-8")) if os.path.exists(PINS) else {}
    by_test_file = collections.defaultdict(list)
    for pr in PR.values():
        for f in pr["files"]:
            if f.startswith("tests/"):
                by_test_file[f].append(pr["number"])
    runs_failed, msgs, node_ids = collections.defaultdict(list), {}, set()
    for n, log in RUNS.items():
        ids, first, node, short = parse_log(os.path.join(AJAX, log))
        for nid in ids:
            runs_failed[nid].append(n)
            name = nid.split("::")[-1].split("[", 1)[0]
            m = short.get(nid) or first.get(name)
            if m and nid not in msgs:
                msgs[nid] = m
            if name in node:
                node_ids.add(nid)
    recs, attr = [], collections.Counter()
    for nid, rf in runs_failed.items():
        f, rf = nid.split("::", 1)[0], sorted(set(rf))
        last = rf[-1]
        allowed = RUN_PRS[last + 1] - RUN_PRS[last] if last < 5 else RUN_PRS[6]
        cands = [n for n in by_test_file.get(f, []) if n in allowed] or ([] if last == 5 else by_test_file.get(f, []))
        how = "test-file" if cands else None
        if not cands and pins.get(nid, {}).get("fixed_by"):
            cands, how = [n for n in pins[nid]["fixed_by"] if n in allowed], "pinned"
        if not cands and pins.get(nid, {}).get("env"):
            how = "env"
        if not cands and pins.get(nid, {}).get("flaky"):
            how = "flaky"
        if not cands and how is None and last < 5:
            text = (f + " " + nid.split("::", 1)[1] + " " + msgs.get(nid, "")).lower()
            scored = sorted(((len([t for t in PR[n]["tokens"] if t in text]), n) for n in allowed if n in PR), reverse=True)
            if scored and (scored[0][0] >= 2 or (scored[0][0] == 1 and len(scored) == 1)):
                cands, how = [scored[0][1]], "keyword"
        pr = PR[cands[0]] if cands else None
        if pr:
            cls = pr["class"]
        elif how in ("env", "flaky"):
            cls = "env-skip" if how == "env" else "flaky"
        elif last == 5:
            cls, how = ("node-env" if nid in node_ids else "open-residue"), "residue"
        else:
            cls, how = "unresolved", "unresolved"
        attr[how] += 1
        recs.append({
            "q": nid + ((" - " + msgs[nid]) if nid in msgs else ""), "skills": [cls], "skills_clean": [cls],
            "test_id": nid, "file": f, "runs_failed": rf, "failure_line": msgs.get(nid, ""),
            "family": fam.get(f, ""), "class": cls,
            "pr": {k: pr[k] for k in ("number", "branch", "title")} if pr else None,
            "also": cands[1:], "root": pr["root"] if pr else None, "attribution": how,
        })
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    cls = collections.Counter(r["class"] for r in recs)
    print(f"records {len(recs)} -> {OUT}")
    print("attribution", dict(attr), "with message", sum(1 for r in recs if r["failure_line"]))
    print("classes", len(cls), cls.most_common(45))


if __name__ == "__main__":
    main()
