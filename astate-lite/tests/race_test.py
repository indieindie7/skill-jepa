"""race_test.py: TESTS.md s8 item 1, two (or more) closers on one .state. Each worker process appends its own LOG line
(locked append) and then regenerates STATE (read LOG -> regenerate -> atomic write). Checks per round: LOG kept every
line, STATE parses (never torn), the surviving STATE holds every worker's claim (the last writer wins, so this one is
informative, not required). Usage: py -3.13 -I race_test.py   -> results/race.json"""
import os, sys, json, subprocess, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import astate_lite as al   # noqa: E402

SEED = 20
WORKER = """
import sys; sys.path.insert(0, %r)
import astate_lite as al
root, i = sys.argv[1], sys.argv[2]
lp = root + '/LOG.md'
al.append_log(lp, al.render_log('2026-11-05', 'event', '[SRC] claim from worker ' + i, 'w' + i + '.py:1', 'confirmed'))
entries, errors = al.read_log(lp)
text, rep = al.regenerate(entries, 'race', '2026-11-05')
al.write_state(root + '/STATE.md', text)
""" % HERE


def _seed(lp):
    al.append_log(lp, "# LOG")
    for k in range(SEED):
        al.append_log(lp, al.render_log("2026-11-01", "event", f"[SRC] seed fact {k}", f"s{k}.py:1", "confirmed"))


def _spawn(d, procs):
    """Starts `procs` closers at once (hidden), returns their stderr texts."""
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    ps = [subprocess.Popen([sys.executable, "-I", "-c", WORKER, d, str(i)], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                           creationflags=flags) for i in range(procs)]
    return [p.communicate()[1].decode(errors="replace") for p in ps]


def _score(d, procs, errs):
    entries, errors = al.read_log(os.path.join(d, "LOG.md"))
    st = al.read_text(os.path.join(d, "STATE.md"))
    parsed = al.parse_state(st)
    claims = {c["text"] for c in parsed["claims"]}
    return {"log_ok": len(entries) == SEED + procs and not errors,
            "state_parses": st.startswith("# STATE") and not parsed["bad_claims"] and len(claims) >= SEED + 1,
            "state_has_all": all(f"claim from worker {i}" in claims for i in range(procs)),
            "worker_errors": sum(1 for e in errs if e.strip()),
            "first_error": next((e.strip().splitlines()[-1][:300] for e in errs if e.strip()), "")}


def one_round(procs):
    with tempfile.TemporaryDirectory() as d:
        _seed(os.path.join(d, "LOG.md"))
        return _score(d, procs, _spawn(d, procs))


def run(procs, rounds):
    rs = [one_round(procs) for _ in range(rounds)]
    return {"procs": procs, "rounds": rounds, **{k: sum(1 for r in rs if r[k]) for k in ("log_ok", "state_parses", "state_has_all")},
            "worker_errors": sum(r["worker_errors"] for r in rs),
            "first_error": next((r["first_error"] for r in rs if r["first_error"]), "")}


if __name__ == "__main__":
    res = {"2x20": run(2, 20), "8x10": run(8, 10)}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "race.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res))
