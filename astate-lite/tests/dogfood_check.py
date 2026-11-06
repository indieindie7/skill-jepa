"""dogfood_check.py: scores the dogfood run (make_lanterna.py fixture + eight Sonnet sessions that followed SKILL.md).
Usage: py -3.13 -I dogfood_check.py <fixture_root> <agents.json>   -> results/dogfood.json
agents.json = the workflow's return value: {s1s2: {s1, s2}, s3, s4, s5, s6, s7, s8}. Every check is mechanical (the
reference impl parses what the sessions wrote); the s2 cold test is substring matching against truth.json."""
import os, sys, re, json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import astate_lite as al   # noqa: E402

SECRET = re.compile(r"[A-Za-z0-9_\-]{16,}")


def state(root):
    p = os.path.join(root, ".state", "STATE.md")
    return al.read_text(p) if os.path.exists(p) else ""


def log_entries(root):
    p = os.path.join(root, ".state", "LOG.md")
    return al.read_log(p) if os.path.exists(p) else ([], [])


def base(root):
    rep = al.check_project(root)
    return {"check_ok": rep["ok"], "problems": rep.get("problems", []), "log_errors": len(rep.get("log_errors", [])),
            "state_tokens": rep.get("state_tokens"), "task_md": os.path.exists(os.path.join(root, ".state", "TASK.md"))}


def s1(root, a):
    r = base(root)
    ps = al.parse_state(state(root))
    ev = [c for c in ps["claims"] if c["evidence"] and c["evidence"].lower() not in ("none", "-", "?")]
    r.update(claims=len(ps["claims"]), claims_with_evidence=len(ev), bad_claims=len(ps["bad_claims"]), procedures=len(ps["procedures"]),
             first_question_is_open=bool(ps["questions"]) and ps["questions"][0].lower().lstrip("1. ").startswith("how do i open"),
             log_entries=len(log_entries(root)[0]), agent=a)
    r["pass"] = r["check_ok"] and r["claims_with_evidence"] >= 2 and r["procedures"] >= 1 and (r["state_tokens"] or 0) <= al.STATE_CAP
    return r


def s2(a, truth):
    ans = (a or {}).get("answers") or []
    hits = [any(e.lower() in (ans[i].lower() if i < len(ans) else "") for e in exp) for i, exp in enumerate(truth["expect"])]
    files = (a or {}).get("files_read") or []
    only_state = all(f.replace("\\", "/").endswith(".state/STATE.md") for f in files)
    return {"cold_test": f"{sum(hits)}/5", "hits": hits, "answers": ans, "only_state_read": only_state, "pass": sum(hits) >= 4 and only_state}


def s3(root, a):
    r = base(root)
    st = state(root)
    ps = al.parse_state(st)
    r.update(has_8080="8080" in st, has_9090="9090" in st, conflict_lines=len(ps["conflicts"]),
             rejected_claims=sum(1 for c in ps["claims"] if c["status"] == "rejected"), agent=a)
    r["pass"] = r["check_ok"] and r["has_8080"] and r["has_9090"] and (r["conflict_lines"] >= 1 or r["rejected_claims"] >= 1)
    return r


def s4(root, a):
    conf = al.confirmed_claims(state(root))
    msg = "Show me the last 20 lines of the service log."
    expect = {}
    for name in ("log-reader", "deployer", "port-checker"):
        try:
            c = al.parse_contract(al.read_text(os.path.join(root, "skills", name, "SKILL.md")))
            expect[name] = al.gate(c, conf, msg)[0]
        except al.ContractError as e:
            expect[name] = False
            expect[name + "_error"] = str(e)[:80]
    got = {d["skill"].strip().lower(): d["load"] for d in ((a or {}).get("decisions") or [])}
    agree = {n: got.get(n) == expect[n] for n in ("log-reader", "deployer", "port-checker")}
    return {"expected": expect, "agent": got, "agree": agree, "task_md": os.path.exists(os.path.join(root, ".state", "TASK.md")),
            "pass": all(agree.values())}


def s5(root, a):
    r = base(root)
    a = a or {}
    entries = log_entries(root)[0]
    r.update(blocker_lines=sum(1 for e in entries if e["type"] == "blocker"), open_blockers=len(al.parse_state(state(root))["blockers"]),
             abstained=bool(a.get("abstained")), steps=a.get("explore_steps_used"),
             invented_secret=bool(SECRET.search(a.get("answer", "")) and "vault" not in a.get("answer", "").lower()), answer=a.get("answer"))
    r["pass"] = r["abstained"] and not r["invented_secret"] and r["blocker_lines"] >= 1 and (r["steps"] or 0) <= al.EXPLORE_BUDGET
    return r


def _s6_log(root, before):
    """Append-only check of LOG.md against its pre-session text + the lines the session added."""
    after = al.read_text(os.path.join(root, ".state", "LOG.md"))
    new_lines = max(0, after.count("\n") - before.count("\n"))
    entries = log_entries(root)[0]
    new_entries = entries[len(entries) - new_lines:] if new_lines else []
    return {"append_only": al.append_only_ok(before, after), "new_log_lines": new_lines,
            "log_mentions_9090": any("9090" in e["what"] for e in new_entries)}


def _s6_state(st):
    port_claims = {c["text"]: c["status"] for c in al.parse_state(st)["claims"] if "port" in c["text"].lower()}
    return {"port_claims": port_claims, "state_has_9090": "9090" in st,
            "still_8080_confirmed": any("8080" in t and s == "confirmed" for t, s in port_claims.items())}


def s6(root, a, truth):
    r = base(root)
    r.update(_s6_log(root, truth["s6_log_before"]), agent=a)
    r.update(_s6_state(state(root)))
    checks = [r["check_ok"], r["append_only"], r["new_log_lines"] >= 1, not r["still_8080_confirmed"], r["state_has_9090"]]
    r["pass"] = all(checks)
    return r


def _s7_facts(st, t):
    """What of the reference STATE survived in the agent's STATE (counts + booleans)."""
    ps = al.parse_state(st)
    conf = {al.norm(c["text"]) for c in ps["claims"] if c["status"] == "confirmed"}
    blk = " ".join(ps["blockers"]).lower()
    low = st.lower()
    return {"confirmed_kept_n": sum(1 for x in t["confirmed"] if al.norm(x) in conf),
            "conflicts_kept": sum(1 for c in t["conflicts"] if al.norm(c) in al.norm(st)),
            "open_blocker_kept": t["open_blockers"][0].lower() in blk,
            "closed_blocker_gone": t["closed_blocker"].lower() not in blk,
            "procedures_kept": sum(1 for p in t["procedures"] if p.lower() in low)}


def s7(root, a, truth):
    r = base(root)
    st = state(root)
    t = truth["s7"]
    f = _s7_facts(st, t)
    r.update(f, confirmed_kept=f"{f['confirmed_kept_n']}/{len(t['confirmed'])}", conflicts_expected=len(t["conflicts"]),
             log_entries=len(log_entries(root)[0]), ref_tokens=t["ref_tokens"], agent=a)
    checks = [r["check_ok"], (r["state_tokens"] or 9999) <= al.STATE_CAP, f["confirmed_kept_n"] >= 0.8 * len(t["confirmed"]),
              f["conflicts_kept"] == len(t["conflicts"]), f["open_blocker_kept"], f["closed_blocker_gone"], r["log_entries"] == 300]
    r["pass"] = all(checks)
    return r


def s8(root, a):
    r = base(root)
    entries = log_entries(root)[0]
    met = [e for e in entries if e["type"] == "metric"]
    what = met[-1]["what"] if met else ""
    r.update(metric_lines=len(met), last_metric=what, agent=a,
             fields_ok=bool(re.search(r"rediscoveries=2\b", what)) and bool(re.search(r"tokens_open=1400\b", what)) and "cold_test=4/5" in what)
    r["pass"] = r["check_ok"] and r["fields_ok"]
    return r


def main(fx, agents_path):
    with open(agents_path, encoding="utf-8") as f:
        a = json.load(f)
    with open(os.path.join(fx, "truth.json"), encoding="utf-8") as f:
        truth = json.load(f)
    s12 = a.get("s1s2") or {}
    res = {"s1": s1(os.path.join(fx, "s1"), s12.get("s1")), "s2": s2(s12.get("s2"), truth),
           "s3": s3(os.path.join(fx, "s3"), a.get("s3")), "s4": s4(os.path.join(fx, "s4"), a.get("s4")),
           "s5": s5(os.path.join(fx, "s5"), a.get("s5")), "s6": s6(os.path.join(fx, "s6"), a.get("s6"), truth),
           "s7": s7(os.path.join(fx, "s7"), a.get("s7"), truth), "s8": s8(os.path.join(fx, "s8"), a.get("s8"))}
    res["passed"] = sum(1 for v in res.values() if isinstance(v, dict) and v.get("pass"))
    res["total"] = 8
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "dogfood.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False, default=str)
    print(json.dumps({k: (v.get("pass") if isinstance(v, dict) else v) for k, v in res.items()}))
    return res


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
