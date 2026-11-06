"""Unit + stress tests for the astate-lite reference implementation (unittest; no pytest on this PC).
Run: py -3.13 -I test_astate_lite.py            -> results/unit.json (timings + stress numbers)."""
import unittest, os, sys, time, json, tempfile, random, string, subprocess
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import astate_lite as al

D = "2026-11-05"
EM = al.EM
STRESS = {}


def L(typ, what, ev="README.md:3", st="confirmed", date=D):
    e, err = al.parse_log(al.render_log(date, typ, what, ev, st), strict=False)
    assert e, err
    return e


def contract(name="demo", requires=("port is 8080",), reads=(), produces=(), effects=(), conflicts=(), under_metadata=True, drop_key=None):
    ind = "  " if under_metadata else ""
    out = ["---", f"name: {name}", "description: test contract"]
    if under_metadata:
        out.append("metadata:")
    for k, vals in (("requires", requires), ("reads", reads), ("produces", produces), ("effects", effects), ("conflicts", conflicts)):
        if k == drop_key:
            continue
        out.append(f"{ind}{k}:")
        out += [f"{ind}  - {v}" for v in vals]
    out += ["---", "body"]
    return "\n".join(out)


class Grammar(unittest.TestCase):
    def test_claim_roundtrip(self):
        c = {"tag": "SRC", "text": "opsd listens on 8765", "status": "confirmed", "evidence": "opsd.py:12", "since": D}
        self.assertEqual(al.parse_claim(al.render_claim(c)), c)

    def test_claim_needs_em_dash(self):
        self.assertIsNone(al.parse_claim(f"- [SRC] x - status: confirmed - evidence: y - since: {D}"))

    def test_claim_needs_tag_and_status(self):
        self.assertIsNone(al.parse_claim(f"- x {EM} status: confirmed {EM} evidence: y {EM} since: {D}"))
        self.assertIsNone(al.parse_claim(f"- [SRC] x {EM} status: maybe {EM} evidence: y {EM} since: {D}"))

    def test_claim_o_slash_tag(self):
        self.assertEqual(al.parse_claim(f"- [ø] x {EM} status: unknown {EM} evidence: none {EM} since: {D}")["tag"], "ø")

    def test_conflict_line(self):
        self.assertTrue(al.CONFLICT_RE.match(f"conflict: port 8080 vs port 9090 {EM} README.md:3, config.toml:2 {EM} ask owner"))

    def test_metric(self):
        e = L("metric", "rediscoveries=2 tokens_open=1800 cold_test=4/5")
        self.assertEqual(e["metric"], {"rediscoveries": 2, "tokens_open": 1800, "cold_test": 4})
        self.assertIsNone(al.parse_log(al.render_log(D, "metric", "rediscoveries=2 cold_test=4/5", "w", "confirmed"))[0])
        self.assertIsNone(al.parse_log(al.render_log(D, "metric", "rediscoveries=2 tokens_open=1 cold_test=6/5", "w", "confirmed"))[0])

    def test_log_fields(self):
        e = L("event", "found the port")
        self.assertEqual((e["date"], e["type"], e["what"], e["evidence"], e["status"]), (D, "event", "found the port", "README.md:3", "confirmed"))

    def test_log_pipe_inside_what(self):
        self.assertEqual(L("event", "cmd: a | b").get("what"), "cmd: a | b")

    def test_log_rejects(self):
        for bad in (f"{D} | event | x | status: confirmed", f"05-11-2026 | event | x | evidence: e | status: confirmed",
                    f"{D} | note | x | evidence: e | status: confirmed", f"{D} | event | x | evidence: e | status: maybe",
                    f"{D} | event | x | e | status: confirmed"):
            self.assertIsNone(al.parse_log(bad)[0], bad)

    def test_log_accepts_rejected(self):                    # GAP-1 fixed: SKILL.md:38 lists rejected
        line = al.render_log(D, "decision", "x", "e", "rejected")
        self.assertIsNotNone(al.parse_log(line, strict=True)[0])
        self.assertIsNotNone(al.parse_log(line, strict=False)[0])

    def test_log_skips_header_and_blank(self):
        for s in ("# LOG — p", "(append-only; never edit old lines)", "", "   \n"):
            self.assertEqual(al.parse_log(s)[1], "skip")

    def test_read_log_bom_crlf(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "LOG.md")
            with open(p, "wb") as f:
                f.write(b"\xef\xbb\xbf# LOG\r\n" + al.render_log(D, "event", "a", "e", "confirmed").encode() + b"\r\n" + b"garbage\r\n")
            entries, errors = al.read_log(p)
            self.assertEqual(len(entries), 1)
            self.assertEqual(len(errors), 1)


class Regen(unittest.TestCase):
    def test_promote_demote_keeps_first_date(self):
        es = [L("event", "[SYN] port is 8080", st="hypothesis", date="2026-11-01"), L("event", "[SRC] port is 8080", st="confirmed", date="2026-11-02"),
              L("event", "[SRC] port is 8080", st="rejected", date="2026-11-03")]
        p = al.derive(es)
        c = list(p["claims"].values())[0]
        self.assertEqual((c["status"], c["since"], c["last"], c["tag"]), ("rejected", "2026-11-01", "2026-11-03", "SRC"))

    def test_conflict_both_sides_kept(self):
        es = [L("event", "[SRC] port is 8080", ev="README.md:3"), L("event", "[SRC] port is 9090", ev="config.toml:2"),
              L("decision", f"conflict: port is 8080 vs port is 9090 {EM} README.md:3, config.toml:2 {EM} ask owner")]
        text, rep = al.regenerate(es, "p", D)
        ps = al.parse_state(text)
        self.assertEqual(len(ps["claims"]), 2)
        self.assertEqual(len(ps["conflicts"]), 1)
        self.assertTrue(al.CONFLICT_RE.match(ps["conflicts"][0]))

    def test_drop_removes(self):
        es = [L("event", "[SRC] old fact"), L("decision", "drop: old fact")]
        self.assertEqual(al.derive(es)["claims"], {})

    def test_blocker_open_close(self):
        es = [L("blocker", "need the API key", st="unknown", ev="owner: john")]
        self.assertEqual(len(al.derive(es)["blockers"]), 1)
        es.append(L("blocker", "need the API key", st="confirmed"))
        self.assertEqual(len(al.derive(es)["blockers"]), 0)

    def test_procedure_latest_wins(self):
        es = [L("event", "procedure: Open server: py app.py"), L("event", "procedure: Open server: py -3.13 app.py --port 8080")]
        self.assertEqual(al.derive(es)["procedures"]["Open server"], "py -3.13 app.py --port 8080")

    def test_questions_max_five(self):
        es = [L("event", f"question: q{i}") for i in range(9)]
        self.assertEqual(len(al.derive(es)["questions"]), 5)

    def test_receipt(self):
        es = [L("event", f"receipt: README.md {EM} abc123 {EM} archived no")]
        self.assertIn("README.md", al.derive(es)["receipts"])

    def test_untagged_fact_is_unknown_tag(self):
        self.assertEqual(list(al.derive([L("event", "plain fact")])["claims"].values())[0]["tag"], "SRC")        # evidence README.md:3
        self.assertEqual(list(al.derive([L("event", "plain fact", ev="me")])["claims"].values())[0]["tag"], "ø")

    def test_deterministic(self):
        es = [L("event", f"[SRC] fact {i}") for i in range(50)]
        self.assertEqual(al.regenerate(es, "p", D)[0], al.regenerate(es, "p", D)[0])

    def test_state_parse_roundtrip(self):
        es = [L("event", f"[SRC] fact {i}", st=s) for i, s in enumerate(("confirmed", "hypothesis", "unknown"))]
        text, _ = al.regenerate(es, "p", D)
        ps = al.parse_state(text)
        self.assertEqual([c["status"] for c in ps["claims"]], ["confirmed", "hypothesis", "unknown"])
        self.assertEqual(ps["bad_claims"], [])

    def test_cap_drop_order(self):
        es = []
        for i in range(400):
            es.append(L("event", f"[SRC] fact number {i} with some padding words to use tokens", st=("rejected", "unknown", "hypothesis", "confirmed")[i % 4], date=f"2026-10-{1 + i % 28:02d}"))
        text, rep = al.regenerate(es, "p", D)
        self.assertLessEqual(rep["tokens"], al.STATE_CAP)
        self.assertFalse(rep["over_cap"])
        kinds = [d.split(":")[0] for d in rep["dropped"]]
        self.assertEqual(kinds, sorted(kinds, key=al.DROP_ORDER.index))
        STRESS["cap_400_claims"] = {"kept": rep["claims"], "dropped": len(rep["dropped"]), "tokens": rep["tokens"], "confirmed_lost": kinds.count("confirmed")}

    def test_cap_keeps_conflicts_procedures(self):
        es = [L("event", f"[SRC] fact {i} padding padding padding padding", st="confirmed") for i in range(500)]
        es += [L("decision", f"conflict: a vs b {EM} src {EM} next"), L("event", "procedure: Open X: run it"), L("event", "question: How do I open X?")]
        text, rep = al.regenerate(es, "p", D)
        ps = al.parse_state(text)
        self.assertEqual((len(ps["conflicts"]), len(ps["procedures"]), len(ps["questions"])), (1, 1, 1))
        self.assertLessEqual(rep["tokens"], al.STATE_CAP)

    def test_single_oversize_procedure_cannot_meet_cap(self):
        es = [L("event", "procedure: Open X: " + "step " * 3000)]
        text, rep = al.regenerate(es, "p", D)
        self.assertTrue(rep["over_cap"])      # reported, not silently truncated


class Gate(unittest.TestCase):
    def test_met_by_claim(self):
        c = al.parse_contract(contract(requires=("port is 8080",)))
        self.assertEqual(al.gate(c, ["port is 8080"], "do the thing"), (True, []))

    def test_met_by_message(self):
        c = al.parse_contract(contract(requires=("port is 8080",)))
        self.assertEqual(al.gate(c, [], "the Port is 8080, go")[0], True)

    def test_unmet_and_partial(self):
        c = al.parse_contract(contract(requires=("port is 8080", "db is sqlite")))
        self.assertEqual(al.gate(c, ["port is 8080"], ""), (False, ["db is sqlite"]))
        self.assertEqual(al.gate(c, [], ""), (False, ["port is 8080", "db is sqlite"]))

    def test_only_confirmed_counts(self):
        c = al.parse_contract(contract(requires=("port is 8080",)))
        state = al.regenerate([L("event", "[SRC] port is 8080", st="hypothesis")], "p", D)[0]
        self.assertFalse(al.gate(c, al.confirmed_claims(state), "")[0])
        state = al.regenerate([L("event", "[SRC] port is 8080", st="confirmed")], "p", D)[0]
        self.assertTrue(al.gate(c, al.confirmed_claims(state), "")[0])

    def test_empty_requires_blocked(self):
        c = al.parse_contract(contract(requires=()))
        self.assertFalse(al.gate(c, [], "anything")[0])       # GAP-8 fixed

    def test_contract_rejections(self):
        for bad, why in ((contract(under_metadata=False), "top-level keys"), (contract(drop_key="effects"), "missing key"),
                         ("name: x\nmetadata:\n  requires:\n", "no front matter"), ("---\nname: x\n", "unclosed")):
            with self.assertRaises(al.ContractError, msg=why):
                al.parse_contract(bad)

    def test_inline_list_and_comments(self):
        c = al.parse_contract("---\nname: x\ndescription: d\nmetadata:\n  requires: [a, b]   # comment\n  reads:\n  produces:\n  effects:\n  conflicts:\n---\n")
        self.assertEqual(c["metadata"]["requires"], ["a", "b"])
        self.assertEqual(c["metadata"]["reads"], [])

    def test_template_itself_is_rejected_until_filled(self):
        tpl = Path(HERE, "..", "templates", "SKILL-CONTRACT.md").read_text(encoding="utf-8")
        with self.assertRaises(al.ContractError):
            al.parse_contract(tpl)          # name empty

    def test_conflicts_hold_both_out(self):
        # GAP-9 decided 11-06 (s8 item 7): a clash between two passing candidates holds both out, order-independent.
        a = al.parse_contract(contract("a", requires=("go",), effects=("sends mail",)))
        b = al.parse_contract(contract("b", requires=("go",), conflicts=("a",)))
        c = al.parse_contract(contract("c", requires=("go",), conflicts=("sends mail",)))
        r = al.gate_many([a, b, c], [], "go")
        self.assertEqual([x["load"] for x in r], [False, False, False])
        self.assertTrue(all("ask the user" in x["why"] for x in r))
        self.assertEqual([x["load"] for x in al.gate_many([b, a], [], "go")], [False, False])


class Files(unittest.TestCase):
    def test_append_only(self):
        before = "a\nb\n"
        self.assertTrue(al.append_only_ok(before, "a\nb\nc\n"))
        self.assertTrue(al.append_only_ok(before, before))
        for after in ("a\nB\nc\n", "a\n", "a\nx\nb\n", "b\na\n"):
            self.assertFalse(al.append_only_ok(before, after), after)

    def test_check_project(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(al.check_project(d)["ok"])
            lp = os.path.join(d, ".state", "LOG.md")
            al.append_log(lp, "# LOG")
            al.append_log(lp, al.render_log(D, "event", "question: How do I open X?", "e", "confirmed"))
            al.append_log(lp, al.render_log(D, "event", "[SRC] port is 8080", "README.md:3", "confirmed"))
            entries, _ = al.read_log(lp)
            text, _ = al.regenerate(entries, "p", D)
            sp = Path(d, ".state", "STATE.md")
            sp.write_text(text, encoding="utf-8")
            self.assertTrue(al.check_project(d)["ok"], al.check_project(d))
            sneaky = f"- [SRC] sneaky edit {EM} status: confirmed {EM} evidence: none {EM} since: {D}\n"
            sp.write_text(text.replace("Claims:\n", "Claims:\n" + sneaky), encoding="utf-8")     # edited STATE, no LOG line
            rep = al.check_project(d)
            self.assertIn("1 STATE claims with no LOG line", rep["problems"])

    def test_receipt_sha(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "src.txt")
            Path(p).write_text("v1")
            r1 = al.receipt_line(p)
            Path(p).write_text("v2")
            self.assertNotEqual(r1, al.receipt_line(p))
            self.assertRegex(r1, r"receipt: .+ — [0-9a-f]{64} — archived no")

    def test_metric_due(self):
        self.assertTrue(al.metric_due([], D))
        self.assertFalse(al.metric_due([L("metric", "rediscoveries=0 tokens_open=1 cold_test=5/5", date="2026-11-01")], D))
        self.assertTrue(al.metric_due([L("metric", "rediscoveries=0 tokens_open=1 cold_test=5/5", date="2026-10-28")], D))

    def test_task_capsule_fits(self):
        t = al.task_capsule("find the port", "port written to STATE", D, frontier=("read README", "read config"), blockers=())
        self.assertLessEqual(al.tokens(t), al.TASK_CAP)

    def test_budget(self):
        b = al.Budget()
        self.assertEqual(sum(b.explore() for _ in range(7)), 5)
        self.assertTrue(b.exhausted)


class Fixes(unittest.TestCase):
    """GAP-1/4/7/8 fixes (11-05), provenance rule and 15-cycle idempotence (TESTS.md s8 items 2-3)."""
    def test_rejected_in_log_reaches_state(self):            # GAP-1
        es = [L("event", "[SRC] port is 8080"), L("decision", "[SRC] port is 8080", st="rejected", date="2026-11-06")]
        text, _ = al.regenerate(es, "p", D)
        self.assertEqual(al.parse_state(text)["claims"][0]["status"], "rejected")

    def test_untagged_tag_from_evidence(self):               # GAP-4
        self.assertEqual(al.split_tag("plain fact", "README.md:9"), ("SRC", "plain fact"))
        self.assertEqual(al.split_tag("plain fact", "src/app.py:3-9, config/app.toml:2"), ("SRC", "plain fact"))
        for ev in ("me", "README.md", "Rui said so", "notes.md lines 5-7", ""):
            self.assertEqual(al.split_tag("plain fact", ev), ("\u00f8", "plain fact"), ev)
        self.assertEqual(al.split_tag("[SYN] x", "README.md:9"), ("SYN", "x"))    # an explicit tag wins

    def test_item_met_exact(self):                           # GAP-7
        self.assertFalse(al.item_met("port is 8080", ["port is not 8080"], ""))
        self.assertFalse(al.item_met("port is 8080", ["the port is 8080 in prod"], ""))
        self.assertTrue(al.item_met("port is 8080", ["Port  is 8080"], ""))
        self.assertFalse(al.item_met("a", [], "anything"))
        self.assertTrue(al.item_met("a", [], "set a to 1"))
        self.assertFalse(al.item_met("", ["x"], "x"))

    def test_empty_requires_never_loads(self):               # GAP-8
        c = al.parse_contract(contract(requires=()))
        self.assertEqual(al.gate(c, ["x"], "anything"), (False, ["requires is empty"]))

    def test_provenance_exact(self):                         # s8 item 3
        es = [L("event", "[SRC] port is 8080")]
        text, _ = al.regenerate(es, "p", D)
        para = text.replace("port is 8080", "the port is 8080")      # paraphrased STATE claim: no LOG line says that
        self.assertEqual(al._unbacked(al.parse_state(para), es), ["the port is 8080"])
        self.assertEqual(al._unbacked(al.parse_state(text), es), [])

    def test_idempotent_15_cycles(self):                     # s8 item 2: 15 closes x 5 LOG lines
        rnd = random.Random(2)
        es, diffs, prev = [], [], set()
        for cyc in range(15):
            for j in range(4):
                i = rnd.randrange(12)
                st = al.CLAIM_STATUSES[rnd.randrange(4)]
                es.append(L("event", f"[SRC] fact-{i} is v{i}", ev=f"src/m{i}.py:{j + 1}", st=st, date=f"2026-11-{1 + cyc:02d}"))
            es.append(L("decision", f"procedure: step {cyc}", ev="me"))
            t1, _ = al.regenerate(es, "p", D)
            t2, _ = al.regenerate(es, "p", D)
            self.assertEqual(t1, t2)                                     # regen is deterministic
            conf = set(al.confirmed_claims(t1))
            derived = {c["text"] for c in al.derive(es)["claims"].values() if c["status"] == "confirmed"}
            self.assertEqual(conf, derived)                              # STATE round-trips the derived confirmed set
            diffs.append({"cycle": cyc, "added": sorted(conf - prev), "removed": sorted(prev - conf)})
            prev = conf
        STRESS["idempotence_15_cycles"] = {"cycles": 15, "log_lines": len(es), "final_confirmed": len(prev),
                                           "cycles_with_change": sum(1 for d in diffs if d["added"] or d["removed"])}

    def test_write_state_atomic(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "STATE.md")
            al.write_state(p, "one\n")
            al.write_state(p, "two\n")
            self.assertEqual(al.read_text(p), "two\n")
            self.assertEqual(os.listdir(d), ["STATE.md"])


class Skeptics(unittest.TestCase):
    """Skeptic round 11-05 (results/cold.json 'skeptics'): refutations of the GAP fixes, now covered."""

    def test_host_port_is_not_a_source(self):
        for ev in ("127.0.0.1:8080", "http://example.com:80/x", "ver 3.13:5", "10.0.0.1:80 timeout", "example.com:8080", "me"):
            self.assertEqual(al.split_tag("fact", ev)[0], "ø", ev)

    def test_source_pointer_forms(self):
        for ev in ("README.md:9", "src/app.py:3-9", "README.md#L9", "src/app.py line 3", "Makefile:12", "config/app.toml:2",
                   "docs/DEPLOY.md:5", "w3.py:1"):
            self.assertEqual(al.split_tag("fact", ev)[0], "SRC", ev)

    def test_item_dot_boundary(self):
        self.assertFalse(al.item_met("v1.2", [], "ship v1.2.3 now"))
        self.assertTrue(al.item_met("v1.2", [], "ship v1.2 now"))
        self.assertTrue(al.item_met("port 8080", [], "use port 8080."))
        self.assertFalse(al.item_met("8080", [], "port 18080"))

    def test_trailing_punctuation(self):
        self.assertTrue(al.item_met("use py", ["Use py."], ""))
        self.assertTrue(al.item_met("port 8080", ["Port 8080."], ""))

    def test_negation_is_a_spec_limit(self):
        # SKILL.md:34 'stated in the user's message': a whole-phrase mention counts, negated or not (documented, not fixed)
        self.assertTrue(al.item_met("8080", [], "do not use 8080"))


class Stress(unittest.TestCase):
    def test_100k_log_lines(self):
        rnd = random.Random(1)
        es = []
        for i in range(100_000):
            k = rnd.randrange(20_000)
            es.append(L("event", f"[SRC] fact {k} about module {k % 97}", st=rnd.choice(al.CLAIM_STATUSES), date=f"2026-{1 + rnd.randrange(11):02d}-{1 + rnd.randrange(28):02d}"))
        t0 = time.perf_counter()
        text, rep = al.regenerate(es, "big", D)
        dt_ = time.perf_counter() - t0
        self.assertLessEqual(rep["tokens"], al.STATE_CAP)
        STRESS["regen_100k_lines"] = {"seconds": round(dt_, 2), "distinct_claims": 20_000, "kept": rep["claims"], "dropped": len(rep["dropped"])}
        self.assertLess(dt_, 60)

    def test_malformed_flood(self):
        rnd = random.Random(2)
        alphabet = string.printable + "—øé中|"
        n_ok = n_bad = 0
        for i in range(10_000):
            s = "".join(rnd.choice(alphabet) for _ in range(rnd.randrange(0, 200)))
            e, err = al.parse_log(s, strict=rnd.random() < 0.5)
            n_ok += e is not None
            n_bad += e is None
            al.parse_claim(s)
            al.parse_state(s)
        STRESS["fuzz_10k_lines"] = {"parsed": n_ok, "rejected": n_bad}

    def test_contract_fuzz(self):
        rnd = random.Random(3)
        pieces = ["---", "name: x", "name:", "description: d", "metadata:", "  requires:", "  - a", "requires:", "- b", "  reads: [x]", "  produces:", "  effects:", "  conflicts:", "   weird: 1", "\t- tab", "", "#c", "metadata: x"]
        ok = err = 0
        for i in range(5_000):
            text = "\n".join(rnd.choice(pieces) for _ in range(rnd.randrange(0, 14)))
            try:
                al.parse_contract(text)
                ok += 1
            except al.ContractError:
                err += 1
        STRESS["contract_fuzz_5k"] = {"accepted": ok, "rejected": err}

    def test_gate_scale(self):
        cs = [al.parse_contract(contract(f"s{i}", requires=(f"needs {i}", f"needs {i % 7}"))) for i in range(2_000)]
        claims = [f"needs {i}" for i in range(0, 2_000, 3)]
        t0 = time.perf_counter()
        r = al.gate_many(cs, claims, "needs 5")
        STRESS["gate_2000_contracts"] = {"seconds": round(time.perf_counter() - t0, 3), "loaded": sum(x["load"] for x in r)}

    def _appenders(self, lp, lock):
        code = ("import sys; sys.path.insert(0, %r); import astate_lite as al\n"
                "for i in range(500): al.append_log(%r, al.render_log('2026-11-05', 'event', '[SRC] w%%s line %%d' %% (sys.argv[1], i), 'e', 'confirmed'), lock=%r)") % (HERE, lp, lock)
        procs = [subprocess.Popen([sys.executable, "-I", "-c", code, str(w)], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)) for w in range(8)]
        errs = [p.communicate()[1].decode(errors="replace") for p in procs]
        entries, errors = al.read_log(lp)
        lines = Path(lp).read_text(encoding="utf-8").count("\n")
        return {"lines": lines, "lost": 4000 - lines, "parsed": len(entries), "torn": len(errors), "proc_errors": sum(1 for e in errs if e.strip())}

    def test_concurrent_appends(self):
        """8 sessions appending at once: 'append-only' needs a lock (unlocked Windows appends lose lines)."""
        with tempfile.TemporaryDirectory() as d:
            t0 = time.perf_counter()
            res = {"unlocked": self._appenders(os.path.join(d, "LOG0.md"), False)}
            res["locked"] = self._appenders(os.path.join(d, "LOG1.md"), True)
            res["seconds"] = round(time.perf_counter() - t0, 2)
            STRESS["concurrent_8x500_appends"] = res
            self.assertEqual(res["locked"]["lines"], 4000)
            self.assertEqual(res["locked"]["torn"], 0)

    def test_mutation_fuzz(self):
        """One-character corruption of valid lines: must never crash; count what still parses."""
        rnd = random.Random(4)
        base = [al.render_log(D, t, "[SRC] port is 8080", "README.md:3", s) for t in al.LOG_TYPES[:3] for s in al.LOG_STATUSES]
        base.append(al.render_claim({"tag": "SRC", "text": "port is 8080", "status": "confirmed", "evidence": "README.md:3", "since": D}))
        acc = 0
        for i in range(5_000):
            s = rnd.choice(base)
            k = rnd.randrange(len(s))
            s = s[:k] + rnd.choice(" |x1-—:[]") + s[k + 1:]
            acc += (al.parse_log(s)[0] is not None) or (al.parse_claim(s) is not None)
        STRESS["mutation_fuzz_5k"] = {"still_parses": acc, "rejected": 5_000 - acc}

    def test_huge_single_line(self):
        s = al.render_log(D, "event", "[SRC] " + "x" * 5_000_000, "e", "confirmed")
        e, err = al.parse_log(s)
        self.assertIsNotNone(e)
        text, rep = al.regenerate([e], "p", D)
        self.assertEqual(rep["claims"], 0)       # the only claim cannot fit the cap and is dropped; reported
        self.assertEqual(len(rep["dropped"]), 1)


class Round2(unittest.TestCase):
    # TESTS.md s8 items 5-9 (11-06): pipes + em-dashes, capsule cap, mutual conflicts, encodings, receipts, PT-BR, 5k message.
    LINE = al.render_log(D, "event", "[SRC] port is 8080", "README.md:3", "confirmed")
    LINE2 = al.render_log(D, "event", "[SRC] logs at var/log/app.log", "README.md:4", "confirmed")

    def _regen(self, d):
        entries, _ = al.read_log(os.path.join(d, ".state", "LOG.md"), strict=False)
        al.write_state(os.path.join(d, ".state", "STATE.md"), al.regenerate(entries, "p", D)[0])

    def _project(self, d, log_bytes=None):
        lp = os.path.join(d, ".state", "LOG.md")
        if log_bytes is None:
            al.append_log(lp, self.LINE)
        else:
            os.makedirs(os.path.dirname(lp), exist_ok=True)
            with open(lp, "wb") as f:
                f.write(log_bytes)
        self._regen(d)
        return lp

    @staticmethod
    def _cap(**kw):
        t = al.task_capsule("g", "done", D, **kw)
        return t if isinstance(t, str) else "\n".join(t)

    def _roundtrip(self, what, ev="README.md:3"):
        text, rep = al.regenerate([L("event", what, ev)], "p", D)
        ps = al.parse_state(text)
        self.assertEqual((ps["bad_claims"], len(ps["claims"])), ([], 1), text)
        return ps["claims"][0]

    # item 5 ---------------------------------------------------------------
    def test_pipe_inside_what_survives(self):
        c = self._roundtrip("[SRC] run is a | b")
        self.assertEqual((c["text"], c["evidence"]), ("run is a | b", "README.md:3"))

    def test_pipe_inside_evidence_is_malformed(self):
        e, err = al.parse_log(al.render_log(D, "event", "[SRC] x", "a | b", "confirmed"))
        self.assertIsNone(e)
        self.assertIn("evidence", err)          # GAP-3 spec limit: the evidence field cannot hold a pipe (rsplit)

    def test_em_dash_inside_claim_and_evidence(self):
        c = self._roundtrip(f"[SRC] range is 1 {EM} 5", f"a {EM} b")
        self.assertEqual((c["text"], c["evidence"]), (f"range is 1 {EM} 5", f"a {EM} b"))
        c = self._roundtrip(f"[SRC] x {EM} status: foo")           # a literal ' — status: ' inside the claim still round-trips
        self.assertEqual(c["text"], f"x {EM} status: foo")

    def test_pipes_em_dash_through_file(self):
        what = f"[SRC] cmd is a | b {EM} c"
        with tempfile.TemporaryDirectory() as d:
            lp = self._project(d)
            al.append_log(lp, al.render_log(D, "event", what, "README.md:5", "confirmed"))
            entries, errors = al.read_log(lp)
            self.assertEqual(errors, [])
            text, _ = al.regenerate(entries, "p", D)
            al.write_state(os.path.join(d, ".state", "STATE.md"), text)
            rep = al.check_project(d)
            self.assertTrue(rep["ok"], rep["problems"])
            self.assertIn(what[6:], [c["text"] for c in al.parse_state(text)["claims"]])

    # item 6 ---------------------------------------------------------------
    def test_capsules_over_cap(self):
        with tempfile.TemporaryDirectory() as d:
            self._project(d)
            self.assertTrue(al.check_project(d)["ok"])
            cap = self._cap(frontier=[f"frontier item number {i}" for i in range(120)])
            al.write_state(os.path.join(d, ".state", "TASK.md"), cap)
            self.assertGreater(al.tokens(cap), 600)
            with open(os.path.join(d, "USER.md"), "w", encoding="utf-8") as f:
                f.write("prefers short replies. " * 110)
            probs = al.check_project(d)["problems"]
            self.assertTrue(any(p.startswith("TASK ") and "> cap 300" in p for p in probs), probs)
            self.assertTrue(any(p.startswith("USER ") and "> cap 300" in p for p in probs), probs)
            al.write_state(os.path.join(d, ".state", "TASK.md"), self._cap())
            os.remove(os.path.join(d, "USER.md"))
            self.assertTrue(al.check_project(d)["ok"])

    # item 7 ---------------------------------------------------------------
    def test_mutual_conflicts_both_orders(self):
        a = al.parse_contract(contract("a", requires=("go",), effects=("sends mail",)))
        b = al.parse_contract(contract("b", requires=("go",), conflicts=("a",)))
        c = al.parse_contract(contract("c", requires=("go",), conflicts=("sends mail",)))
        d = al.parse_contract(contract("d", requires=("go",)))
        for order in ([a, b, c, d], [d, c, b, a], [b, a, d, c]):
            r = {x["name"]: x for x in al.gate_many(order, [], "go")}
            self.assertEqual({n: r[n]["load"] for n in "abcd"}, {"a": False, "b": False, "c": False, "d": True})
            self.assertTrue(all("ask the user" in r[n]["why"] for n in "abc"))
            self.assertIsNone(r["d"]["why"])
        b2 = al.parse_contract(contract("b", requires=("never",), conflicts=("a",)))      # a failing candidate holds nobody out
        self.assertEqual([x["load"] for x in al.gate_many([a, b2], [], "go")], [True, False])
        self.assertEqual([x["load"] for x in al.gate_many([b2, a], [], "go")], [False, True])

    # item 8 ---------------------------------------------------------------
    def test_encodings_never_crash(self):
        both = self.LINE + "\r\n" + self.LINE2 + "\r\n"
        cases = {"crlf": (both.encode("utf-8"), True, None),
                 "utf16_ps51_redirect": (both.encode("utf-16"), False, "LOG.md not UTF-8 (utf-16)"),
                 "bom_after_append": ((self.LINE + "\n").encode("utf-8-sig") + (self.LINE2 + "\n").encode("utf-8-sig"), True, None),
                 "utf16_appended_to_utf8": ((self.LINE + "\n").encode("utf-8") + (self.LINE2 + "\r\n").encode("utf-16-le"), False, "mixed")}
        for name, (data, ok, prob) in cases.items():
            with tempfile.TemporaryDirectory() as d:
                lp = self._project(d, data)
                entries, errors = al.read_log(lp)
                rep = al.check_project(d)
                self.assertEqual(([e["what"] for e in entries], errors), (["[SRC] port is 8080", "[SRC] logs at var/log/app.log"], []), name)
                self.assertEqual(rep["ok"], ok, (name, rep["problems"]))
                if prob:
                    self.assertTrue(any(prob in p for p in rep["problems"]), (name, rep["problems"]))

    def test_utf16_state_and_append_guard(self):
        with tempfile.TemporaryDirectory() as d:
            lp = self._project(d)
            sp = os.path.join(d, ".state", "STATE.md")
            state_text = al.read_text(sp)  # read BEFORE opening for write (11-06: the in-place read saw a truncated file)
            with open(sp, "wb") as f:
                f.write(state_text.encode("utf-16"))
            rep = al.check_project(d)
            self.assertTrue(any("port is 8080" in c for c in al.confirmed_claims(al.read_text(sp))))
            self.assertTrue(any(p.startswith("STATE.md not UTF-8") for p in rep["problems"]), rep["problems"])
            with open(lp, "wb") as f:
                f.write("# LOG\n".encode("utf-16"))
            with self.assertRaises(ValueError):
                al.append_log(lp, self.LINE)

    # item 9 ---------------------------------------------------------------
    def test_receipt_mismatch_after_source_edit(self):
        with tempfile.TemporaryDirectory() as d:
            lp = self._project(d)
            src = os.path.join(d, "README.md")
            with open(src, "w", encoding="utf-8") as f:
                f.write("port 8080\n")
            al.append_log(lp, al.render_log(D, "event", al.receipt_line("README.md", root=d), "close", "confirmed"))
            self._regen(d)
            self.assertTrue(al.check_project(d)["ok"], al.check_project(d)["problems"])
            with open(src, "a", encoding="utf-8") as f:
                f.write("port 9090\n")
            probs = al.check_project(d)["problems"]
            self.assertTrue(any(p.startswith("receipt mismatch: README.md") for p in probs), probs)
            al.append_log(lp, al.render_log(D, "event", al.receipt_line("README.md", root=d), "close", "confirmed"))
            self._regen(d)
            self.assertTrue(al.check_project(d)["ok"])                       # the latest receipt per path is the one checked

    def test_gate_ptbr_and_5k_message(self):
        c = al.parse_contract(contract("pt", requires=("configuração do deploy",)))
        self.assertTrue(al.gate(c, [], "A configuração do deploy está pronta.")[0])
        self.assertTrue(al.gate(c, ["Configuração do deploy"], "")[0])
        self.assertFalse(al.gate(c, [], "configuracao do deploy pronta")[0])         # accents are not folded: exact rule (spec limit)
        filler = "palavra " * 625                                                   # 5,000 characters
        self.assertTrue(al.gate(c, [], filler + "configuração do deploy")[0])
        self.assertFalse(al.gate(c, [], filler + "configuração do build")[0])


class Round2Skeptics(unittest.TestCase):
    # 11-06: the two skeptics' reproduced findings on the round-2 diff, each pinned after its fix (TESTS.md s10).
    LINE, LINE2 = Round2.LINE, Round2.LINE2
    HEAD = f"# LOG {al.EM} p\n(append-only)\n\n"

    def _log(self, d, data):
        lp = os.path.join(d, ".state", "LOG.md")
        os.makedirs(os.path.dirname(lp), exist_ok=True)
        with open(lp, "wb") as f:
            f.write(data)
        return lp

    def _regen(self, d):
        entries, _ = al.read_log(os.path.join(d, ".state", "LOG.md"), strict=False)
        al.write_state(os.path.join(d, ".state", "STATE.md"), al.regenerate(entries, "p", D)[0])

    # encoding-and-io ------------------------------------------------------
    def test_stray_nul_keeps_the_utf8_log(self):
        data = (self.HEAD + self.LINE + "\n" + self.LINE.replace("8080", "he\x00llo") + "\n" + self.LINE2 + "\n").encode("utf-8")
        self.assertEqual(al._decode(data)[1], "utf-8")                      # was: utf-16-le / mixed, every entry lost
        with tempfile.TemporaryDirectory() as d:
            entries, errs = al.read_log(self._log(d, data))
            whats = [e["what"] for e in entries]
            self.assertIn("[SRC] port is 8080", whats)
            self.assertIn("[SRC] logs at var/log/app.log", whats)
            self.assertEqual(len(entries) + len(errs), 3, (whats, errs))

    def test_bomless_and_mixed_utf16_logs_refuse_appends(self):
        tail = (self.LINE2 + "\n").encode("utf-16-le")
        cases = (("bom-less", (self.LINE + "\n").encode("utf-16-le")),
                 ("mixed", (self.HEAD + self.LINE + "\n").encode("utf-8") + tail),
                 ("tail past the head sniff", (self.HEAD + (self.LINE + "\n") * 200).encode("utf-8") + tail))   # > SNIFF bytes of UTF-8 first
        for name, data in cases:
            with tempfile.TemporaryDirectory() as d:
                lp = self._log(d, data)
                with self.assertRaises(ValueError, msg=name):
                    al.append_log(lp, self.LINE)
                self.assertEqual(Path(lp).read_bytes(), data, name)        # nothing appended, lock released cleanly

    def test_lone_cr_separates_entries(self):
        with tempfile.TemporaryDirectory() as d:
            entries, errs = al.read_log(self._log(d, (self.LINE + "\r" + self.LINE2 + "\r").encode("utf-8")))
            self.assertEqual((len(entries), errs), (2, []))                 # was: one entry, no error

    def test_mixed_tail_starting_with_a_wide_char(self):
        self.assertEqual(al._decode(b"abc\n" + "Āxyz\n".encode("utf-16-le")), ("abc\nĀxyz\n", "mixed utf-8 + utf-16"))
        self.assertEqual(al._decode(b"abc\n" + "xyz\n".encode("utf-16-le")), ("abc\nxyz\n", "mixed utf-8 + utf-16"))
        self.assertEqual(al._decode("xyz\n".encode("utf-16-le")), ("xyz\n", "utf-16-le"))
        self.assertEqual(al._decode(b""), ("", "utf-8"))

    # gate-and-receipts ----------------------------------------------------
    def test_receipt_paths_em_dash_dot_slash_upper_sha_and_folder(self):
        with tempfile.TemporaryDirectory() as d:
            lp = os.path.join(d, ".state", "LOG.md")
            al.append_log(lp, self.LINE)
            name = f"a {al.EM} b.txt"
            for fn in (name, "f.txt"):
                with open(os.path.join(d, fn), "w", encoding="utf-8") as f:
                    f.write("one\n")
            os.makedirs(os.path.join(d, "docs"))
            stale = "0" * 64
            for what in (al.receipt_line(name, root=d),
                         f"receipt: ./f.txt {al.EM} {stale} {al.EM} archived no",   # older spelling of the same path: superseded below
                         f"receipt: f.txt {al.EM} {al.sha256_file(os.path.join(d, 'f.txt')).upper()} {al.EM} archived no",
                         f"receipt: docs {al.EM} {stale} {al.EM} archived no"):
                al.append_log(lp, al.render_log(D, "event", what, "close", "confirmed"))
            self._regen(d)
            probs = al.check_project(d)["problems"]
            self.assertEqual(probs, ["receipt names a folder, not a file: docs"], probs)   # was: wrong split, false mismatch, PermissionError
            with open(os.path.join(d, name), "a", encoding="utf-8") as f:
                f.write("two\n")
            probs = al.check_project(d)["problems"]
            self.assertTrue(any(p.startswith(f"receipt mismatch: {name}") for p in probs), probs)

    def test_capsule_or_log_as_a_folder_is_reported_not_raised(self):
        with tempfile.TemporaryDirectory() as d:
            al.append_log(os.path.join(d, ".state", "LOG.md"), self.LINE)
            self._regen(d)
            os.makedirs(os.path.join(d, "TASK.md"))
            self.assertIn("TASK.md is a folder, not a file", al.check_project(d)["problems"])
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, ".state", "LOG.md"))
            rep = al.check_project(d)
            self.assertFalse(rep["ok"])
            self.assertIn("LOG.md is a folder, not a file", rep["problems"])

    def test_clash_matches_like_the_gate(self):
        a = al.parse_contract(contract("A", requires=("go",), effects=("Sends Mail.",)))
        b = al.parse_contract(contract("b", requires=("go",), conflicts=("a",)))
        c = al.parse_contract(contract("c", requires=("go",), conflicts=("sends mail",)))
        self.assertEqual([x["load"] for x in al.gate_many([a, b, c], [], "go")], [False, False, False])   # was: all three loaded
        self.assertTrue(all("ask the user" in x["why"] for x in al.gate_many([c, a], [], "go")))

    def test_tokens_count_cjk_per_char(self):
        self.assertEqual(al.tokens("a" * 400), 100)
        self.assertGreaterEqual(al.tokens("日" * 400), 400)                 # was: 100, under TASK_CAP
        with tempfile.TemporaryDirectory() as d:
            al.append_log(os.path.join(d, ".state", "LOG.md"), self.LINE)
            self._regen(d)
            with open(os.path.join(d, ".state", "TASK.md"), "w", encoding="utf-8") as f:
                f.write("# TASK\n" + "目" * 400 + "\n")
            self.assertTrue(any(p.startswith("TASK ") and "> cap" in p for p in al.check_project(d)["problems"]))


class Gap11(unittest.TestCase):
    # TESTS.md s10 GAP-11 (11-06, user "Yes, add it"): TASK template + task_capsule say where the blocker question goes.
    def test_blocker_note_in_template_and_capsule(self):
        tpl = Path(HERE, "..", "templates", "TASK.md").read_text(encoding="utf-8")
        cap = al.task_capsule("find the port", "port in STATE", D, frontier=("read README",), blockers=("need a docs/ops pointer",))
        for text in (tpl, cap):
            self.assertIn(al.BLOCKER_NOTE, text)
            self.assertLessEqual(al.tokens(text), al.TASK_CAP)
        q = "stg listener port not found in 5 steps; ask the user: extend the budget or point to the file"
        self.assertIsNotNone(al.parse_log(al.render_log(D, "blocker", q, "docs/ops", "unknown"), True)[0])      # question in the what-text
        bad = f"{D} | blocker | port not found | evidence: docs/ops | status: unknown; ask user: {q}"               # the s2/s3 shape
        self.assertIsNone(al.parse_log(bad, True)[0])
        self.assertEqual(len(al.derive([L("blocker", q, "docs/ops", "unknown")])["blockers"]), 1)


if __name__ == "__main__":
    t0 = time.perf_counter()
    res = unittest.main(exit=False, verbosity=1).result
    out = {"tests": res.testsRun, "failures": len(res.failures), "errors": len(res.errors), "seconds": round(time.perf_counter() - t0, 2), "stress": STRESS,
           "failed": [f"{t.id()}: {m.strip().splitlines()[-1]}" for t, m in res.failures + res.errors]}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", "unit.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))
    sys.exit(1 if res.failures or res.errors else 0)
