"""test_qstd_falsifiers.py: one test per falsifier met in real use (notes/qstd-replay-20261102.md, skill quality-standard),
each on GENERATED code over many seeds, with a positive case (gate must FAIL) and a negative control (gate must PASS),
plus metamorphic invariants: edits that change no behaviour must change no metric.  py -3.13 -I test_qstd_falsifiers.py"""
import os, random, re, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qstd

SEEDS = range(12)
OK = {"ok": True, "rc": 0}


def module(seed, nfn=6, branches=(2, 6), tag="fn"):
    """Random module: nfn flat-if functions, every line unique (no accidental dup windows)."""
    r = random.Random(seed); out = []
    for i in range(nfn):
        nb = r.randint(*branches)
        body = "".join(f"    if x == {r.randint(0, 99)}:\n        x = x * {r.randint(2, 9)} + {seed * 10000 + i * 100 + j}\n" for j in range(nb))
        out.append(f"def {tag}_{seed}_{i}(x):\n{body}    return x\n\n")
    return "".join(out)


def branchy(name, n, loc_pad=0, off=0):
    """fn with CC n+1 (n flat ifs on constants off..off+n); loc_pad extra unique lines to weight it."""
    pad = "".join(f"    y_{k} = x + {k}\n" for k in range(loc_pad))
    return f"def {name}(x):\n{pad}" + "".join(f"    if x == {i}:\n        x += {i}\n" for i in range(off, off + n)) + "    return x\n"


def verdict(a, b, suite=OK):
    return qstd.judge(qstd.compare(qstd.metrics(a), qstd.metrics(b)), suite)[0]


class Falsifiers(unittest.TestCase):
    """Each test names the real event that falsified the naive assumption."""

    def test_copy_patch_variant_fails(self):
        """skilljepa c6aeb10: a6c_nudger.py = copy of a6b with a few lines changed (dup .85). Fresh copy must FAIL; import+wrap must PASS."""
        for s in SEEDS:
            m = module(s); patched = m.replace(f"def fn_{s}_0(", f"def fn_{s}_0_v2(").replace("x * 2", "x * 3")
            self.assertEqual(verdict({"a.py": m}, {"a.py": m, "b.py": patched}), "FAIL", s)
            wrap = f"from a import fn_{s}_0\n\ndef fn_{s}_0_v2(x):\n    return fn_{s}_0(x) + 1\n"
            self.assertEqual(verdict({"a.py": m}, {"a.py": m, "b.py": wrap}), "PASS", s)

    def test_pasted_branch_pairs_cross_cc10(self):
        """SlopCodeBench 'forge' main(): 7 pasted branch pairs, argparse -> hand flag loop. Growth past CC10 FAILs; dict dispatch PASSes."""
        for s in SEEDS:
            r = random.Random(s); grown = branchy("main", r.randint(10, 20))
            self.assertEqual(verdict({"m.py": branchy("main", 4)}, {"m.py": grown}), "FAIL", s)
            dispatch = "def main(x):\n    return OPS.get(x, lambda v: v)(x)\n" + "".join(f"def op{i}(v):\n    return v + {i}\n" for i in range(12)) + "OPS = {" + ", ".join(f"{i}: op{i}" for i in range(12)) + "}\n"
            self.assertEqual(verdict({"m.py": branchy("main", 4)}, {"m.py": dispatch}), "PASS", s)

    def test_big_repo_crossed_rule_carries(self):
        """codes repo (700+ files): 14 fns crossed CC10 while erosion share moved +.004 (< THR). FAIL must come from `crossed`."""
        heavy = {f"f{i}.py": branchy(f"g{i}", 8, loc_pad=30) for i in range(300)}      # CC 9 each: just under the line
        a = dict(heavy, **{"t.py": branchy("t", 4)}); b = dict(heavy, **{"t.py": branchy("t", 11)})
        d = qstd.compare(qstd.metrics(a), qstd.metrics(b))
        self.assertLess(d["d_erosion"], qstd.THR); self.assertEqual(len(d["crossed"]), 1)
        self.assertEqual(qstd.judge(d, OK)[0], "FAIL")

    def test_green_narrow_suite_is_not_pass(self):
        """playtest 10555ee: 17/17 new tests green, next commit fixed 4 bugs the suite missed. Without the OLD suite the verdict is UNTESTED, never PASS."""
        m = module(1)
        self.assertEqual(verdict({"a.py": m}, {"a.py": m}, suite=None), "UNTESTED")
        self.assertEqual(verdict({"a.py": m}, {"a.py": m}, suite={"ok": False, "rc": 1}), "FAIL")

    def test_delete_dead_code_rewarded(self):
        """'Agents refuse to ever delete': removing the duplicated copy must PASS and lower dup; it is never a regression by itself."""
        for s in SEEDS:
            m = module(s); d = qstd.compare(qstd.metrics({"a.py": m, "b.py": m}), qstd.metrics({"a.py": m}))
            self.assertLess(d["d_dup"], 0, s); self.assertEqual(qstd.judge(d, OK), ("PASS", [], 1.0), s)

    def test_split_cc23_into_helpers_passes(self):
        """qstd.py's own first draft: gate() CC 23 FAILed its gate; split along sub-tasks -> PASS with lower erosion."""
        big = {"q.py": branchy("gate", 22)}
        parts = {"q.py": "".join(branchy(f"part{k}", 7, off=7 * k) for k in range(3)) + "def gate(x):\n    return part2(part1(part0(x)))\n"}
        self.assertEqual(verdict({"q.py": branchy("gate", 2)}, big), "FAIL")
        d = qstd.compare(qstd.metrics(big), qstd.metrics(parts))
        self.assertEqual(d["crossed"], []); self.assertLess(d["d_erosion"], 0); self.assertEqual(qstd.judge(d, OK)[0], "PASS")

    def test_revert_is_neutral(self):
        """water-gym foam rounds: flagged commits then reverted. Back at the accepted base every delta is 0 and verdict PASS."""
        for s in SEEDS:
            m = {"a.py": module(s)}; d = qstd.compare(qstd.metrics(m), qstd.metrics(dict(m)))
            self.assertEqual((d["d_erosion"], d["d_dup"], d["crossed"]), (0.0, 0.0, []), s)

    def test_broken_file_cannot_look_like_cleanup(self):
        """A file that no longer parses drops out of the fn weights, so erosion FALLS. That must read as FAIL, not improvement."""
        a = {"a.py": branchy("g", 12), "b.py": module(2)}; b = dict(a, **{"a.py": "def g(x:\n    return\n"})
        self.assertEqual(verdict(a, b), "FAIL")

    def test_reward_orders_by_damage(self):
        """Reward feeds JEPA (plan sec 10 Q1): PASS > UNTESTED > FAIL, and a bigger rise costs more within one verdict."""
        mk = lambda e: {"d_erosion": e, "d_dup": 0.0, "d_cc10": 0, "d_parse": 0, "crossed": []}
        self.assertGreater(qstd.judge(mk(0.0), OK)[2], qstd.judge(mk(0.0), None)[2])
        self.assertGreater(qstd.judge(mk(0.0), None)[2], qstd.judge(mk(0.0), {"ok": False, "rc": 1})[2])
        self.assertGreater(qstd.judge(mk(0.1), OK)[2], qstd.judge(mk(0.3), OK)[2])


class Invariants(unittest.TestCase):
    """Behaviour-neutral edits must move no metric (no hallucinated rot, no gaming)."""

    def same(self, files, other, seed):
        a, b = qstd.metrics(files), qstd.metrics(other)
        for k in ("erosion", "dup", "cc10", "fns"): self.assertEqual(a[k], b[k], (k, seed))
        self.assertEqual(qstd.compare(a, b)["crossed"], [], seed)

    def test_rename_identifiers(self):
        for s in SEEDS:
            m = module(s); self.same({"a.py": m}, {"a.py": re.sub(r"\bx\b", "value", m).replace(f"fn_{s}_", f"renamed_{s}_")}, s)

    def test_reorder_functions(self):
        for s in SEEDS:
            parts = module(s).split("\n\n"); random.Random(s).shuffle(parts)
            self.same({"a.py": module(s)}, {"a.py": "\n\n".join(parts)}, s)

    def test_comments_blanks_whitespace(self):
        """Comment/blank padding cannot dilute dup share; re-indentation and spacing cannot create windows."""
        for s in SEEDS:
            m = module(s); dup2 = {"a.py": m, "b.py": m}
            padded = {"a.py": m.replace("\n", "\n# note\n\n"), "b.py": m.replace("    ", "\t").replace(" = ", "   =  ")}
            self.assertEqual(qstd.metrics(dup2)["dup"], qstd.metrics(padded)["dup"], s)
            self.same({"a.py": m}, {"a.py": m.replace("\n", "\n\n# c\n")}, s)

    def test_move_file_is_not_a_crossing(self):
        """Renaming/moving a file keeps its fns; a CC 12 fn that moved was not written this change."""
        for s in SEEDS:
            src = module(s) + branchy(f"old_{s}", 11)
            d = qstd.compare(qstd.metrics({"a.py": src}), qstd.metrics({"pkg/b.py": src}))
            self.assertEqual(d["crossed"], [], s)

    def test_two_line_overlap_is_not_dup(self):
        """Rule 2 limit (video-notes tree/427): a few shared lines between two fns are fine; the window is 6."""
        for s in SEEDS:
            r = random.Random(s); shared = "".join(f"    t_{k} = x + {k}\n" for k in range(r.randint(1, qstd.WIN - 1)))
            src = f"def a(x):\n{shared}    return x * 2\n\ndef b(x):\n{shared}    return x * 3\n"
            self.assertEqual(qstd.metrics({"a.py": src})["dup"], 0.0, s)


if __name__ == "__main__": unittest.main()
