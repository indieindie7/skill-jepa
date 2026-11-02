"""test_qstd.py: qstd metrics on synthetic code. py -3.13 -I test_qstd.py"""
import os, subprocess, sys, tempfile, unittest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qstd

FLAT = "def f(x):\n    return x + 1\n"
BRANCHY = "def g(x):\n" + "".join(f"    if x == {i}:\n        x += {i}\n" for i in range(12)) + "    return x\n"   # CC 13
BLOCK = "".join(f"    total_{i} = compute(value_{i}, offset={i})\n" for i in range(8))


class Metrics(unittest.TestCase):
    def test_cc(self):
        self.assertEqual(qstd.fns(FLAT), [("f", 1, 2)])
        self.assertEqual(qstd.fns(BRANCHY)[0][1], 13)
        self.assertEqual(qstd.fns("def h(a, b):\n    return [x for x in a if x and b or x]\n")[0][1], 1 + 2 + 2)   # comprehension + its if + 2 bool ops

    def test_nested_scopes(self):
        F = dict((q, c) for q, c, _ in qstd.fns("class A:\n    def m(self, x):\n        def inner(y):\n            if y: return 1\n        return inner(x)\n"))
        self.assertEqual(F, {"A.m": 1, "A.m.inner": 2})

    def test_erosion_share(self):
        m = qstd.metrics({"a.py": FLAT, "b.py": BRANCHY})
        self.assertGreater(m["erosion"], .9); self.assertEqual(m["cc10"], 1)
        self.assertEqual(qstd.metrics({"a.py": FLAT})["erosion"], 0.0)

    def test_dup_copy_paste(self):
        one = qstd.metrics({"a.py": "def a():\n" + BLOCK})
        two = qstd.metrics({"a.py": "def a():\n" + BLOCK, "b.py": "def b():\n" + BLOCK})
        self.assertEqual(one["dup"], 0.0); self.assertGreater(two["dup"], .8)

    def test_crossed(self):
        grown = BRANCHY.replace("def g", "def f")
        d = qstd.compare(qstd.metrics({"a.py": FLAT}), qstd.metrics({"a.py": grown}))
        self.assertEqual(d["crossed"], ["a.py::f 1->13"]); self.assertGreater(d["d_erosion"], .5)

    def test_syntax_error_counted(self):
        self.assertEqual(qstd.metrics({"bad.py": "def (:\n"})["parse_err"], 1)


def put(d, src):
    with open(os.path.join(d, "m.py"), "w", encoding="utf-8") as f: f.write(src)


class Gate(unittest.TestCase):
    def test_gate_on_temp_repo(self):
        with tempfile.TemporaryDirectory() as d:
            run = lambda *a: subprocess.run(["git", "-C", d, *a], capture_output=True, check=True)
            run("init", "-q"); run("config", "user.email", "t@t"); run("config", "user.name", "t")
            put(d, FLAT); run("add", "m.py"); run("commit", "-qm", "base")
            ok = f'"{sys.executable}" -c "pass"'
            self.assertEqual(self._gate(d, ok), 0)                                     # unchanged + suite ok -> PASS
            self.assertEqual(self._gate(d, None), 2)                                   # no suite -> UNTESTED
            put(d, BRANCHY.replace("def g", "def f"))
            self.assertEqual(self._gate(d, ok), 1)                                     # fn crossed CC10 + erosion rise -> FAIL
            put(d, FLAT)
            self.assertEqual(self._gate(d, f'"{sys.executable}" -c "raise SystemExit(1)"'), 1)   # old suite broken -> FAIL

    def _gate(self, d, test):
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()) as out: rc = qstd.gate(d, test)
        self.assertTrue(out.getvalue().startswith("qstd ")); return rc


if __name__ == "__main__": unittest.main()
