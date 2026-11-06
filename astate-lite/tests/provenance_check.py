"""provenance_check.py: TESTS.md s8 item 3, the exact provenance rule ('never edit STATE without a LOG line') run on the
dogfood sessions whose STATE was hand-written (s1, s3: STATE claims paraphrase the LOG lines).
Usage: py -3.13 -I provenance_check.py <lanterna_root>   -> results/provenance.json"""
import os, sys, json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import astate_lite as al   # noqa: E402


def one(root):
    rep = al.check_project(root, strict=True)
    ps = al.parse_state(al.read_text(os.path.join(root, ".state", "STATE.md")))
    unb = rep.get("unbacked_claims", [])
    entries, _ = al.read_log(os.path.join(root, ".state", "LOG.md"), strict=False)
    regen = al.parse_state(al.regenerate(entries, "lanterna", "2026-11-05")[0])     # the cure: STATE derived by the impl
    return {"claims": len(ps["claims"]), "unbacked": len(unb), "ok": rep.get("ok"), "problems": rep.get("problems", []),
            "unbacked_claims": unb, "unbacked_after_impl_regen": len(al._unbacked(regen, entries))}


if __name__ == "__main__":
    fx = sys.argv[1]
    res = {s: one(os.path.join(fx, s)) for s in ("s1", "s3", "s5", "s6")}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "provenance.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(json.dumps({s: {k: r[k] for k in ("claims", "unbacked", "ok")} for s, r in res.items()}))
