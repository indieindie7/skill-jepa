"""Tripwire dispatcher: 5-fold check of prototypes built from real labelled messages (tree/tripwire-print.md s3d).
  py -3.13 -I -X utf8 tripwire_real_train.py <outdir> [--write]
<outdir> holds real_msgs.txt + sage_labels.txt (tripwire_real_collect.py + localread labels). Private text: never publish.
Compares prototype sets old (source PROTOS) / old+real / real(+old cancel), per threshold pair, on held-out folds.
--write: saves ~/.claude/hooks/tripwire_protos_real.json = {"protos": {class: [texts]}} from ALL labelled messages."""
import json
import re
import sys
from pathlib import Path

H = Path(__file__).resolve().parents[1] / "hooks"
sys.path.insert(0, str(H if (H / "tripwire.py").exists() else Path.home() / ".claude/hooks"))
import tripwire_dispatch as td  # noqa: E402

OUT = Path(sys.argv[1])
K = 5
GRID = [(s, m) for s in (0.50, 0.55, 0.60, 0.65, 0.70) for m in (0.0, 0.02, 0.03, 0.05, 0.08)]


def load():
    msgs = {int(m.group(1)): m.group(2) for m in (re.match(r"(\d+)\. (.*)", ln) for ln in open(OUT / "real_msgs.txt", encoding="utf-8")) if m}
    labs = {int(m.group(1)): m.group(2) for m in (re.match(r"(\d+): ([a-z_]+)", ln) for ln in open(OUT / "sage_labels.txt", encoding="utf-8")) if m}
    return [(n, msgs[n], labs[n]) for n in sorted(labs) if n in msgs and labs[n] in td.PROTOS]


def embed_all(texts):
    out = []
    for i in range(0, len(texts), 24):
        out += td.embed([t[:600] for t in texts[i:i + 24]])
    return out


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def nearest(q, protos):
    scores = {}
    for cls, vecs in protos.items():
        top = sorted((dot(q, v) for v in vecs), reverse=True)[:2]
        scores[cls] = sum(top) / len(top) if top else -1.0
    rank = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return rank[0][0], rank[0][1], rank[0][1] - rank[1][1]


def script_tier(text):
    if td.STOP.search(text):
        return "cancel"
    if td.URL_ONLY.match(text):
        return "new_task"
    return None


def fold_protos(kind, old, rows, vecs, fold):
    p = {c: list(v) if kind != "real" or c == "cancel" else [] for c, v in old.items()}
    if kind != "old":
        for i, (_, _, lab) in enumerate(rows):
            if i % K != fold:
                p[lab].append(vecs[i])
    return p


def run(kind, old, rows, vecs):
    """Held-out predictions: [(label, script class or None, nearest class, score, margin)]."""
    pred = []
    for fold in range(K):
        p = fold_protos(kind, old, rows, vecs, fold)
        for i, (_, text, lab) in enumerate(rows):
            if i % K == fold:
                pred.append((lab, script_tier(text)) + nearest(vecs[i], p))
    return pred


def score(pred, smin, mmin, silent_remark):
    esc = wrong = spoke = 0
    for lab, sc, cls, s, m in pred:
        if sc == "new_task":
            spoke += 1
            wrong += lab != "new_task"
        elif sc == "cancel" or cls == "cancel" or s < smin or m < mmin or (cls == "remark" and not silent_remark):
            esc += 1
        else:
            spoke += 1
            wrong += cls != lab
    return esc, spoke, wrong


def main():
    rows = load()
    vecs = embed_all([t for _, t, _ in rows])
    old = {c: embed_all(t) for c, t in td.PROTOS.items()}
    n = len(rows)
    print("labelled", n, {c: sum(1 for r in rows if r[2] == c) for c in td.PROTOS})
    res = {}
    for kind in ("old", "old+real", "real"):
        report(kind, run(kind, old, rows, vecs), n, res)
    (OUT / "real_train.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    if "--write" in sys.argv:
        write_protos(rows)


def report(kind, pred, n, res):
    acc = sum(1 for lab, sc, cls, _, _ in pred if (sc or cls) == lab)
    print(f"\n{kind}: nearest (after script tier) == label {acc} of {n}")
    for silent in (False, True):
        best = None
        for smin, mmin in GRID:
            esc, spoke, wrong = score(pred, smin, mmin, silent)
            ok = spoke and wrong / spoke <= 0.10
            if ok and (best is None or esc < best[0]):
                best = (esc, spoke, wrong, smin, mmin)
        cur = score(pred, td.MIN_SCORE, td.MIN_MARGIN, silent)
        print(f"  remark {'silent' if silent else 'escalates'}: at current {td.MIN_SCORE}/{td.MIN_MARGIN} esc {cur[0]} spoke {cur[1]} wrong {cur[2]};"
              f" least escalation with wrong <= 10 %: {best}")
        res[f"{kind}|{'silent' if silent else 'esc'}"] = {"current": cur, "best": best, "acc": acc, "n": n}


def write_protos(rows):
    p = {c: [] for c in td.PROTOS}
    for _, text, lab in rows:
        p[lab].append(text[:300])
    dst = Path.home() / ".claude/hooks/tripwire_protos_real.json"
    dst.write_text(json.dumps({"protos": p}, ensure_ascii=False, indent=0), encoding="utf-8")
    print("wrote", dst, {c: len(v) for c, v in p.items()})


main()
