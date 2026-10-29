"""a10_frontier.py: ledger A10 (notes/scans-2026-10-29.md sec 4 Stage 1): frontier_recall(K) = P(a gold skill in candidate top-K), labelled rows only.
  C:/llm/qlora/venv/Scripts/python.exe a10_frontier.py [--rows data/holdout.jsonl] [--out notes/a10-frontier-20261029.md]   (read-only, CPU)
Frontiers: A0 = arms.A0 production-style (trigger + 0.1*BM25 + prev-skill prior); bm25 = pure BM25(SKILL.md) over all skills; trig = trigger score only.
Gold skills missing from the current skill set are counted apart (renamed/removed, unrecoverable by any frontier)."""
import argparse, collections, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import arms, opsd

KS = (1, 3, 5, 8, 10, 16)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--rows", default=os.path.join(HERE, "data", "holdout.jsonl"))
    ap.add_argument("--out", default=os.path.join(HERE, "notes", "a10-frontier-20261029.md")); a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.rows, encoding="utf-8")]; a0 = arms.A0()
    lab = [r for r in rows if [s for s in r["skills"] if s != "none"]]
    hit = {f: collections.Counter() for f in ("A0", "bm25", "trig")}; miss = collections.Counter(); gone = 0; n = 0; poolsz = []
    for r in lab:
        gold = {s for s in r["skills"] if s != "none"}; live = gold & set(a0.names)
        if not live: gone += 1; continue
        n += 1; qt = opsd._tok(r["q"])
        rk = {"A0": [k for k, _ in a0.score(r)],
              "bm25": [k for k, v in sorted(((nm, a0.bm25(qt, nm)) for nm in a0.names), key=lambda kv: -kv[1]) if v > 0],
              "trig": [k for k, v in sorted(((nm, a0.trig(r["q"], nm) or 0.0) for nm in a0.names), key=lambda kv: -kv[1]) if v > 0]}
        poolsz.append(len(rk["A0"]))
        for f, lst in rk.items():
            for K in KS + (999,):
                if live & set(lst[:K]): hit[f][K] += 1
        if not live & set(rk["A0"][:10]): miss.update(live)
    L = [f"# A10 frontier_recall(K) 2026-10-29", f"rows {len(rows)}, labelled {len(lab)}, gold gone from skill set {gone}, scored n={n}; A0 pool size median {sorted(poolsz)[len(poolsz)//2] if poolsz else 0}", "",
         "| frontier | " + " | ".join(f"K={k}" for k in KS) + " | all>0 |", "|---" * (len(KS) + 2) + "|"]
    for f in hit: L.append(f"| {f} | " + " | ".join(f"{hit[f][k]/max(n,1):.3f}" for k in KS) + f" | {hit[f][999]/max(n,1):.3f} |")
    L += ["", "gold skills most often outside A0 top-10 (lexically distant candidates): " + ", ".join(f"{k} {v}" for k, v in miss.most_common(12))]
    open(a.out, "w", encoding="utf-8").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
