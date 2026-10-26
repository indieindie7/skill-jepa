"""ajax_eval.py: route a Windows pytest failure (data/ajax-win-failures.jsonl `q`) to its fix class.
  py -3.13 -I ajax_eval.py [--no-emb] [--out data/eval-ajax-<date>.json]
Routers: bm25_knn (token BM25 over q, k=5 vote), emb_knn (EmbeddingGemma on SKILLJEPA_EMB_URL, cosine k=5 vote), class_centroid.
Splits: loo (leave-one-record-out: the same test file's other failures are visible, close to what a re-run sees) and
file (leave-one-FILE-out: the router never saw that test module; the honest number for a new failing module).
Hit@1/Hit@3 over all records and over the test-file/keyword/residue subsets (unresolved records are skipped as labels)."""
import argparse, collections, datetime, json, math, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "ajax-win-failures.jsonl")
TOK = re.compile(r"[a-z0-9]+")


def toks(s):
    return [t for t in TOK.findall(s.lower()) if len(t) > 1]


def bm25(docs, k1=1.2, b=0.75):
    N, df, avg = len(docs), collections.Counter(), sum(map(len, docs)) / max(len(docs), 1)
    for d in docs:
        df.update(set(d))
    idf = {t: math.log(1 + (N - n + 0.5) / (n + 0.5)) for t, n in df.items()}
    tfs = [collections.Counter(d) for d in docs]

    def score(q, i):
        tf, L = tfs[i], len(docs[i])
        return sum(idf.get(t, 0) * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * L / avg)) for t in set(q) if t in tf)
    return score


def vote(scored, k):
    """scored: [(score, label)] desc -> ranked labels by summed score of the top-k."""
    agg = collections.defaultdict(float)
    for s, lab in scored[:k]:
        agg[lab] += s
    return [l for l, _ in sorted(agg.items(), key=lambda x: -x[1])]


def folds(recs, split):
    if split == "loo":
        for i in range(len(recs)):
            yield [i], [j for j in range(len(recs)) if j != i]
    else:
        byf = collections.defaultdict(list)
        for i, r in enumerate(recs):
            byf[r["file"]].append(i)
        for f, test in byf.items():
            yield test, [j for j in range(len(recs)) if recs[j]["file"] != f]


def evaluate(recs, split, routers, k=5):
    hits = collections.defaultdict(lambda: collections.Counter())
    for test, train in folds(recs, split):
        for name, rank in routers.items():
            for i in test:
                r = rank(i, train, k)
                lab, a = recs[i]["class"], recs[i]["attribution"]
                for key in ("all", a):
                    hits[name][key + ":n"] += 1
                    hits[name][key + ":h1"] += lab == (r[0] if r else None)
                    hits[name][key + ":h3"] += lab in r[:3]
    out = {}
    for name, c in hits.items():
        out[name] = {key: {"n": c[key + ":n"], "hit1": round(c[key + ":h1"] / c[key + ":n"], 3), "hit3": round(c[key + ":h3"] / c[key + ":n"], 3)}
                     for key in ("all", "test-file", "keyword", "residue", "pinned") if c[key + ":n"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-emb", action="store_true"); ap.add_argument("--out"); ap.add_argument("--data", default=DATA)
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
    recs = [r for r in recs if r["attribution"] != "unresolved"]
    docs = [toks(r["q"]) for r in recs]
    labs = [r["class"] for r in recs]
    routers = {}
    # bm25 over the full corpus idf (train-only idf would need a rebuild per fold; the idf leak is tiny for 600 docs)
    score = bm25(docs)
    routers["bm25_knn"] = lambda i, train, k: vote(sorted(((score(docs[i], j), labs[j]) for j in train), reverse=True), k)
    emb = None
    if not a.no_emb:
        try:
            sys.path.insert(0, HERE)
            import numpy as np
            from skilljepa import embed
            emb = embed([r["q"] for r in recs])
            sims = emb @ emb.T
            routers["emb_knn"] = lambda i, train, k: vote(sorted(((float(sims[i, j]), labs[j]) for j in train), reverse=True), k)

            def centroid(i, train, k):
                byc = collections.defaultdict(list)
                for j in train:
                    byc[labs[j]].append(j)
                cs = {c: emb[idx].mean(0) for c, idx in byc.items()}
                return [c for c, _ in sorted(((c, float(emb[i] @ v) / (np.linalg.norm(v) + 1e-9)) for c, v in cs.items()), key=lambda x: -x[1])]
            routers["emb_centroid"] = centroid
        except Exception as e:
            print("skip emb:", str(e)[:160], file=sys.stderr)
    res = {"n": len(recs), "classes": len(set(labs)), "majority": round(collections.Counter(labs).most_common(1)[0][1] / len(recs), 3)}
    for split in ("loo", "file"):
        res[split] = evaluate(recs, split, routers)
        for name, d in res[split].items():
            print(f"{split:4} {name:13} " + "  ".join(f"{key} h1={v['hit1']:.3f} h3={v['hit3']:.3f} n={v['n']}" for key, v in d.items()))
    print("n", res["n"], "classes", res["classes"], "majority", res["majority"])
    out = a.out or os.path.join(HERE, "data", f"eval-ajax-{datetime.date.today():%Y%m%d}.json")
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1)
    print("->", out)


if __name__ == "__main__":
    main()
