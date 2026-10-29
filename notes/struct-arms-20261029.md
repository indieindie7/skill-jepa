# struct arms 10-29 (struct_arms.py, data/struct-arms-20261029.json)
Frozen holdout (hash ok), 400 labelled / 1237 rows. A0 hit1 .350, brief tokens top1 488 / top3 1372.
- T2 external state (last-3 sid history + train transition prior, weights on VAL): test hit1 .3525 vs .350, val .296 vs .286 = noise. Tokens unchanged. No idea queued.
- T1 branch routing (BM25 over tree/*.md): GT = transcript Read/cat of tree file <=20 min after query, same sid, gold skill (usage.jsonl has no file field); 68 rows (59 oracle-skill, 11 where A0 top1 right). Branch hit1 .46 (hit3 .64) vs production brief order .17, listed-in-brief-8 .34. Branch files avg ~3.9kt vs SKILL.md ~2.9kt: tokens only drop if the brief names 1-3 branches instead of 8 (~100t saved). Idea #123 (quality).
- T3 top-3 lines only: 173t vs brief 515t (-66%). No quality ground truth; proxy (served line ref in later-opened tree file) .55 n=11. Idea #124 (metric other -> review, not auto-approved).
Caveats: small n for T1/T3 proxy; reads via cat/Read only; Agent/other tool reads not counted.
