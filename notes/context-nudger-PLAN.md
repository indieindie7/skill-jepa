# Context nudger: autonomy + loop-breaking (chat "jepa-context work chat", started 2026-10-27)

## Index
1. Mission (lines 10-14): autonomy/creativity for models, nudger = small space
2. Assumptions to falsify (lines 15-24): A1-A7, each with "what would make this wrong" + test
3. Nudger design (lines 25-35): the few bits per turn, where they plug in
4. Metrics (lines 36-41): autonomy, novelty, falsifier, cost
5. Status + next step (lines 42-)

## 1. Mission
User 10-27: "enriching and falsifying assumptions for breaking agents from their core loops ... giving autonomy and creativity for the models while being a small nudger that uses small space."
Tandem chat: "JEPA-skill work chat" owns the router code (avb_rerank.py, trace_merge.py, opsd /api/route, plan sec 9). This chat owns: the assumption ledger below, the nudger design, and experiments that test them. Never edit router files here without telling that chat.
Their reply 10-27: hold eval split + AVB head (avb_rerank.py, data/pairs*.jsonl, data/avb-*.json), judge_fresh.py + data/judge_disagree.jsonl, trace loop (trace_merge.py, data/trace_state.json). DO NOT TOUCH: ops/opsd.py route(), route_trace.py, route.log, data/*.jsonl, nightly routines. Read freely; branch for edits.
Ground: SELFIMPROVE_PLAN.md sec 9 (all 4 boxes built 10-27) + notes/intelligence-autonomy-2026-10-26.md sec "Breaking the alignment loop".

## 2. Assumptions to falsify
Each: claim -> what would make it wrong -> cheap test -> result.
- A1 Closed loop = confirmation bias. Evidence: router==label .867, fresh judge==router .233 (judge_fresh.py, 30 rows). Wrong if: judge disagrees equally on context-free prompts (then gap = missing context, not bias). Test: split judge_disagree.jsonl by prompt context-dependence (Sage tag), compare agree rates. Result: PENDING.
- A2 Low surprise = time to change rules (Ashby 2nd loop). Wrong if: rule changes after a low_surprise flag give no hit1/falsifier gain. Test: when flag fires, swap one rule (K, cand source, data mix), 3 nights A/B. Result: flag needs 7 runs, not fired yet.
- A3 5% exploration slot surfaces useful unseen skills. Wrong if: explore_hit_rate stays ~0 after 14 d, or hits are only skills already in top-5. Test: trace_state history explore_hit_rate; count explore hits not in top-k. Result 10-27 (nudger_measure.py, 1866 rows): 0 live explore rows yet; retro least-routed-14d candidate read in window .0091 < uniform random .0136 -> least-routed policy is WORSE than random; reads outside top-3 happen on .593 of read rows, dominated by meta skills (ops-manager 180, local-gpu-ai 157). A3 as built: likely falsified; the slot needs a better source (see sec 3 'outside').
- A4 Refute templates carry information. Wrong if: refuted rate == miss rate (template just restates hit/miss). Test: corr(refuted, class miss) over 200 rows; if |r|>.9 templates are redundant, need model-written refutations. Result 10-27: FALSIFIED (retro over 1866 rows): refuted is a function of (conf,class) in 11/13 cells; the 2 mixed cells are full_read + a second skill read (372/567), which is not a refutation of top-1. BUG found: trace_pairs.refuted()/ops route_trace copy invert the low ('no skill read' -> returns True when nothing was read = confirmed) and none templates; only high/medium correct. Reported to JEPA-skill chat 10-27.
- A5 Fresh-context judge = outside input. Wrong if: its disagreement never changes a decision (logged, never acted on = no loop break). Test: pick 10 disagreements, run the judge's skill for the next same-prompt, log outcome. Result: PENDING.
- A6 A tiny nudger (few bits/turn) can raise model autonomy without dominating. Wrong if: with nudger on, routed-skill entropy and novel-choice rate do not rise, or hit1 falls >.05. Test: entropy of route.log top-1 per day, nudger on vs off (seeded by day). Result: NOT BUILT. Baseline 10-27: daily top-1 entropy fell 4.36 bits (10-17, 51 distinct) -> 1.3-2.0 bits (10-21..27, 7-10 distinct); video-notes = .55 of all top-1 (router pulls unrelated prompts to it, e.g. this chat's goal prompt scored video-notes 4.1). Loop narrowing visible (confounded by which projects were active).
- A7 Self-labels beat outside labels (sec 1: Sage relabel no gain .576 vs .599). Wrong if: on held-out *future* rows (next 7 nights) the outside-labelled head generalises better. Test: train both on rows < day D, eval on rows > D. Result: PENDING (needs trace volume).

## 3. Nudger design (small space)
Principle: the nudger never answers; it changes the question. Budget: <= 3 short lines in the prompt, <= 1 extra candidate, no weights of its own at first (rules, then a tiny head once rules prove out).
Bits per turn:
- explore: one out-of-top-k candidate (built, sec 9).
- refute: one "what would make this wrong" line (built as template; upgrade = model writes it in <= 12 words, stored in route.log col 7).
- surprise: one flag when agreement too high (built; action missing -> A2).
- outside: one unplanned item (least-routed skill, random memory line, user's unplanned video note) injected 1 in 20 turns; log read/no-read like explore.
- counter: when top-1 confidence > .9, show the strongest *other* candidate with one reason (cheap, stops lock-in).
Plug points: opsd.route() (candidates), UserPromptSubmit hook (prompt lines), trace_merge.py (metrics). Nothing new persists except route.log columns.
Creativity side: nudger prefers candidates that are far (cosine) from the last 5 routed skills when confidence is low; never when high.

## 4. Metrics (beside hit1 + abstain)
- autonomy: entropy of daily top-1 skill distribution; novel-choice rate = top-1 not seen in last 14 d.
- falsifier_score (built): refuted / with_prediction.
- outside_hit_rate: outside items read / shown.
- cost: extra prompt tokens per turn (target < 60) and extra route latency.

## 5. Status + next step
- 10-27: file created; assumptions A1-A7 listed; tandem chat greeted, waiting for its state reply.
- 10-27 18:41: nudger_measure.py (read-only) -> notes/nudger-measure-2026-10-27.md. A4 falsified + inversion bug; A3 least-routed worse than random; A6 baseline entropy collapse. Committed here, not pushed.
- NEXT: (1) JEPA chat fixes refuted() low/none inversion; (2) A3 replacement source: candidate = skill read outside top-3 most often in last 14 d that is not in cands (usual-suspects slot) vs random, measure retro hit rate; (3) A1 split via Sage on gpu 1 when free.

## 6. Peer chats' holds + inputs (10-27 replies)
- Tree-router chat: read-only for me: ~/.claude/hooks/recall.py (BIG_TREE filter), recall_judge.py (+recall-judge.json, 2110 labels), route_effort.py, ops/opsd.py prev_prior()+route(). Eval tools: ops/treeroute_eval.py --sid-only (48 rows: base+prev Hit@1 .646 / Hit@3 .833), ops/recall_replay.py. Labels: ~/.claude/skills/usage.jsonl.
- Windows support 3: backing research Documents/Tools/llm/research/runs/intelligence-autonomy/report.md (index at top; sec 2 Ashby ultrastability, sec 5 act-vs-ask refs, snippet-verified only).
- AJAX chat owns ajax_export.py, ajax_eval.py, data/ajax-*.jsonl/json, eval-ajax-20261026.json, README "Second eval set". Falsifier baseline: bm25 kNN Hit@1 .884 record-LOO vs .448 file-held-out, majority .222 (file split = honest; use this split style for A7).
- GPU 10-27 evening: gpu 1 = DOA review (Strata 8090) until ~20:10; gpu 0 free.
