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
- A1 Closed loop = confirmation bias. Evidence: router==label .867, fresh judge==router .233 (judge_fresh.py, 30 rows). Wrong if: judge disagrees equally on context-free prompts (then gap = missing context, not bias). Test: split judge_disagree.jsonl by prompt context-dependence (Sage tag), compare agree rates. Result 10-27 (a1_split.py, 79 rows): judge==router free .306 (n36) vs ctx .116 (n43); judge says none .306/.512; router==label .889/.907. Verdict WEAKENED: ~half the gap = missing context, but .306 on context-free rows still = bias (judge agrees w/ router on 3 of 10 even with full info). Keep A1 as "bias + context", nudger must carry context bits, not only novelty.
- A2 Low surprise = time to change rules (Ashby 2nd loop). Wrong if: rule changes after a low_surprise flag give no hit1/falsifier gain. Test: when flag fires, swap one rule (K, cand source, data mix), 3 nights A/B. Result: flag needs 7 runs, not fired yet.
- A3 5% exploration slot surfaces useful unseen skills. Wrong if: explore_hit_rate stays ~0 after 14 d, or hits are only skills already in top-5. Test: trace_state history explore_hit_rate; count explore hits not in top-k. Result 10-27 (nudger_measure.py, 1866 rows): 0 live explore rows yet; retro least-routed-14d candidate read in window .0091 < uniform random .0136 -> least-routed policy is WORSE than random; reads outside top-3 happen on .593 of read rows, dominated by meta skills (ops-manager 180, local-gpu-ai 157). A3 as built: falsified; JEPA chat switched explore_pick to seeded uniform random (commit 3048d2b, opsd restarted 10-27). Usual-suspects test 10-27 (nudger_measure.py, causal 14 d, candidate = skill most often read outside top-3, not in row top-3): global .148 (262/1769), per-chat .182 (280/1539), vs random .0136, least-routed .0091; without meta skills ops-manager/local-gpu-ai: .076 / .101 (picks goobi-brawler, swat4-mod). 7-13x random, but it is a systematic-miss readout (exploitation), not exploration -> proposal: split the slot: 'suspect' (per-chat usual suspect, measures router blind spots) vs 'explore' (random, measures true novelty); never mix their hit rates.
- A4 Refute templates carry information. Wrong if: refuted rate == miss rate (template just restates hit/miss). Test: corr(refuted, class miss) over 200 rows; if |r|>.9 templates are redundant, need model-written refutations. Result 10-27: FALSIFIED (retro over 1866 rows): refuted is a function of (conf,class) in 11/13 cells; the 2 mixed cells are full_read + a second skill read (372/567), which is not a refutation of top-1. Sign check 10-27 (JEPA chat): NOT a bug; col 7 = the refuting EVENT, refuted=True means it happened (low + nothing read = route was useless). JEPA chat: falsifier_score = calibration readout only (plan + docstrings).
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
- 10-27: JEPA chat ACCEPTED split slot, building (Sonnet): col 6 explore=<skill> (5% random) or suspect=<skill> (5% per-sid usual suspect, global fallback); classes suspect_hit/miss apart; suspect_hit_rate separate in trace_state. LIVE 661d362 pushed, opsd restarted; global fallback when sid < 3 reads, 10-min cache, meta skills not excluded; first rows 10-28 07:05.
- NEXT: (1) DONE by JEPA chat: explore=random 3048d2b, refuted sign kept as designed; judge_fresh on explore rows offered to this chat; (2) DONE 10-27: usual-suspects retro per-chat .182 vs random .0136 (sec 2 A3); proposed split slot to JEPA chat; (3) DONE 10-27 A1 WEAKENED (sec 2 A1; a1_split.py e29d3e2).
- 10-28 10:07 A3 live readout too early: 4 explore / 2 suspect rows since 661d362, 0 hits each (nudger_measure.py suspect line added). Re-read when >= 30 rows each (~1 week at current rate).

## 6. Peer chats' holds + inputs (10-27 replies)
- Tree-router chat: read-only for me: ~/.claude/hooks/recall.py (BIG_TREE filter), recall_judge.py (+recall-judge.json, 2110 labels), route_effort.py, ops/opsd.py prev_prior()+route(). Eval tools: ops/treeroute_eval.py --sid-only (48 rows: base+prev Hit@1 .646 / Hit@3 .833), ops/recall_replay.py. Labels: ~/.claude/skills/usage.jsonl.
- Windows support 3: backing research Documents/Tools/llm/research/runs/intelligence-autonomy/report.md (index at top; sec 2 Ashby ultrastability, sec 5 act-vs-ask refs, snippet-verified only).
- AJAX chat owns ajax_export.py, ajax_eval.py, data/ajax-*.jsonl/json, eval-ajax-20261026.json, README "Second eval set". Falsifier baseline: bm25 kNN Hit@1 .884 record-LOO vs .448 file-held-out, majority .222 (file split = honest; use this split style for A7).
- GPU 10-27 evening: gpu 1 = DOA review (Strata 8090) until ~20:10; gpu 0 free.
- 10-28 input (Windows support 3): video-notes tree/363 Liquid AI LFM3 2B family; D1 2B decision model = JSON + confidence -> candidate abstain/falsifier head, cheap vs Sage 9 GB; parked idea #120 (ops/ideas.py), no test yet. Would fit A6 nudger as the small-space judge.
