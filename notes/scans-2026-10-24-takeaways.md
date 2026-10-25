# Takeaways from the two scanned reports (2026-10-24)

Source: the user's printed reports, scanned as `Documents/IMG_20261024_0001..0034.pdf`
(0001-0025 "Modular Cognitive Architecture: Freeform Composition Across Representations",
0026-0034 "Router Experimental Platform - Implementation Roadmap", dated 22 Oct 2026, status
"proposed architecture and experimental plan", no measurements). OCR text: `scans-2026-10-24.txt`
(pages 13, 14, 17 re-OCR'd from the rotated image at the end of that file). Gemma 8083 section
summaries: `scans-2026-10-24-gemma.md`.

## The reports' core stance
- The lexical router (our opsd trigger/BM25 router, Hit@1 0.62) stays the control baseline.
  Add a layer only when the previous one has evidence; every branch is tested against a frozen
  baseline with explicit falsifiers and route/outcome traces.
- Sequence: instrument first -> selective (section-level) retrieval + typed compaction ->
  hybrid retrieval -> graph expansion -> outcome learning -> JEPA last.
- "JEPA is a bounded candidate-fusion correction, not the first or sole router."
- Raw history is not memory: persistent state is typed, versioned, source-linked, governed; only
  observations (not latent predictions) may update the factual state.
- Assumption that flips the roadmap: if section-level fast reading does not cut context
  materially, the retrieval layers lose their reason.

## Proposals vs what we already have
| Proposal | Have | Missing |
|---|---|---|
| Fast reading: rank headers, expand only the needed section | opsd `/api/route` returns a ~300-token brief, skills have `tree/` branches | no measured context-cost-vs-success number; brief vs full-skill A/B |
| Typed context compiler, protection classes (preserve/reference/disposable) | compaction pipeline (PreCompact capture, working.md, value check) | nothing typed: survival of instructions across compaction is not measured |
| Explicit skill graph (REQUIRES / USES / CONFLICTS_WITH), bounded expansion | skills link each other by name in prose | no edges, no "unnecessary activation" count |
| Abstention as 4 states (SUFFICIENT / EXPAND / SEARCH / ABSTAIN) | `gate_soft`, `gate_soft_abstain_head` (binary route/abstain) | expand/search states; no-match accuracy on deliberately unrelated prompts |
| Outcome-weighted training labels | `--labels clean` (drops upkeep edits) | weighting by task outcome, not by what was read |
| Provenance per skill (model, hash, revision, failure log) | git history of `~/.claude/skills` | none in the router |
| Resource Pareto (quality per retained token / ms / VRAM) | usage_report, cap.py | not joined to router decisions |
| JEPA as fusion correction over lexical + semantic candidates | skill-jepa predictor, 0.568 seen-chat Hit@1 | not plugged into the opsd router as a re-ranker |

## Ranked next experiments (cheapest first, each with its pass/fail)
1. Instrument the opsd router: log query, candidates, chosen skill, whether the brief was enough
   (next Read of the full SKILL.md = "insufficient"). Pass: 1 week of route/outcome traces.
2. Brief vs full skill A/B on the trace set: tokens injected and Hit@1. Pass: >=40% fewer tokens
   at equal Hit@1 (the report's "assumption that could flip").
3. No-match set: 100 prompts that touch no skill; measure abstain accuracy of gate_soft vs the
   abstain head vs "always abstain" (0.793). Pass: beats 0.793.
4. Typed compaction survival: tag working.md lines preserve/reference/disposable, count which
   survive 3 compactions. Pass: 100% preserve-class survival.
5. JEPA as re-ranker: fuse skill-jepa energy with the BM25 score on the opsd candidates.
   Pass: Hit@1 > 0.62 on the same label set, no loss on unseen chats.
Later: REQUIRES edges between skills (sequential loading), outcome-weighted labels, provenance.

## 10-24 23:3x: experiment 1 DONE (instrumented)
- ops/route.log now has a 5th column = session id (opsd.route(q, sid); hooks/route_effort.py sends &sid=). ops/route_trace.py joins it with ~/.claude/skills/usage.jsonl and classes each route brief_only / full_read / miss / none+read; daily routine "route-trace" 06:50 (--days 7) -> ops/reports/<day>/route-trace.md.
- Baseline over all 1402 old routes (time-window join, overcounted): brief_only 37%, full_read 33%, miss 30%, Hit@1 0.528 / Hit@3 0.763 among routes followed by a read. With-sid rows start 10-24; read the 7-day report after ~10-31 for the clean number.
- Finding: brief() is not ~300 tokens; top briefs [(13121, 'video-notes', 308), (762, 'ops-manager', 8), (752, 'local-gpu-ai', 13)] tokens (tree 'branches:' line). Experiment 2 must cap branches first.
- User 10-24 (video tree/307, GRPO): "grpo seems interesting for the jepa skill" -> GRPO = the outcome-weighting route: reward from route_trace classes (+1 full/brief hit, 0 none, -1 miss), small local policy, TRL GRPOTrainer on the 3060. After experiment 2.
- 10-24 23:5x: brief cap done (opsd.BRIEF_BRANCHES=8, query-matched + newest; video-notes 13121 -> ~900 tokens per brief, median brief 399). Experiment 2 can now start: after ~10-31 compare with-sid rows before/after 23:5x on tokens injected vs full_read share. Branch matching is word overlap; 'video' matches every video-notes branch, so the tie-break (newest) does most of the work.
