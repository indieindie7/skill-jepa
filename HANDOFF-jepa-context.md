# HANDOFF: jepa-context (context-nudger falsification chat), parked 2026-11-04

## Index
1. Mission: falsify loop-breaking/nudger assumptions (lines 14-18)
2. Partners + ownership: who owns which files (lines 19-23)
3. Repo, paths, commands (lines 24-33)
4. Results ledger: tests, verdicts, commits (lines 34-54)
5. Decisions (lines 55-60)
6. Open items (lines 61-68)
7. Next step (lines 69-71)

Draft note: local Sage draft failed (returned "Nothing relevant found"); written from files, every commit id checked with `git log -1`.

## 1. Mission
User (10-31): "the goal will be for enriching and falsifying assumptions for breaking agents from their core loops ... giving autonomy and creativity for the models while being a small nudger that uses small space."
Method: pre-register each assumption (pass rule + "wrong if") in the ledger, run offline replay/holdout test, write RESULT on the same line, commit.
Parked 11-04 by user: "lets just park the jepa research we were doing, ask them to write handoffs and like archive them".

## 2. Partners + ownership
- JEPA-skill work chat local_846d63fc-a1a5-4fc6-8019-1f03a55e940f (itself parked 11-04, see HANDOFF-jepa-skill.md): owns ops/opsd.py route(), route_trace.py, route.log, data/*.jsonl, nightly routines, avb_serve.py. It pushed this chat's commits.
- Jepa-user chat local_8ab44135-6ba0-414e-8c05-d0bc4850144f: owns A17u / T3k kernel, a17_user_profile.py.
- Rule: never edit the partners' files; data/*.json is gitignored, never commit it.

## 3. Repo, paths, commands
- Repo: C:\Users\john\Documents\Tools\skilljepa (public github.com/indieindie7/skill-jepa). No push from this chat.
- Ledger: notes/context-nudger-PLAN.md section 2 (A6 line 23, A13 32, A14 33, A13b 34, A16 35, A17 36, A17u 37, A6b 38, A6c 39).
- Torch python: C:/llm/qlora/venv/Scripts/python.exe
- Scripts: a17_svts.py (gen -> data/a17-prompts.json, 300 prompts), a6b_nudger.py, a6c_nudger.py (= a6b + arm N3), a17_user_profile.py (peer; exports table_probs, norm).
- A6c outputs: data/a6c-results.json, notes/a6c-table-raw.md, log data/a6c_run.log; embedding cache data/a6b-emb.npz.
- Notes: notes/a17-svts-20261031.md, notes/a6b-nudger-20261031.md, notes/a6c-table-raw.md.
- Heavy runs: check `py -3.13 -I ~/.claude/hooks/hoggate.py status` first; GPU preferred.
- Working notes: C:\Users\john\.claude\compact-captures\91ce318d-dbb1-4300-921e-6638fed36b16\working.md

## 4. Results ledger
| Test | Verdict | Commit | Key numbers |
|---|---|---|---|
| A1 | WEAKENED | | |
| A3 explore slot | FALSIFIED as built (least-routed worse than random .0091 < .0136) | | live readout pending |
| A4 refute templates | FALSIFIED | | refuted = f(conf,class) in 11/13 cells |
| A9 AVB rerank OOD | NOT REJECTED | d985ca7 (JEPA chat) | hit1 .412->.490 time, .262->.328 session |
| A10 frontier recall | FALSIFIED | | loss 13.9 pts at K=10 |
| A11 compact state | NOT SUPPORTED | bf2168e | J0 .468 / J1 .499 / J2 .500 |
| A12 risk-coverage | NOT SUPPORTED | 3fffa67 | C1 head thr best |
| A13 self-shrink | PARTIAL / NOT REJECTED | | d16b0 45x smaller, time ok |
| A14 learned abstain | NOT SUPPORTED | fb3bb5c | |
| A13b shrink retry | NOT SUPPORTED | 58cfab1 | d16b0 holdout .595 < .600, keep d256b2 |
| A16 DR-Zero proposer | FALSIFIED | 9068304 | R2 .551 <= R1 .562 <= R0 .566 |
| A17 SVTS auto-gold | FALSIFIED | 0fdc391 | Spearman auto vs holdout -.57 over 16 variants |
| A17u per-chat profile (peer) | TABLE WINS | 148d892 | T1 56.8 / T3 51.8 / T0 32.9 warm; T3k warm 63.0, perm 46.5 |
| A6b per-chat prior nudger | NOT SUPPORTED | 91190d2 | hit1 31.1 -> 62.9, entropy 2.61 -> 2.48, KL med 1.10 (cold table 5.79) |
| A6c base cold + T3k warm | SUPPORTED | c6aeb10 | hit1 54.6 (+23.5), entropy 2.75 > 2.61, KL med .207 bits |
| A6c frozen holdout (JEPA chat) | PASS rank-only | fec2004 | hit1 .632 vs .605, no session drop |
| A6c shadow logging (JEPA chat) | live, log only | 6833d27 | route.log col 10 nud=<skill> |

## 5. Decisions
- Synthetic prompts are neither training data (A16) nor a valid proxy eval (A17); use real labelled rows only.
- Keep prod head d256b2 (A13b).
- Nudger = per-chat T3k kernel residual only once warm (>= 10 labelled rows); base router while cold. A cold per-chat table narrows entropy (echo chamber).
- A6c is a rank-only nudge, shadow-logged in avb_serve, not routed live yet.

## 6. Open items
- A6c: promote from shadow (col 10) to live route once enough shadow rows agree; JEPA-skill owner's call.
- A3: live explore readout needs n >= 30 rows per arm (was suspect 19/1 hit, explore 20/0).
- A5, A7, A8: PENDING in ledger.
- Optional A17 retry with a real LLM paraphrase instead of the template rewrite (no GPU then).
- Papers to tie in: byte/EOT student (video-notes tree/370), LFM3 D1 (tree/363).
- Untracked logs in repo root (a12_run.log, a13b_*.log): leave or delete, not committed.

## 7. Next step
Nothing running. Whoever resumes: read ledger section 2 lines 32-39, then check shadow col 10 in route.log against labels to decide A6c promotion.
