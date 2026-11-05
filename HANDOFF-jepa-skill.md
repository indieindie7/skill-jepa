# HANDOFF: jepa-skill work chat (skill router self-improvement), parked 2026-11-04

## Index
1. Mission - repo, goal, who owns which files (L12-18)
2. State of prod - head, holdout numbers, what is live (L19-24)
3. Results log - supported / rejected experiments with commits (L25-34)
4. Live shadows and nightly jobs - route.log columns, trace_merge (L35-41)
5. Paths and commands - exact files, gates, scripts (L42-50)
6. Open items - what is unfinished, in order (L51-57)
7. Exact next step - first action for whoever resumes (L58-59)

## 1. Mission
- Repo `C:\Users\john\Documents\Tools\skilljepa` (github.com/indieindie7/skill-jepa, branch main). Goal: self-improvement env for the opsd skill router (EmbeddingGemma 8084 + AVB head).
- Full history and pre-registrations: `SELFIMPROVE_PLAN.md` (results lines 165-177; qstd sec 10 line 179; A18b line 187).
- This chat owned: opsd.py route hooks, avb_serve.py, route_trace.py, trace_merge.py, arms.py, wsp_a18.py, astate.py, a6c_holdout.py.
- Peer-owned: shrink.py, a10-a17 scripts, a6c_nudger.py.
- Quality-standard (qstd) work has its own handoff: `HANDOFF-quality-standard.md` (933adba).

## 2. State of prod
- Prod head `data/avb_head.npz` (d256b2), unchanged. Frozen holdout: hit1 .605, hit3 .770, acc_with_abstain .740.
- Holdout = `data/holdout.jsonl` (untracked, NEVER regenerate); check with `py -3.13 -I holdout_check.py --verify` ("holdout hash ok").
- Promotion bar for any change: holdout hit1 >= .600 (prod - .005) AND no per-session drop, then 1 day shadow, then promote.
- Nothing has changed routing behaviour since 10-31; all new items are shadow logs only.

## 3. Results log
- A6c rank-only session nudge: holdout PASS 11-04 (fec2004). w=.3: hit1 .632, hit3 .815, acc_abst .744, 7 sessions none drop. w=.5 post hoc .652 (not pre-registered). Adding the prior to all slots breaks abstain: nudge must be rank-only (gate stays prod pnone). Results `data/a6c-holdout.json`.
- A6c shadow wired 11-04 (6833d27): avb_serve.nud() writes route.log col 10.
- A18 weight-space edit: NOT SUPPORTED (8ba63ec, hit1 .605 -> .610, bar +.02).
- A18b standardized scalars: NOT SUPPORTED (b16cd61, hit1 .615).
- 11-05 keep-or-drop (X0-X9 on the frozen holdout + live probe): learned head DROPPED, nothing promoted; X6 head gate passed a mis-specified bar (lab_cov .42), hybrid X9 failed every F, LR gate + A6c nudge not promoted. opsd `AVB_GATE = False`, shadow col 8 only. PLAN "11-05 keep-or-drop" + two lines after it.
- A11-A17, A6b: not supported (see PLAN 165-176). A15 Grassmann mixer NOT SUPPORTED (PLAN 174).
- ASTATE claim store `astate.py` + `tests/idea_astate_test.py` (6b850b2), idea #144 status review (gain .55).
- compact_check scoped check (`~/.claude/hooks/compact_check.py`): failed real test, opt-in only `COMPACT_SCOPED=1`.
- Ideas: #131 rejected, #124 parked (needs quality ground truth), #141 queued (needs GPU).

## 4. Live shadows and nightly jobs
- route.log `C:\Users\john\Documents\Tools\ops\route.log`: col 8 `avb=` head logits, col 9 `gf=` A14 gate inputs (since 10-31), col 10 `nud=<skill>` A6c pick (since 11-04 ~23:00).
- Col 10 caveat: live history uses usage.jsonl skill reads only (>= 600 s old); the holdout used labelled turns including none. The shadow day measures this gap.
- Nightly 07:05 `trace_merge.py`: embedder retry 6 x 60 s added (bfeacaa) after 8084 was down on 11-03 and 11-04.
- opsd: `pythonw.exe -I C:\Users\john\Documents\Tools\ops\opsd.py` (start hidden). Restart it after any avb_serve.py edit (module is cached).
- Backups: `avb_serve.py.bak-1104nud`, `trace_merge.py.bak-1104`, `compact_check.py.bak-1103astate`.

## 5. Paths and commands
- Holdout score of a head: `py -3.13 -I holdout_check.py --score HEAD.npz`.
- A6c holdout: `py -3.13 a6c_holdout.py 0.3` (writes data/a6c-holdout.json).
- A6c offline replay (peer): `a6c_nudger.py --seeds 3 --min 30 --cold 10 --w 0.3 --temp 0.1 --mindayrows 20`.
- Ideas: `py -3.13 -I ~/Documents/Tools/ops/ideas.py list|set <id> <status> "<note>"`.
- Heavy jobs: `py -3.13 -I ~/.claude/hooks/hoggate.py status`, then `gpu_queue.py req <chat> <job> <gb> <min> gpu=1`; nothing on gpu 0 while a game runs.
- Head format: din 2316 = 3*768 + 12 scalars, K 16.
- Untracked logs a12_run.log, a13b_*.log: leave alone.

## 6. Open items
1. CLOSED 11-05: learned-head line DROPPED (SELFIMPROVE_PLAN.md "## 11-05 keep-or-drop ablation" + the two "11-05" lines after it). A6c nudge judged (n=10 tie, not promoted); LR gate re-tested on the holdout (not supported); head-as-gate X6 passed a mis-specified bar, failed live + hybrid X9. Nothing promoted; opsd `AVB_GATE = False`, shadow col 8 continues.
2. If a gate is ever retried: bar must include lab_cov / lab_correct (X9 pattern in live_gate_20261105.py) and a live probe on canonical skill phrases before promotion.
3. Prospector pre-registration (not started).
4. Q1 qstd reward (needs >= 50 rows in data/qstd_rewards.jsonl; 5 on 11-04); judge qstd hook ~11-10 (see HANDOFF-quality-standard.md).
5. #141 waits for a free GPU via opsd.

## 7. Exact next step
- None on the learned line. Optional cleanup: stop shadow col 8 logging (opsd _avb_log) if nobody consumes it; production stays hand router + session table + NOMATCH .35. opsd must run on Python 3.13 (`pyw -3.13 -I opsd.py`).
