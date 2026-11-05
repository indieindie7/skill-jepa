# HANDOFF-quality-standard.md (skill-jepa side chat "quality standard", 2026-11-02..04, PARKED 11-04)

Receivers: read ONLY this index first. Open one section by its line range only when the current step needs it.
Values below were checked against `git log`, `ls` and the files on 2026-11-04; the draft was assembled from saved notes (working.md, compact captures, SKILL.md, notes/), not from chat memory.

## Index
1. Mission and status: code-rot gate shipped, hook live, parked (lines 16-22)
2. Deliverables: paths, commits, what each does (lines 23-34)
3. How the gate works: verdicts, thresholds, CLI, exit codes (lines 35-45)
4. Global hook trial: settings.json entry, backup, judge rule (lines 46-51)
5. JEPA tie-in: reward log, SELFIMPROVE_PLAN sec 10 Q1 pre-reg (lines 52-57)
6. Decisions and findings: cog trial, falsifier holes, verdict "silver" (lines 58-65)
7. Gotchas met on the way (lines 66-72)
8. Open items and exact next step for whoever resumes (lines 73-79)

## 1. Mission and status
Side session of skill-jepa local_846d63fc-a1a5-4fc6-8019-1f03a55e940f. Task: a SlopCodeBench-style code-rot gate for agent code changes (regressions, duplication, complexity erosion), as a tool plus a skill, wired into the JEPA reward loop.
Status 11-04: tool, tests, skill, memory, hook and reward log are DONE and pushed to skill-jepa `main` (last qstd commit d628980). User parked the JEPA research 11-04 ("lets just park the jepa research ... write handoffs ... archive them").
What keeps running without anyone: the PostToolUse hook (sec 4) and the reward log (sec 5). Both are harmless to leave on; sec 8 says when to judge them.
Verified claims: `py -3.13 -I -m unittest discover -s . -p "test_qstd*.py"` in skilljepa -> 22 tests OK (11-03); `py -3.13 -I qstd.py gate .` -> PASS on skilljepa itself.
Not verified: the hook's catch/noise ratio on real edits (needs a week of use); Q1 (needs >= 50 reward rows, 5 now).

## 2. Deliverables
All in `C:/Users/john/Documents/Tools/skilljepa/` unless noted.
- `qstd.py` (339 lines, stdlib only): metrics (erosion share, duplicate-window share, fns over CC 10, parse errors), `compare`, `judge`, CLI `gate|diff|scan|replay|cog|hook`.
- `test_qstd.py` (75 lines, 8 tests) + `test_qstd_falsifiers.py` (138 lines, 14 tests x 12 seeds on generated code). Runner: `.qstd_test` = `py -3.13 -I -m unittest discover -s . -p "test_qstd*.py"`.
- `notes/qstd-replay-20261102.md` (34 lines): replay on 6 repos, self-test, cog trial, falsifier suite results.
- `SELFIMPROVE_PLAN.md` line 179 `## 10. Quality reward: qstd code-rot gate` (Q1 pre-registration).
- `data/qstd_rewards.jsonl` (UNTRACKED, 5 rows): reward rows appended by `gate`.
- Skill `~/.claude/skills/quality-standard/SKILL.md` (36 lines: Tool, Rules 1-10, Hook, JEPA tie-in, Branches) + `tree/01-standards-survey.md` (SOLID/DRY/KISS/YAGNI/Ousterhout/CUPID survey).
- Memory `~/.claude/projects/C--Users-john-Documents-Tools-skilljepa/memory/quality-standard-qstd.md`.
- Commits on skill-jepa main (oldest first): f557e22 gate + replay + plan sec 10; 450b95d 1-line output; 516015c cog cmd; 521d35b falsifier suite + 2 gate holes; 12b4576 reward log default on; d628980 hook warns on non-parsing file.
- Related video notes: `~/.claude/skills/video-notes/tree/442` (SlopCodeBench, the source), 427 (overengineering limit), 447 (Casey Big OOPs), 448 (Eon Amiga constraints).

## 3. How the gate works
- `py -3.13 -I qstd.py gate <repo> [--test "<full suite cmd>"] [--accept] [--allow-cc] [--no-log] [--json]`. Suite cmd also read from `<repo>/.qstd_test`.
- Baseline = last accepted snapshot `.git/qstd_accepted.json` (written by `--accept`), else HEAD. Working tree is compared against it.
- Verdicts: PASS exit 0; FAIL exit 1; UNTESTED exit 2 (no suite given or found). UNTESTED is never PASS.
- FAIL when: suite fails; erosion share or dup share rises > .005 (`THR`); any fn crosses CC 10 (`crossed`, limit is > 10, CC 10 passes); any file stops parsing (`d_parse > 0`).
- A fn with the same qualname+CC that existed in base under another path counts as moved, not crossed.
- Dup window `WIN` = 6 normalized lines; comments/blank lines/indent are neutral, spacing changes around operators are not.
- Output: one verdict line, plus one compact JSON line on FAIL; `--json` for the full dict. Reward in judge(): PASS > UNTESTED > FAIL, bigger erosion rise = lower reward.
- Other commands: `diff <repo> A [B]`, `scan <paths>`, `replay <repo> [--n N]`, `cog <repo>` (SonarSource cognitive complexity vs CC, report only), `hook` (reads PostToolUse JSON on stdin).
- Big repos (700+ files): the erosion share barely moves; trust `crossed`, not the share.

## 4. Global hook trial
- `~/.claude/settings.json` PostToolUse entry (line ~117): matcher `Edit|Write`, command `py -3.13 -I "C:/Users/john/Documents/Tools/skilljepa/qstd.py" hook`, timeout 15. Backup before the change: `~/.claude/settings.json.bak-1103qstd`.
- Behaviour: one stderr line (exit 2) when an edited `.py` has a fn crossing CC 10, dup +.02 vs its HEAD blob, or no longer parses; silent otherwise. Non-`.py` and new-to-git files are compared against empty.
- Live since 11-03 (user: "Yes trial everywhere"); verified live with a CC-13 scratch file -> "qstd hook_live.py: fn crossed CC10: f -->13 -> split into helpers".
- Judge rule (SKILL.md Hook section): after a week of real edits count catches vs noise -> keep, tune thresholds, or revert by restoring the backup.

## 5. JEPA tie-in
- `gate` appends a reward row to `skilljepa/data/qstd_rewards.jsonl` by default since 11-03 (`--no-log` skips). File stays untracked.
- SELFIMPROVE_PLAN.md sec 10 Q1 (pre-registered 11-02): H1 = prompts routed to and reading quality-standard get a lower FAIL share than unrouted ones. Measure only at >= 50 rows. 5 rows on 11-04.
- Router: `ops/skillc.py` index rebuilt 11-03 so "refactor done regression code quality" routes to quality-standard (score 3.548).
- Not wired (ideas only, user asked 11-03): gate as rejecter for local models (retry <= 2 on FAIL), best-of-k drafts, DPO on PASS/FAIL pairs once hundreds of rows exist. Goodhart risk noted: old suite = truth, metrics = proxy.

## 6. Decisions and findings
- Gate stays McCabe CC > 10. Cognitive complexity (`cog`) correlates Spearman .988 on skilljepa (316 fns) and 1.0 on playtest (59 fns); kept as a report for deep nesting only. water-gym skipped (Rust).
- Replay (notes file): real misses that became falsifier tests: skilljepa c6aeb10 copy-patch (dup .85); SlopCodeBench forge main() growth; codes repo 700+ files where 14 fns crossed while share moved +.004; playtest 10555ee green narrow suite then 4 bugs; water-gym revert neutral.
- Two gate holes found by the suite and fixed in 521d35b: a file that stopped parsing dropped out of the weights so erosion fell and read as PASS (now d_parse FAIL); a moved file with a CC > 10 fn read as crossed (now moved).
- qstd.py's own first draft failed its gate (gate() CC 23); split into helpers. judge/scan/hook/fns.walk sit at CC 10 exactly.
- Skill verdict given to the user 11-03: "silver", golden only after real use (>= 50 reward rows, hook judged).
- Rules the user set here: caveman = speech only, think high (CLAUDE.md CAVEMAN line); rule 2 limit: a few duplicate lines beat a one-caller abstraction (tree/427); subagent briefs = numbered steps; on FAIL fix and re-gate up to 2 rounds without asking.

## 7. Gotchas met on the way
- `ops/skillc.py` dies on a 0-byte skill JSON (`ops/skills/hydrophobia-mod.json` was empty); delete the file and recompile. Logged in ops-manager tree/12.
- Git-Bash `/c/...` paths fail `os.path.isfile` on Windows; use `cygpath -m`. `$(...)` strips the trailing newline from hook stdin tests.
- Removing spaces around `=` changes normalized lines (norm collapses whitespace runs, does not delete them); only re-indent/extra spaces are neutral.
- localread.py handoff draft failed 11-04 23:46: three Sage llama-servers on 8093 started at once by parallel handoff chats, both GPUs full, ConnectionResetError. Run one handoff at a time.
- Hook feedback shows up as PostToolUse stderr; the Edit/Write itself still succeeds.

## 8. Open items and exact next step
Parked by the user 11-04; nothing is requested. If someone resumes:
1. Hook judge (due ~11-10): grep stderr catches in recent sessions vs false alarms; keep, tune (`THR`, dup +.02, CC 10 in qstd.py), or revert via `settings.json.bak-1103qstd`.
2. Q1: when `wc -l data/qstd_rewards.jsonl` >= 50, run the sec 10 analysis (FAIL share routed vs unrouted) and write the verdict into SELFIMPROVE_PLAN.md sec 10 and notes/.
3. Optional: wire `gate` as a rejecter in the local-model autorun loop (retry <= 2), then best-of-k; see sec 5.
4. Any new gate miss: add a seeded test to `test_qstd_falsifiers.py` (FAIL case + PASS control), fix qstd.py, run `.qstd_test`, self-gate `py -3.13 -I qstd.py gate .`, commit with explicit file names.
Done criteria for the parked state: this file committed on skill-jepa main (not pushed), peer "Windows support 3" (local_ef5a917c-1733-4bc8-b174-d6fc85a3d8f5) told the path, chat archived.
