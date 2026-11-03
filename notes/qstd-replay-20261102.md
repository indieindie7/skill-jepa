# qstd replay on real histories (2026-11-02)
Tool: `qstd.py replay REPO` (first-parent, oldest first). Metrics per SlopCodeBench (video-notes tree/442): erosion = share of fn weight (CC*sqrt(LOC)) in fns CC>10; dup = share of normalized lines inside a 6-line window seen >= 2x; crossed = fns that went from CC<=10 (or absent) to CC>10. Gate line: rise > .005 or any crossed fn.

| repo | commits | flagged | erosion first -> last | worst commit (dEro / dDup / crossed) |
|---|---|---|---|---|
| Tools/skilljepa | 75 | 38 | .320 -> .835 | 9b3de4c +.137/0/1; c6aeb10 +.010/+.059/2 |
| Tools/playtest | 8 | 3 | 0 -> .5+ | 10555ee +.503/0/2 |
| github/water-gym | 79 | 22 | - | e1ee90e +.472/0/3 |
| github/manxtt-decomp | 11 | 3 | - | 91b1176 +.164/0/3 |
| github/scorcher-decomp | 13 | 3 | - | no erosion/dup rise (crossed only) |
| github/codes (last 120) | 120 | 30 | ~.71-.73 | 973ebda +.004/0/14 |

Known-bad checks:
- skilljepa c6aeb10 (A6c nudger) = copy-patch of a6b_nudger.py: the pair has dup .852, a6c main() CC 83. Gate flags it on dup (+.059) and 2 crossed fns. The exact SlopCodeBench pattern (fresh copy instead of reuse).
- playtest 10555ee (local model writes reactions.py + tickets.py, tests 17/17 green) flagged erosion +.503; the NEXT commit c773c6f is "applies reviewer fixes (case-insensitive word match, empty target, None loc, dup ids)" = bugs the green suite missed.
- water-gym foam rounds 6/8 (f890267, 9b996ee) flagged erosion+dup; later commits b179425/d188064 revert foam defaults.
- skilljepa erosion rose .32 -> .835 over 75 commits (one-off experiment scripts with big main()); 38/75 commits would have failed.
- Big repos (codes, 700+ files): repo-wide erosion share barely moves (+.004) even when 14 fns cross CC 10 -> the `crossed` rule carries the gate there, not the share.

Self-test: qstd.py's own first draft FAILED its gate (6 fns crossed CC10: gate 23, files_at 12, hook 11, main 11); split into helpers -> PASS, replay numbers unchanged.
Raw replay tables: %TEMP%/claude-cap/replay_*.txt (not kept).

## 2026-11-03 cognitive complexity trial (qstd.py cog <repo>)
- SonarSource cognitive complexity (nesting-weighted) vs McCabe CC, Spearman over all fns:
  - skilljepa 316 fns rho .988 (cc>10: 63, cog>=15: 58); top cog-only: eval_all.py::table cc10 cog24, ajax_eval.py::evaluate cc9 cog19.
  - playtest 59 fns rho 1.0 (5 vs 5); cog-only tickets.py::merge cc10 cog17.
  - water-gym skipped (Rust, metric is Python-only).
- Verdict: on our code cognitive ~= CC; gate stays CC10. `cog` stays a report for spotting deep nesting.

## 2026-11-03 falsifier suite (test_qstd_falsifiers.py, 14 tests x 12 seeds on generated code)
- One test per real falsifier: c6aeb10 copy-patch, SlopCodeBench forge main() growth, codes big-repo crossed rule, playtest 10555ee green-narrow-suite = UNTESTED, delete rewarded, qstd gate() CC23 split, water-gym revert neutral, broken file, reward order; invariants: rename, reorder, comments/indent, file move, <6-line overlap.
- Found and fixed 2 gate holes: (1) a file that stops parsing dropped out of the weights -> erosion FELL -> PASS; now d_parse>0 = FAIL. (2) moving/renaming a file with a CC>10 fn was reported as `crossed`; now same qualname+CC elsewhere in base = moved, not crossed.
- Dropped one wrong invariant of mine: removing spaces around `=` changes normalized lines (norm collapses whitespace runs, does not delete them); only re-indent/extra spaces are neutral.
- .qstd_test now runs both files (unittest discover test_qstd*.py). Self-gate PASS; judge/scan/hook sit at CC 10 exactly (limit is >10).
