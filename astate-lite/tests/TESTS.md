# ASTATE-lite test campaign (2026-11-05)

Index (read one section by its line range; numbers live in results/*.json, this file quotes them):
1. Verdict — what holds, what breaks, lines 15-30
2. Harness — files, how to rerun, lines 31-42
3. Unit + stress — 78 tests, concurrency, cap, speed, fuzz, lines 43-56
4. Gate benchmark — deterministic gate vs the 1237-row holdout, lines 57-74
5. Dogfood — 8 Sonnet sessions on a fake project, lines 75-92
6. GAP-1..10 — where the skill text under-specifies and what the impl chose, lines 93-110
7. Adversarial review — 6 skeptics, ranked findings, lines 111-148
8. Untested / next, lines 149-165
9. Fix round 11-05 — GAP-1/4/7/8 fixed, s8 checks 1-4 run, 2 skeptics, provenance decision, lines 166-189
10. Round 2 11-06 — s8 items 5-10 run, budget abstention 3/3, 9 skeptic fixes + 2 lock bugs, GAP-11 fixed 11-06, lines 190-235

## 1. Verdict
The skill's core loop works when a mechanical regenerate exists: cold test 5/5 from STATE alone (s2), gate decisions 3/3 (s4),
abstain on a secret in 2 steps with a blocker logged (s5), LOG-first fix with append-only intact (s6), metric line correct (s8).
It breaks where the prose relies on the agent instead of a mechanism:
- "STATE derived from LOG" names no procedure. 3 of 8 sessions wrote STATE by hand; claims drifted from LOG, status fields grew
  qualifiers ("confirmed (not run)"), and on a 300-line LOG the session compressed 40 claims into 3 ranges (0 of 34 reference
  confirmed claims survive) and left a closed blocker open.
- The ~2k-token cap and "never drop a confirmed claim" cannot both hold past ~60 claims (cap_400_claims: 39 confirmed lost).
- Append-only LOG with several writers needs a file lock (unlocked 8x500 appends lost 692 of 4000 lines on Windows).
- The deterministic gate as written (trigger stated in the message) over-loads: no variant beats always-abstain on the holdout.
  Requiring a confirmed STATE claim as well raises precision from .045 to .394 but still loses to abstaining.
Keep the skill; ship it with the reference impl (regen/check/gate commands) and treat sections 6-8 as the fix list.
Fix round 11-05 (s9): GAP-1/4/7/8 fixed, s8 checks 1-4 run (race, idempotence, provenance, cold 6/6 sessions 7/7), 59 tests green.
Round 2 11-06 (s10): s8 items 5-10 run; budget abstention 3/3 Sonnet sessions (blocker format 1/3 clean, GAP-11); 9 skeptic
findings + 2 append_log lock bugs fixed; 78 tests green; the gate bench rerun with CIs leaves the s4 verdict as it was.

## 2. Harness
All in ~/.claude/skills/astate-lite/tests/ (SKILL.md wording was never edited):
- astate_lite.py (~730 lines): reference impl. CLI `regen <log> <state> [--project X] [--date D]`, `check <root>`, `gate <contract> <state> <message>`.
- test_astate_lite.py (~730): 78 unittest tests (Grammar, Regen, Gate, Files, Fixes, Skeptics, Stress, Round2, Round2Skeptics, Gap11) -> results/unit.json.
- gate_bench.py (97): gate variants vs skilljepa/data/holdout.jsonl -> results/gate_bench.json.
- make_lanterna.py (237): dogfood fixture (fake project "lanterna", copies s1..s8, truth.json).
- dogfood_check.py (169): scores the sessions' .state files -> results/dogfood.json.
- race_test.py, provenance_check.py (s9): results/race.json, results/provenance.json; cold fixtures: scratchpad build_cold.py.
Rerun: `py -3.13 -I -m unittest test_astate_lite -v` (4 s); `py -3.13 -I gate_bench.py`; `py -3.13 -I make_lanterna.py <dir>`,
run one Sonnet session per copy with the prompts in the workflow script (session dir workflows/scripts/astate-lite-dogfood-*.js),
then `py -3.13 -I dogfood_check.py <dir> <agents.json>`.

## 3. Unit + stress (results/unit.json)
78 tests, 0 failures, 0 errors, 5.11 s (47 at campaign time, +12 in the s9 fix round, +18 in round 2, +1 Gap11 11-06, s10).
| stress key | result | meaning |
|---|---|---|
| concurrent_8x500_appends | unlocked lines 3308 lost 692 torn 0; locked lines 4000 lost 0 | append-only needs a cross-process lock (GAP-10); impl locks with msvcrt/fcntl |
| cap_400_claims | kept 61 dropped 339 tokens 1971 confirmed_lost 39 | the 2k cap drops confirmed claims once a project passes ~60 claims |
| regen_100k_lines | 0.3 s, 20000 distinct claims, kept 79 | regenerate is linear after the one-sort trim (first version was O(n^2), 304.8 s) |
| gate_2000_contracts | 0.835 s, loaded 998 | gate cost is negligible |
| mutation_fuzz_5k | still_parses 2054 rejected 2946, no crash | one-char mutations of valid lines: the parser is strict but never throws |
| fuzz_10k_lines, contract_fuzz_5k | 0 accepted | random text never parses: these two tests only prove no crash (weak) |
| idempotence_15_cycles (s9) | cycles 15, log_lines 75, final_confirmed 6, cycles_with_change 7 | regen(regen(x)) == regen(x) after every cycle; confirmed set == derived set |
Grammar tests pin: claim line round-trip, tags SRC/SYN/ø, 4 statuses in STATE and LOG (GAP-1, fixed s9), `|` inside "what" kept by
right-split (GAP-3), untagged fact -> [SRC] with a source-pointer evidence else [ø] (GAP-4, s9), metric line parse, contract header under `metadata:` only.

## 4. Gate benchmark (results/gate_bench.json)
Data: skilljepa/data/holdout.jsonl, 1237 prompts in chat order, 400 labelled with a skill, 837 "none". Trigger table: frozen
a0_index.json (2026-10-29). "Confirmed STATE claim" simulated per chat from earlier labelled rows.
| variant | hit1 | acc_abst | fr_none | loads | precision |
|---|---|---|---|---|---|
| always_abstain | 0 | .677 | 0 | 0 | - |
| name (skill named in message) | .007 | .672 | .011 | 12 | .25 |
| strong (trigger weight >= .8) | .060 | .377 | .472 | 536 | .045 |
| two_items | .020 | .534 | .221 | 251 | .032 |
| any trigger | .275 | .203 | .832 | 1055 | .104 |
| strong_or_state | .188 | .390 | .513 | 623 | .12 |
| state_and_any (STATE claim AND trigger) | .255 | .653 | .157 | 259 | .394 |
Reading: the skill's gate ("every requires item confirmed in STATE or stated in the message") has no threshold, so any
trigger-only reading loads far too often. Adding the STATE condition is the only variant near the abstain baseline.
Baseline note: acc_abst .677 here = share of "none" rows; the skill-jepa README's .793 used a different accuracy definition
(labelled rows weighted), so the two are not comparable. The A0 hand router at NOMATCH could not be read from
ablate-20261105.json (no X0 key), so no router-vs-gate comparison is claimed.

## 5. Dogfood (results/dogfood.json, fixture make_lanterna.py)
Fake project: README says port 8080, config/app.toml says 9090 (code reads config), logs at var/log/lanterna.log, deploy owner
Rui, secrets only in a vault. 8 Sonnet sessions, each told to read SKILL.md + templates and follow them; no program execution.
| session | task | mechanical | substance |
|---|---|---|---|
| s1 open/work/close from nothing | find start + logs | FAIL (1 off-grammar claim, 6 claims without LOG line, first question not "How do I open X?") | right: port 9090 with README conflict, logs path, procedures |
| s2 cold read of s1 STATE only | 5 standing questions | PASS 5/5 | - |
| s3 port question | both sides kept | FAIL (LOG line 4 malformed, 2 status qualifiers) | right: conflict line, README marked stale hypothesis |
| s4 gate, 3 contracts | load/abstain | PASS 3/3 agree with impl (log-reader yes, deployer no, port-checker malformed) | - |
| s5 secret request | abstain | PASS (2 explore steps, blocker logged, no value invented) | metric line written with 3 fields, not 5 |
| s6 "just fix STATE" | LOG first | PASS (1 LOG line then STATE, append-only held) | - |
| s7 regenerate from 300-line LOG | cap | FAIL (0/34 confirmed kept textually, closed blocker still open, 7 off-grammar claims, +1 LOG line) | lossy: 40 facts -> 3 range claims, 637 tokens vs reference 1231 |
| s8 weekly metric | metric line | PASS | - |
Mechanical 5/8, substance 7/8. Pattern in every failure: the session hand-wrote STATE. Where the impl's regen was the
procedure (seeded copies s4-s6, s8) nothing drifted.
Caveat: s2's 5/5 reads a STATE that s1 wrote by hand (richer than a regenerated one would be), so the cold test is an upper
bound; the harness forbade running programs, so no session could compute sha256 receipts.

## 6. GAP-1..10 (choices in astate_lite.py, grep "GAP-")
1 LOG has no "rejected" status (STATE does) — fixed 11-05 (s9): LOG accepts the 4 STATE statuses; SKILL.md:38 lists rejected,
  only templates/LOG.md omitted it (template fixed too).
2 "tokens" without a tokenizer — impl: (len+3)//4.
3 No escaping of `|` in "what happened" — impl: right-split keeps the first 2 and last 2 fields.
4 Untagged fact — fixed 11-05 (s9): [SRC] when evidence is a source pointer (file.ext:N, #LN, "line N", Makefile:N), else [ø].
5 LOG -> STATE section mapping undefined — impl: prefixes conflict:/procedure:/question:/receipt:/drop:; a blocker closes
  when a later same-text blocker line has a status other than unknown. Dogfood s7 read that differently (kept it open).
6 Cap drop order undefined — impl: rejected -> unknown -> hypothesis -> confirmed, oldest first; never drops conflicts,
  procedures, blockers, questions, receipts. Consequence: confirmed claims do get dropped (section 3).
7 "is a confirmed claim" / "stated in the message" undefined — fixed 11-05 (s9): exact normalised claim (trailing .;:! ignored)
  or whole phrase in the message ('.' inside a token continues it). A negated mention still counts: spec limit.
8 Empty `requires` — fixed 11-05 (s9): never loads.
9 Two passing skills, `conflicts` — impl: given order, later one blocked by name or by an effect of an earlier loaded one.
10 Append-only with several writers — impl: file lock (needed, section 3).
Also seen in dogfood, not numbered in the impl: status qualifiers in parentheses (s1, s3, s7), metric line without
evidence/status (s5), "nothing else" vs the close line every session appends (s7).

## 7. Adversarial review (6 Sonnet skeptics, workflow wf_fb224081-af2, raw: scratchpad adversarial.json)
Lenses: literal wording, contradictions, promises vs numbers, impl GAP-1..5, impl GAP-6..10 + harness, completeness.
Their verdicts: the skill names outcomes but no procedures (every dogfood failure maps to an undefined term); several rules
cannot all hold (cap vs derive, derive vs "drop what no longer matters", gate vs "no threshold"); the harness is partly
self-confirming. Findings merged and ranked; claims with a cited line were checked against the files.
High:
- H1 Derivation has no mechanism (SKILL.md:22,30). No LOG-to-STATE mapping, no merge rule. The impl needs the prefix convention
  (GAP-5) that the skill never states; s4 passes only because its seed LOG already used those prefixes; s1/s3 LOGs have none, so
  regen would yield no Conflicts/Procedures from them. Fix: mandate the regen script, one LOG line = one claim, latest wins,
  never merge; forbid hand-written STATE; make `check` a required close step with a provenance rule (every STATE claim has a
  LOG line).
- H2 Cap vs recall (SKILL.md:19 vs :30). Drop order undefined; the impl drops confirmed claims silently (39 lost at 400). Fix:
  never drop confirmed silently: spill to an overflow file with a pointer or fail the close loudly; report drops in the metric
  line. The ~60-claim cliff depends on the chars/4 estimator; report chars/3 too.
- H3 Gate safety (SKILL.md:34,52). The skill's own table concedes always-abstain .793 is unbeaten; the bench says the same for
  the gate (.677 on the holdout) and the best variant's precision .394 means about 6 of 10 loads are wrong. GAP-7 substring
  either way lets a short requirement match a longer claim, negations included; GAP-8 empty `requires` loads, against "No match:
  do the task without a skill". Fix: exact normalised claim match, empty requires never loads, abstain by default, load only on
  an explicit skill name or STATE claim + trigger; define the tie rule for two passing skills that conflict (GAP-9 is invented).
- H4 Status grammar (SKILL.md:38 vs templates). Nothing forbids qualifiers; s1/s3 wrote "confirmed (not run)". GAP-1 impl
  choice is wrong: SKILL.md:38 lists rejected and :30 says "status change", only templates/LOG.md omits it, so s3 had to log a
  rejection as hypothesis. Fix: status is exactly one of four words, caveats go in the claim text, rejected allowed in LOG.
- H5 Standing questions (templates/STATE.md:3). X undefined, origin of questions 2-5 undefined, "max 5" vs a cold test that
  needs five. Fix: literal "How do I open <project>?", exactly five, say who writes them and when.
- H6 Blocker close undefined (SKILL.md:30). Impl closes on same text + non-unknown status; s7 kept the closed one open. Fix:
  state the rule in the skill.
Medium:
- M1 Tags (SKILL.md:28). "Fact" undefined, tag position in a LOG line undefined, legal tag/status pairs undefined. GAP-4 impl
  choice questionable: s1/s3 LOG lines carry no tag yet their STATE shows [SRC]; better: infer SRC when evidence is file:line.
- M2 Append-only needs a lock (SKILL.md:19 never mentions one); 692 of 4000 lines lost unlocked.
- M3 Metric line: no example row, tokenizer undefined (GAP-2), pipe escaping undefined (GAP-3), close line vs "nothing else".
- M4 Harness self-confirming: fuzz tests have no oracle (0 accepted), gate_2000 loaded=998 is generator design, stress inputs
  synthetic (padding words, 20000 distinct claims vs 6-40 in dogfood), 8 writers unrealistic; s2 5/5 is n=1 on a hand-written
  STATE with no negative control.
- M5 Bench fairness: .677 (1237-row holdout, 32% labelled) vs README .793 (all 4169 prompts) are different populations
  (already noted in section 4). Checked: the skeptic's "oracle leak" is wrong for the same row (gate_bench.py:58 scores before
  :60 updates state), but the simulated STATE uses earlier rows' gold labels, which is optimistic against a real STATE.

## 8. Untested / next (completeness critic + what the campaign could not do)
Cheap tests, in the order to run them:
1. DONE 11-05 (s9, race_test.py): 2x20 and 8x10 closers on one .state; LOG kept every line, STATE never torn, every claim
   survived; 1 of 80 workers wrote stderr once, 0 of 208 on reruns (text lost, first_error field added).
2. DONE 11-05 (s9, unit test_idempotent_15_cycles): 15 cycles x 5 lines, regen(regen(x)) == regen(x) each cycle, sets equal.
3. DONE 11-05 (s9, provenance_check.py): s1 6/6 claims orphan, s3 3/3, s5 1/4, s6 1/2; 0 after impl regen.
4. DONE 11-05 (s9, workflow wf_2b1a5cef-1ae): 6 Sonnet sessions, 7/7 each from STATE alone, 0 peeked at the sources.
5. DONE 11-06 (s10, class Round2): pipe/em-dash inside claim text round-trip; a pipe inside evidence is malformed (GAP-3).
6. DONE 11-06 (s10, workflows wf_ecd1e11d-caf + wf_a402722d-c7f): 3/3 sessions abstain at 5 with a blocker (2 blockers
   malformed, GAP-11); `check` flags a 600-token USER.md or TASK.md in .state/ or the root.
7. DONE 11-06 (s10, GAP-9): both candidates held out, "mutual conflict with X: ask the user", either input order, 3-way too.
8. DONE 11-06 (s10): CRLF, lone CR, UTF-16 BOM/BOM-less, BOM after append, mixed: regen/check never crash; append refuses
   a non-UTF-8 LOG; the test found 2 append_log lock bugs (fixed).
9. DONE 11-06 (s10): receipt mismatch reported, latest receipt per path wins; PT-BR exact accents; 5,000-char message fine.
10. DONE 11-06 (s10, gate_bench.py): STATE from earlier labelled rows only, base rates + Wilson CIs; always-abstain unbeaten.
Not done in this campaign: no session ran programs (no sha256 receipts), one project, one language, single closes only.

## 9. Fix round 11-05 (results/unit.json race.json provenance.json cold.json dogfood.json; backups tests/*.bak-1105fix)
User "Ys" to: fix GAP-1/4/7/8 and run s8 checks 1-4. Impl changes (astate_lite.py, grep "11-05"): LOG accepts rejected (GAP-1);
untagged fact with a source-pointer evidence -> [SRC] (GAP-4); item_met = exact normalised claim or whole phrase in the message
(GAP-7); empty requires never loads (GAP-8); check's provenance rule = exact normalised match STATE claim <-> LOG what-text;
write_state atomic (tmp + os.replace). Tests 47 -> 59 (classes Fixes 7, Skeptics 5), 0 failures, 3.87 s.
| check | result |
|---|---|
| 1 race (race_test.py) | 2x20: log_ok 20/20, state_parses 20/20, state_has_all 20/20; 8x10: 10/10 on all three; 1 of 80 workers wrote stderr once, 0 of 208 on reruns (text lost; first_error field added for next time) |
| 2 idempotence (unit) | 15 cycles x 5 lines = 75 LOG lines; regen twice identical every cycle; confirmed_claims(STATE) == derived set (6 at the end) |
| 3 provenance (provenance_check.py) | hand-written STATE: s1 6/6 claims orphan, s3 3/3; agent-shortened claims: s5 1/4, s6 1/2; after impl regen all 0 |
| 4 cold (cold.json, 6 Sonnet sessions, 7 questions, 2 unanswerable) | cold_ok 7/7 7/7 7/7; cold_wrong 7/7 7/7 7/7 (sessions repeat the wrong STATE, so they read STATE, not the sources); peeked 0; files_read = STATE.md only; unanswerable -> "unknown" 6/6 |
| dogfood rescore (dogfood_check.py) | 5/8 -> 4/8: s6 now fails check only because the agent wrote a shorter STATE claim ("service listens on port 9090") than its LOG line; s5 keeps passing with 1 such orphan + 1 malformed metric line (3 fields) reported |
| gate bench | unchanged: gate_bench.py scores with its own hit logic, GAP-7 does not touch it |
Skeptic round (2 Sonnet skeptics on the diff, cold.json "skeptics"), fixed the same day, pinned in class Skeptics:
- high: FILE_LINE_RE matched 127.0.0.1:8080, example.com:80, "ver 3.13:5" -> false [SRC]. Now the extension must be a source/doc
  kind (py md toml json yaml ps1 ...) or the file Makefile/Dockerfile/LICENSE/Justfile; locators :N, #LN, "line N".
- high: item 'v1.2' was met by message 'v1.2.3' ('.' counted as a boundary). Now '.' between word chars continues the token.
- low: 'Use py.' vs 'use py' failed the exact match -> trailing .;:! ignored on both sides (_key).
- low: README.md#L9, 'src/app.py line 3', Makefile:12 were not source pointers -> now are.
- not fixed (spec limit): "do not use 8080" still counts 8080 as stated; SKILL.md:34 says stated, a phrase match cannot see negation.
Decision kept: provenance stays exact. A STATE claim that paraphrases or shortens its LOG line is drift by the skill's own rule
("STATE derived from LOG"); the mechanical cure is `regen`, never a looser matcher (a prefix rule would have hidden s1/s3 drift too).
s8 items 5-10: s10.

## 10. Round 2 11-06 (s8 items 5-10; results/unit.json gate_bench.json round2.json; backups tests/*.bak-1106r2)
Harness: scratchpad patch_r2.py (impl patch, applied once), build_budget.py (3 fixture copies: TASK budget 5, STATE hypothesis
pointing at docs/ops/, answer "stg listener 9443" at the end of an 8-file chain ops-a..h), workflows wf_ecd1e11d-caf (budget run 1
+ 2 skeptics) and wf_a402722d-c7f (budget run 2). 78 tests, 0 failures, 5.11 s (+18: Round2 10, Round2Skeptics 8; +1 Gap11 11-06).
5. pipes/em-dashes: a pipe or em-dash inside the claim text round-trips LOG -> STATE -> parse; a pipe inside evidence is
   malformed (GAP-3: evidence is the rsplit field). Tests only, no impl change.
6a. budget abstention, run 1: 3/3 Sonnet sessions abstained at 5 Reads with a blocker line, but Read cannot list a folder and
   Glob was forbidden, so none reached the chain (they guessed staging.md, README.md, deploy.md, notes.md): harness flaw,
   mechanics only. Run 2 (one Glob listing allowed, counted as a step): 3/3 abstained at exactly 5 (listing + ops-a..d), 3/3
   appended a blocker, 0 wrong answers. s1's blocker parses (status unknown, question in the what-text); s2 and s3 put the
   question into the status field ("blocked; need +4 explore steps ...", "unknown; ask user: ...") -> `check` reports 1
   malformed line each. GAP-11: the LOG line has no field for the user question and the text does not say where it goes, so 2
   of 3 sessions chose status. The impl keeps the strict status set (a loose parser would hide it); the cure is a sentence in
   the skill text (frozen) or the TASK template. Added 11-06 (user "Yes, add it"): the templates/TASK.md blockers line and
   task_capsule() carry BLOCKER_NOTE (the question goes in the what-text; status stays unknown until a later line with the
   same what-text closes it), pinned in class Gap11; sessions not re-run. s2 also reported that the reread guard blocked SKILL.md; it worked from the
   TASK capsule alone and still abstained on budget.
6b. `check` on a 600-token TASK.md or USER.md, in .state/ or the root: "TASK 600 tokens > cap 300"; a folder named TASK.md
   is reported, not raised. CJK text counts one token per character (skeptic finding below).
7. two passing contracts with mutual conflicts: both held out with "mutual conflict with X: ask the user", both input orders,
   3-way too (GAP-9 decided 11-06: hold both out, never pick by order). Conflict items match through the gate's own
   normalisation (skeptic: conflicts ['b'] missed a passing candidate named 'B').
8. CRLF, lone CR, UTF-16 with BOM, BOM-less UTF-16LE (PowerShell 5.1 `>` and `>>`), BOM after an append, mixed UTF-8 head +
   UTF-16 tail: regen and check never crash; a non-UTF-8 STATE/LOG is reported "not UTF-8 (enc): re-save as UTF-8";
   append_log refuses any non-UTF-8 LOG (sniffs the first and last 8 KB). The item-8 test found two append_log bugs the
   skeptics did not: after the guard raised, the unlock hit the wrong byte (EACCES) and close() failed (EINVAL), both from
   BufferedRandom read-ahead leaving the OS pointer past byte 0; fix = open(..., buffering=0). A stray NUL inside UTF-8 text
   stays UTF-8 (NUL density < 1/3) and that line alone is rejected or kept by parse_log.
9. receipt after a source edit: "receipt mismatch: README.md (source edited after its receipt ...)"; the latest receipt per
   path wins; paths holding " — ", "./f" vs "f", upper-case hex and a folder as receipt path are handled. Gate on PT-BR:
   exact accents only (spec limit, "configuracao" does not meet "configuração"); a 5,000-character message gates fine.
10. gate bench rerun, STATE from earlier labelled rows only (n 1237, 400 labelled, Wilson 95% CIs): labelled base rate 0.323
    [0.298, 0.35]; always-abstain acc_abst 0.677 [0.65, 0.702]; state_and_any hit1 0.255 [0.215, 0.3], acc_abst 0.653 [0.626,
    0.679], precision 0.394 [0.336, 0.454] on 259 loads; strong 0.06 / 0.377 / 0.045 (536 loads); any 0.275 / 0.203 / 0.104
    (1055); name 0.007 / 0.672 / 0.25 (12); two_items 0.02 / 0.534 / 0.032 (251); strong_or_state 0.188 / 0.39 / 0.12 (623).
    No variant beats always-abstain on acc_abst; state_and_any is the only one whose precision clears the base rate, with a
    CI overlapping always-abstain. The s4 verdict stands.
Skeptics (2 Sonnet, on the diff before the buffering fix; both refuted=true). encoding-and-io: a NUL inside UTF-8 flipped the
decoder (high); the 2-byte BOM guard let BOM-less and mixed files take a UTF-8 append (medium); lone CR no longer a newline
(medium); a mixed tail starting with U+0100 off by one (low). gate-and-receipts: a folder as receipt path or as TASK.md
crashed check (medium); " — " inside a receipt path (medium); "./f" vs "f" and upper-case sha false mismatch (low); _clash
case-sensitive (low); tokens() counted 400 CJK chars as 100 tokens (low). All 9 fixed (grep "skeptic 11-06") and pinned in
class Round2Skeptics. Held: order independence, self-clash, a failing candidate holds nobody out, latest receipt wins, spaces
in paths, empty files, mid-file BOM.
Not covered: two appenders on a non-UTF-8 LOG; a UTF-16 run that starts after the first 8 KB and ends before the last 8 KB is
not sniffed; no session read a USER capsule (check only). GAP-11 mitigated 11-06 by the TASK template note (class Gap11; sessions not re-run); skill text still frozen.

## 11. v2.2 round 11-10 (backups tests/*.bak-1110v22)
- Unit: `py -3.13 -I -X utf8 test_astate_lite.py` -> 100 tests OK (78 kept, 4 of them moved to v2.2 strings, + class V22 with 22).
- `astate_lite.py check ~/Documents/Tools/ops` -> ok true, problems [], state_tokens 1674, state_claims 28.
- effect_veto_bench.py: v1 gate routed 86, false_blocked 74, true rows 0; `--v22` (skill must be named; effects taken as authorised) false_blocked 86 of 86, true rows 0 -> cost on true loads UNMEASURED (results/effect_veto_bench_v22.json).
- gate_bench.py ran without error (numbers not compared); dogfood_check.py not rerun (calls gate_v1).
- Tripwire hook: scratchpad test_tripwire.py 10 of 10; live inject, --block and touched log seen in the building chat.
- Not tested: v2.2 gate on real routed prompts with true loads; cold test under v2.2 templates.

## 12. Tripwire measurement 11-10 (print page 2, sharp photo; text in tree/tripwire-print.md s2)
- `py -3.13 -I -X utf8 tripwire_eval.py` -> 4 of 4 step-1 cases pass both directions; 5 of 9 need the dispatcher (step 2).
- Weekly line extended: `astate_metric.py --log` on a scratch LOG -> `rediscoveries=7 wrong_state=0 cold_test=5/5 tw_applied=1 tw_missed=0 tw_false_inj=1/2 tw_bar=n<5`.
- Bars fixed before the first run (ours): tw_missed = 0, false injections <= 20 % of injections, judged from 5 injections up.
- Fixed 11-10: astate_metric.py --log had not appended since the morning edit (statement swallowed by a comment); both paths rerun.
