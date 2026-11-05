# HANDOFF-jepa-user-profile (A17u per-user latent profile) - parked 2026-11-04

## Index (root)
- 1. Mission + status (PARKED 2026-11-04): what this chat did, why it stopped (lines 14-21)
- 2. Decisions + verdicts: table wins cold, kernel residual wins warm, subtext real (lines 23-26)
- 3. Run 1 results (a17_user_profile.py): T0..T4 warm hit1 table + reading (lines 28-49)
- 4. Follow-up results (a17u_followups.py): T3k/T3lr/T5/T2d numbers + reading (lines 51-74)
- 5. Permuted-history control: T3k vs T3kperm, SUBTEXT verdict (lines 76-102)
- 6. Files, paths, commands: repo files, data outputs (gitignored), how to rerun (lines 104-113)
- 7. Peer results that set production shape: A6b/A6c: base router cold + T3k warm (lines 115-122)
- 8. Scans: this chat's take + gaps: C1-C2 unscanned, C3/C6/C7/E2 open (lines 124-127)
- 9. Open items + next step: what a successor would do first (lines 129-131)

## 1. Mission + status (PARKED 2026-11-04)
Per-user latent profile (A17u) vs per-session transition table in the skill-jepa router; falsification plan from printed scans.
PARKED by the user on 2026-11-04 ("lets just park the jepa research ... write handoffs and archive them"). Chat archived after this file.
Owner chat: "Jepa-user latent inference profile for subtext" (session 6db1f042-30da-4113-89ec-c9e452728da4). Sibling handoffs in this repo: HANDOFF-jepa-context.md, HANDOFF-jepa-skill.md, HANDOFF-quality-standard.md.
Commits of this chat (all pushed, main == origin at park time):
- eb4255a A17u permuted-history control: T3k 63.0 vs T3kperm 46.5 warm hit1 -> kernel residual gain is subtext, not session frequency
- 7103dd3 A17u follow-ups: kernel residual T3k beats per-session table (63.0 vs 56.8 warm hit1); T2d decoupled fusion fixes false routes
- 148d892 A17 per-session latent profile vs transition table: TABLE WINS (T1 56.8 / T3 51.8 / T0 32.9 warm hit1); 10-31 scan notes + WinRT OCR script

## 2. Decisions + verdicts
- A17u run 1 148d892 TABLE WINS; follow-ups 7103dd3 T3k 63.0 warm (latent SUPPORTED); perm control eb4255a T3k 63.0 vs T3kperm 46.5 = SUBTEXT. All pushed, main == origin.
- Peer A6c: production = base router cold + T3k warm.
- Handoff drafted by Sage (localread.py --model sage), Claude checks values only. NO push.

## 3. Run 1 results (a17_user_profile.py)
### Run 2026-10-31 13:40 (seeds=3, ctx=prevskill, w=0.3, c=1.0, cold=10, min=30)

| arm | cold hit1 | warm hit1 | all acc_abst | all false_route_none |
|---|---|---|---|---|
| T0 | 23.0 +- 1.4 | 32.9 +- 1.3 | 68.0 +- 0.2 | 4.9 +- 0.2 |
| T1 | 57.8 +- 0.0 | 56.8 +- 0.0 | 70.8 +- 0.0 | 6.0 +- 0.0 |
| T1perm | 40.7 +- 6.2 | 53.1 +- 1.9 | 67.3 +- 0.3 | 2.1 +- 0.4 |
| T2 | 58.1 +- 0.5 | 60.3 +- 0.4 | 58.4 +- 0.2 | 39.3 +- 1.0 |
| T3 | 42.2 +- 0.9 | 51.8 +- 1.2 | 69.0 +- 0.3 | 0.9 +- 0.2 |
| T4 | 58.9 +- 0.9 | 60.3 +- 0.2 | 52.5 +- 0.2 | 51.7 +- 0.8 |

Verdict: TABLE WINS: per-session table/fusion >= latent profile on warm hit1

Per-session warm hit1 seed 0 (T0/T1/T1perm/T2/T3/T4): 586de7b7 28/52/50/54/37/53, ab4031b2 32/67/63/68/67/70, 948cc452 56/58/51/67/63/70, d4e4ed81 49/67/8/72/62/70, ce153384 20/50/40/37/30/43, f3249b13 21/31/62/41/41/31, 36aff851 19/50/58/62/62/58, 8d3c6b2d 50/68/77/77/68/82, a4c10a49 36/68/45/64/50/64

### Reading (2026-10-31)
- TABLE WINS by the preregistered rule: per-session table T1 56.8 warm hit1 vs latent profile T3 51.8 vs global JEPA T0 32.9 (3 seeds, 9 sessions, macro). Fusion T2/T4 60.3 on hit1 but break abstention (false_route_none 39-52 %, acc_abst 58/52 vs T0 68): the log-prior shifts energy scale, tau retune on train rows did not transfer.
- Fingerprint: T1perm 53.1 (shuffled labels keep per-session class FREQUENCY, lose transitions). So ~20 pts of the table gain = session frequency prior, ~4 pts = real transitions. Not within the 1.0 warning band, but most of the "personal" signal is "which 3 skills this chat uses".
- T3 is the only arm that improves acc_abst over T0 (69.0 vs 68.0) with the lowest false routes (0.9 %): the latent residual is a safe but weak personal correction.
- Per-session: T3 ties/beats T1 only on ab4031b2, 948cc452, 36aff851, f3249b13; loses hard on 586de7b7 (37 vs 52) and a4c10a49.
- Conclusion for the falsification plan's flip branch: on this data a per-chat frequency/transition table is the cheaper personalised router; JEPA latent profile does not earn the slot. Next candidates (not run): T3 with per-session residual fitted as a low-rank map instead of a mean shift; fusion with rank-based (not log-prob) prior so tau survives; per-session table as the production "ctx" feature.

## 4. Follow-up results (a17u_followups.py)
### Follow-up run 2026-10-31 14:13 (seeds=3, w=0.3, temp=0.1, rank=8)

| arm | cold hit1 | warm hit1 | all acc_abst | all false_route_none |
|---|---|---|---|---|
| T0 | 23.0 +- 1.4 | 32.9 +- 1.3 | 68.0 +- 0.2 | 4.9 +- 0.2 |
| T1 | 57.8 +- 0.0 | 56.8 +- 0.0 | 70.8 +- 0.0 | 6.0 +- 0.0 |
| T2d | 58.1 +- 0.5 | 60.3 +- 0.4 | 68.0 +- 0.2 | 4.9 +- 0.2 |
| T2r | 24.8 +- 2.6 | 34.3 +- 1.4 | 67.9 +- 0.4 | 5.3 +- 0.2 |
| T3 | 42.2 +- 0.9 | 51.8 +- 1.2 | 69.0 +- 0.3 | 0.9 +- 0.2 |
| T3k | 54.4 +- 0.9 | 63.0 +- 0.5 | 70.2 +- 0.1 | 1.7 +- 0.1 |
| T3lr | 46.3 +- 0.5 | 60.3 +- 0.7 | 69.0 +- 0.1 | 1.8 +- 0.0 |
| T5 | 59.3 +- 1.0 | 62.8 +- 0.1 | 68.2 +- 0.2 | 4.9 +- 0.2 |

Verdict: T3k SUPPORTED (latent beats table); T3lr SUPPORTED (latent beats table); T5 SUPPORTED (latent beats table); T2d FUSION FIXED

Per-session warm hit1 seed 0 (T0/T1/T2d/T2r/T3/T3k/T3lr/T5): 586de7b7 28/52/54/29/37/46/50/54, ab4031b2 32/67/68/36/67/67/72/69, 948cc452 56/58/67/57/63/67/69/70, d4e4ed81 49/67/72/52/62/72/69/70, ce153384 20/50/37/20/30/43/40/50, f3249b13 21/31/41/17/41/48/45/38, 36aff851 19/50/62/23/62/69/65/65, 8d3c6b2d 50/68/77/50/68/77/73/82, a4c10a49 36/68/64/36/50/73/68/68

### Reading (follow-ups, 2026-10-31)
- First verdict reversed: the per-session latent residual beats the table once it is PROMPT-CONDITIONAL. T3k kernel residual warm hit1 63.0 vs table T1 56.8 (+6.2), acc_abst 70.2 (best of all arms, T0 68.0), false routes 1.7 %; no table used at all. T3lr low-rank 60.3 (+3.5). T5 (T3k + table log prior, T0 abstention) 62.8, no gain over T3k alone. Mean shift T3 (51.8) was the wrong functional form, not a wrong hypothesis.
- Fusion fixed: T2d (rank by fused energy, abstain by T0 tau) keeps 60.3 hit1 with T0's 4.9 % false routes; retuning tau on the shifted scale was the bug. Rank prior T2r (34.3) is too weak at w=0.3.
- Cold bucket: T3k 54.4 < T1 57.8 (table wins the first 10 labelled rows; the kernel needs neighbours), both far above T0 23.0. Per-session T3k loses to T1 only on 586de7b7 (46 vs 52) and ce153384 (43 vs 50).
- Caveat: T3k residuals are built from the session's own labels, so part of its gain is still the session frequency prior (T1perm showed ~20 pts of that). Open test: T3k fed a label-permuted history (kernel over shuffled residuals) to split prompt-conditional subtext from frequency.
- Flip-branch verdict now: JEPA latent earns the personalised-local-correction slot as a kernel residual over past (pred, label) pairs; production candidate = T3k with T1 for the cold bucket.

## 5. Permuted-history control
### 6. Permuted-history control for T3k (preregistered 2026-10-31, a17u_followups.py --perm)
Question: is T3k's gain prompt-conditional subtext or the session frequency prior in disguise? Control = same kernel, same past preds, but residuals built from LABEL-PERMUTED history (labels shuffled among the session's labelled rows, rng seed+1000; keeps per-session class frequency, destroys the (pred, label) pairing). Arms added: T3perm (mean shift on permuted residuals), T3kperm (kernel on permuted residuals). Rule: SUBTEXT iff warm hit1 T3k - T3kperm >= 2.0 (mean over 3 seeds); FREQUENCY iff within 2.0 (the kernel only re-learns which skills this chat uses); report T3perm vs T3 the same way.

### Follow-up run 2026-10-31 14:35 (seeds=3, w=0.3, temp=0.1, rank=8)

| arm | cold hit1 | warm hit1 | all acc_abst | all false_route_none |
|---|---|---|---|---|
| T0 | 23.0 +- 1.4 | 32.9 +- 1.3 | 68.0 +- 0.2 | 4.9 +- 0.2 |
| T1 | 57.8 +- 0.0 | 56.8 +- 0.0 | 70.8 +- 0.0 | 6.0 +- 0.0 |
| T2d | 58.1 +- 0.5 | 60.3 +- 0.4 | 68.0 +- 0.2 | 4.9 +- 0.2 |
| T2r | 24.8 +- 2.6 | 34.3 +- 1.4 | 67.9 +- 0.4 | 5.3 +- 0.2 |
| T3 | 42.2 +- 0.9 | 51.8 +- 1.2 | 69.0 +- 0.3 | 0.9 +- 0.2 |
| T3k | 54.4 +- 0.9 | 63.0 +- 0.5 | 70.2 +- 0.1 | 1.7 +- 0.1 |
| T3lr | 46.3 +- 0.5 | 60.3 +- 0.7 | 69.0 +- 0.1 | 1.8 +- 0.0 |
| T5 | 59.3 +- 1.0 | 62.8 +- 0.1 | 68.2 +- 0.2 | 4.9 +- 0.2 |
| T3perm | 32.2 +- 1.8 | 47.7 +- 1.5 | 69.2 +- 0.1 | 1.0 +- 0.2 |
| T3kperm | 34.8 +- 1.0 | 46.5 +- 1.9 | 68.1 +- 0.2 | 0.6 +- 0.3 |

Verdict: T3k SUPPORTED (latent beats table); T3lr SUPPORTED (latent beats table); T5 SUPPORTED (latent beats table); T2d FUSION FIXED; SUBTEXT: T3k - T3kperm = +16.5 warm hit1 (T3 - T3perm = +4.1)

Per-session warm hit1 seed 0 (T0/T1/T2d/T2r/T3/T3k/T3lr/T5/T3perm/T3kperm): 586de7b7 28/52/54/29/37/46/50/54/33/33, ab4031b2 32/67/68/36/67/67/72/69/51/61, 948cc452 56/58/67/57/63/67/69/70/57/46, d4e4ed81 49/67/72/52/62/72/69/70/52/48, ce153384 20/50/37/20/30/43/40/50/40/47, f3249b13 21/31/41/17/41/48/45/38/34/31, 36aff851 19/50/62/23/62/69/65/65/62/50, 8d3c6b2d 50/68/77/50/68/77/73/82/64/64, a4c10a49 36/68/64/36/50/73/68/68/55/59

### Reading (permuted-history control, 2026-10-31)
- SUBTEXT by the preregistered rule: T3k 63.0 vs T3kperm 46.5 warm hit1 (+16.5, 3 seeds); permuting the session's labels drops the kernel residual below the mean shift and 10 pts below the table (56.8). The (pred, label) pairing carries the gain, not the per-session class frequency. Mean shift keeps only +4.1 over its permuted twin (51.8 vs 47.7): the mean shift was mostly frequency, the kernel is mostly subtext.
- Per-session: T3kperm beats T3k only on ce153384 (47 vs 43, 40 labelled rows); everywhere else the pairing wins by 6-27 pts.
- Standing verdict for the flip branch: a prompt-conditional per-session latent residual (T3k) is a real personalised correction beyond frequency/transition tables. Cold bucket (first 10 labelled rows) still belongs to the table. Production shape: T1 cold -> T3k warm, T0 abstention; next = C4-style ids-stripped check on new sessions is already implicit (leave-one-session-out), remaining risk is the single-user data.

## 6. Files, paths, commands
- Repo C:/Users/john/Documents/Tools/skilljepa: a17_user_profile.py, a17u_followups.py, notes/a17-user-profile-20261031.md (sections 1-6 + readings), notes/scans-2026-10-31.md; data/ gitignored.
- Handoff source: C:\Users\john\AppData\Local\Temp\claude\29abb69cf93759cc\6db1f042-30da-4113-89ec-c9e452728da4\scratchpad\handoff_src.md (423 lines); draft -> scratchpad\handoff_draft.md
- Target: C:\Users\john\Documents\Tools\skilljepa\HANDOFF-jepa-user-profile.md (numbered section index at top with line ranges)
- Log: ~/.claude/skills/ops-manager/tree/12-skill-jepa-router-log.md; memory skill-jepa-prototype.md
- Spec (preregistered protocol + verdict rules): notes/a17-user-profile-20261031.md sections 1-3.
- Scans OCR: notes/scans-2026-10-31.md (+ -raw.txt); OCR script Documents\Tools\skilljepa
otes\ocr_winrt.ps1.
- Rerun: `py -3.13 a17_user_profile.py` ; `py -3.13 a17u_followups.py` ; `py -3.13 a17u_followups.py --perm` (cwd = repo; outputs to data/).
- Log of this work: ~/.claude/skills/ops-manager/tree/12-skill-jepa-router-log.md (A17u lines), memory skill-jepa-prototype.md.

## 7. Peer results that set production shape
- 10-31 A17u follow-ups (a17u_followups.py, 3 seeds, 9 sids): kernel residual T3k warm hit1 63.0 vs table 56.8, acc_abst 70.2 (best), false route 1.7; T3lr 60.3; T2d decoupled fusion 60.3 w/ T0 abstention fixes false routes; reverses TABLE WINS -> latent SUPPORTED as prompt-conditional correction; cold bucket table still better (54.4 vs 57.8); open: T3k on permuted history. notes/a17-user-profile-20261031.md
- 2026-10-31 A17 SVTS auto-gold (context chat, 0fdc391): graph-derived auto-gold eval FALSIFIED as proxy, Spearman vs holdout -.57 over 16 variants (rewards lexical A0, ranks prod d256b2 last); do not use a17_svts.py set as eval. A6b (per-chat prior nudger) pre-registered ledger line 38.
- 10-31 A17u permuted-history control: T3k 63.0 vs T3kperm 46.5 warm hit1 (+16.5) -> SUBTEXT, kernel residual gain = (pred,label) pairing not session frequency; mean shift T3 only +4.1 over its perm. Flip branch stands: T3k = real personalised correction; prod shape T1 cold -> T3k warm, T0 abstention. notes/a17-user-profile-20261031.md section 6
- 2026-10-31 A6b (context chat): per-chat prior nudger NOT SUPPORTED (ledger 38): hit1 base 31.1 -> T3k-warm/table-cold 62.9, but daily entropy 2.48 < base 2.61 and median KL 1.10 bits > 1; cold table = loud narrowing part (KL 5.79), warm T3k .56 bits. Table-only arm echo chamber (6.5 distinct/day).
- 10-31 peer A6b (a6b_nudger.py, ledger 38) used T3k warm (hit1 62.9, 13 skills/day) + T1 cold (narrows routing: 6.5/day, cold KL 5.79 bits); peer runs A6c = T3k warm + base router cold. No objection from this chat.
- 2026-10-31 A6c (context chat, ledger 39): nudger = base router cold (<10 labelled rows) + T3k kernel residual warm: SUPPORTED. hit1 31.1 -> 54.6, daily top-1 entropy 2.75 (base 2.61, prod 2.13), 15 distinct/day, median KL .21 bits. Production candidate; router code = JEPA chat.
- 10-31 peer A6c SUPPORTED (ledger 39): T3k warm + base router cold: hit1 54.6 vs base 31.1, entropy 2.75, median KL .21 bits; dropping the cold T1 table fixed routing narrowing -> production shape = base cold, T3k warm (not T1 cold).

## 8. Scans: this chat's take + gaps
### This chat's take (2026-10-31)
- Already in skilljepa: arms.py / struct_arms.py (A0/A1/B1-like), a12_risk.py + a14_abstain.py (C5), a11_state.py (C8 typed state), a1_split.py (session split = E1), freeze_index.py. Gaps = C3 (isolated BM25 variants, warm/cold, fixed k), C6 (task-success arm, frozen executor), C7 (label families), E2 (stage-timed runtime report).
- Flip branch = the user's new title ("user latent inference profile for subtext"): JEPA as per-user latent predicting unstated intent, judged against a per-user transition/frequency table. Preregister: user-held-out vs session-held-out vs random; win only if it beats the per-user table on NEW sessions of the SAME user and adds nothing worse on new users.

## 9. Open items + next step
Handoff open items to list: C1-C2 pages unscanned; C3/C6/C7/E2 gaps; single-user data caveat; production shape per A6c.
Next step if resumed: wire T3k warm correction behind the base router in production (peer A6c shape), on a second user's data before trusting the per-user claim (all numbers above = one user, 9 eval sessions).
