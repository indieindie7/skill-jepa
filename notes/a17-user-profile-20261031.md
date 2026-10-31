# A17 per-user latent profile ("subtext") vs per-user transition table - preregistered 2026-10-31
## Index
1. Question + flip assumption - L6-9
2. Protocol (data, split, arms, metrics) - L11-24
3. Preregistered verdict rules - L26-30
4. Results (filled by a17_user_profile.py) - L32+

## 1. Question
Falsification-plan "central assumption": if latent state helps only within stable users/projects, JEPA = personalised local correction, not a global router. One user on this machine -> profile unit = session/chat (sid). Does a per-session latent residual (what the prompt does not say) beat a per-session transition/frequency table at predicting the next skill, on the SAME session's later turns, without hurting new sessions?

## 2. Protocol
- Data: data/pairs.jsonl (4185 rows, 46 sid, 862 labelled). Eval sessions = sid with >= 30 labelled rows (9: 586de7b7 282, ab4031b2 176, 948cc452 133, d4e4ed81 71, ce153384 40, f3249b13 39, 36aff851 36, 8d3c6b2d 32, a4c10a49 32).
- Split: prequential leave-one-session-out. Global predictor (sj.fit, ctx=prevskill, 40 ep) trained on all OTHER sessions; tau tuned on train rows. Held-out session walked in time order; every profile statistic uses rows strictly before the current row (online, no lookahead). Random split not run: leaks by construction.
- Arms (same embeddings, same tau machinery):
  T0 global JEPA energies (control = base-only JEPA).
  T1 per-session table: p(skill|prev_skill) with backoff session-frequency -> global transition -> global frequency (Dirichlet smoothing a=b=2); K+1 classes incl. none; abstain = none argmax.
  T1perm: T1 fed a label-shuffled session history (C4 fingerprint control; must collapse to global).
  T2 fusion: T0 energy - w*log p_table(skill), w=0.3; tau retuned on train rows with the global table.
  T3 latent profile: pred' = norm(pred + c*s*delta), delta = mean over past labelled rows of (Y[label] - pred_row), s = m/(m+5), c=1.0; energies 1-pred'@Y.T; T0 tau.
  T4 = T2 + T3.
- Buckets per session: cold = first 10 labelled rows (new-user regime), warm = rest. Metrics: hit1 (labelled rows), acc_with_abstain (all rows), false_route_on_none. Macro mean over the 9 sessions, mean +- sd over 3 seeds.

## 3. Verdict rules (fixed before running)
- JEPA-as-personal-correction SUPPORTED iff warm hit1: T3 > T1 and T3 > T2 by >= 2.0 pts (mean over seeds) AND cold hit1 T3 >= T0 - 1.0.
- TABLE WINS (JEPA adds nothing personal) iff T1 or T2 >= T3 on warm hit1.
- FINGERPRINT WARNING iff T1perm warm hit1 within 1.0 pt of T1 (session identity, not transitions, carries the gain).
- Any other pattern = inconclusive; report, no claim.

## 4. Results
(pending)

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
