# Self-improvement plan: skill router -> tiny trainable model in a mixture of models (2026-10-25)

Goal (user, 10-25): prepare the environment so the skill router keeps improving by itself from
its own traces, and ends as a very small trainable model that loads on a GPU beside other small
LLMs ("almost like a mixture of models for diverse robust thinking").

Status key: DONE = exists and measured; NEXT = ready to start; LATER = needs an earlier step.

## 0. What exists (exact values from files, not from memory)
- Production router: opsd `/api/route` (BM25/trigger, 127.0.0.1:8765). Hit@1 0.621 / Hit@3 0.835
  in-sample on 862 labelled prompts (README sec "Note: the production trigger/BM25 router").
  10-25 per-session previous-skill prior (usage.jsonl, 6 h window, half-life 30 min):
  Hit@1 .646 / Hit@3 .833 vs .583 / .771 on 48 sid-labelled rows (`ops/treeroute_eval.py --sid-only`).
- skill-jepa (this repo, HEAD aa8ff56): frozen EmbeddingGemma 300M (llama-server 8084, CPU, -ngl 0)
  + residual MLP predictor, energy 1-cos, abstain tau, gate_soft, --ctx prevskill. Seen chats
  Hit@1 0.647 (jepa+prevskill), unseen 0.50 (zero-shot+prevskill), time split 0.58. Abstain:
  nothing beats always-abstain 0.793 by much (per-branch tau 0.801). Numpy, ~1 min CPU.
- Trace loop (experiment 1, DONE 10-24): `ops/route.log` (conf, top-3, query, sid) joined with
  `~/.claude/skills/usage.jsonl` by `ops/route_trace.py` -> classes brief_only / full_read /
  miss / none+read; report `ops/reports/<day>/route-trace.md`, daily routine "route-trace".
  Baseline on old routes: brief_only 37 %, full_read 33 %, miss 30 %, Hit@1 0.528.
- Brief cap DONE 10-24 (opsd.BRIEF_BRANCHES=8; video-notes brief 13k -> ~900 tokens, median 399).
- Rules (CLAUDE.md 10-25): big local LLMs + MCP off when idle, 8084 embedder stays; two GPUs,
  start on the one with most free VRAM; nothing on gpu 0 (4070 Ti) while a game exe is in
  nvidia-smi; Claude never reads transcripts (localread.py / autorun / opsd /api/ask do).

## 1. Data loop (NEXT, cheap, CPU)
Reward source = the trace, not hand labels. Each route row becomes one training example:
  (query, prev skill, session id, candidates) -> chosen skill, outcome class.
- Reward: +1 brief_only or full_read of the top skill (hit), 0 none+none, -1 miss / none+read
  (tree/307 "For us", takeaways sec "10-24 23:3x").
- Script to add: `skilljepa/trace_pairs.py` = `build_data.py` shape but reading route_trace rows
  instead of transcripts; writes `data/trace_pairs.jsonl` (git-ignored like data/). Resume + lock
  like `Documents/github/odysseus/scripts/cook_sft_alex_conversations.py`; a judge pass like
  `scripts/odysseus_conversation_qa.py` only for rows where usage.jsonl is ambiguous.
- Volume: ~10-20 sid rows/day now; experiment 2 (brief vs full A/B) reads the 7-day trace from
  ~10-31. GRPO needs a few hundred rewarded rows -> first training run ~mid-Nov unless replayed
  from old route.log (1402 rows, time-window join, overcounted).
- Pass/fail: trace_pairs.jsonl regenerates nightly with 0 duplicate keys; >= 300 rows with
  reward != 0 before step 3.
- 10-26 20:30 LIVE: trace_merge.py nightly (sec 6), 213 clean rows on day 1, +.033 hit1 on the AVB head (sec 4 numbers in sec 6 line); new-sid volume from here ~10-20 rows/day, next retrain fires itself at 300 new rows.
- 10-26 21:40 SAGE RELABEL = NO GAIN (`trace_relabel.py` 59 lines: Sage-A1.5 on 8093 judges each brief_only/miss trace row against its label + 7 nearest skills by embedding + none; needs chat_template_kwargs enable_thinking=false or content comes back empty; ~8 min per 145 rows, GPU picked by start_own, stopped after). 145 judged: agree 9, other skill 48, none 88 (too conservative). Retrain with relabelled rows `data/avb-trace-relabel-20261026.json`: head .576/.743, refit .602+-.016/.728; drop-none variant .574/.750, .585/.731; baseline self-labels .599/.751, .583/.754. VERDICT: router self-labels beat Sage relabels on both axes; keep the self-labelled loop, drop the judge pass. Head still data-bound: next = wait for the nightly rows (auto-retrain at 300) or more none-aware data (data/nomatch.jsonl into train).

## 2. Keep the frozen-encoder router as the control (DONE, keep)
gate_soft + prevskill stays the baseline every later model must beat on the same label set
(`py -3.13 skilljepa.py eval --ctx prevskill --split time`). Experiment 5 (JEPA as re-ranker
behind BM25, pass = Hit@1 > 0.62) is the first thing the data loop pays for.

## 3. Outcome-weighted policy: GRPO on the 3060 (LATER, after step 1 volume)
- Policy = small instruct model (Qwen 3.5 0.5B-1.5B class or Gemma 4 small, uncensored variant
  first per the 10-25 rule) that reads (prev skill, query, 8 BM25 candidates) and emits one skill
  name or "none". Group of 8 samples per prompt, reward from step 1, TRL GRPOTrainer, LoRA r=16,
  bf16, 1 GPU (3060 12 GB fits <= 1.5B with LoRA; 4070 Ti only when no game is running).
- Reference: Ajax recipe (tree/307): SFT on ~2k filtered examples -> GRPO -> abliteration. Ours:
  SFT on trace_pairs hits (step 1) -> GRPO with the trace reward -> eval vs step 2.
- Pass/fail: Hit@1 on the time split > gate_soft 0.58 and > BM25+prior .646 on sid rows; abstain
  accuracy on the no-match set (experiment 3, 100 prompts) > 0.793.

### 3b. ES / EGGROLL branch (FIRST, before GRPO; from sec 7.1, user decision 10-15)
- Same policy model + prompt format as 3. Fitness = trace reward from step 1 (right skill +1,
  wrong 0, "none" when the row is none +1), averaged over a batch of 256 rows per candidate.
- OpenES (arXiv 1703.03864): population 32, antithetic pairs, sigma 0.02, Adam lr 0.01 on the
  LoRA r=16 weights only (not full weights: 1.5B x 32 copies does not fit). EGGROLL
  (2511.16652): perturbations are low-rank `x W^T + s * (x B_i) A_i`, so 32 candidates = 32
  small (A,B) pairs resident at once, one forward each; no backward pass, no reference model.
- Hardware: inference-only, bf16, fits the 3060 at 1.5B with batch 8 x 32 candidates ~ 20 min
  per generation; 4070 Ti only when no game exe is in nvidia-smi (`llm/gpu.py pick`).
- Script to write: `skilljepa/es_train.py --algo openes|eggroll --rows data/trace_pairs.jsonl
  --exclude-eval` (sec 8 fix: eval sids out). Log fitness per generation to `data/es-<date>.jsonl`.
- Pass/fail: same bar as 3 (gate_soft 0.58 / BM25+prior .646 / abstain .793) PLUS beat the
  production rerank tier (sec 7.2, BM25 + idea #10 TF-IDF rerank) on the same rows via
  `eval_all.py`. Go to GRPO (3) only if ES plateaus below the bar after 30 generations.

- 10-26 smoke (Sonnet wrote `skilljepa/es_train.py`, 164 lines, qlora venv, reuses es_lab hooks): both algos run on the 3060. eggroll: base acc 0.203 -> 0.203 after 2 gens (best 0.219), 6.1 s/gen, 7.3 GB; openes: 0.203 -> 0.188, 4.0 GB. Chance = 1/9. Train rows 794 (81 none), eval = md5(query)%5==0 + session fold 0 sids (eval-*.json hold no sids). `--seqs 128` (256 = 13 GB, 60 s/gen). LoRA hits only the 6 full-attention layers (12 modules, 640k params); `--targets` can add linear-attention projections. Qwen3.5 runs the slow torch path (no flash-linear-attention / causal_conv1d). Full run estimate pop 32 batch 128 ~1.5-2 min/gen (guess). NEXT: 30-gen eggroll run on the free GPU, then eval_all.py row.

- 10-26 full run (eggroll, pop 32, batch 128, --exclude-eval, gpu 1, started 11:06, log `data/es-20261026-eggroll-full.log`, per-gen `data/es-20261026-eggroll.jsonl`): INTERIM at gen 12/30 (11:31): fit_mean 0.13 -> 0.10-0.12, acc_eval 0.129 -> 0.125 flat = chance (1/8), p_eval 0.12 = uniform over the 9 letters, 70-95 s/gen (gen 12: 419 s, GPU contention). Diagnosis: the untuned 0.8B model puts near-uniform mass on the letters, so the fitness landscape is flat and ES noise (sigma 0.02 on 639k LoRA params) cannot climb it. FIX BEFORE MORE ES: (1) SFT warm start on the 794 train rows (few minutes, LoRA r=16, 2 epochs) until the letter format is followed and acc > 0.3, then ES from that LoRA (`--init-lora`); (2) fitness = mean log P(correct letter) (not raw P) so small improvements register; (3) `--targets` incl. the linear-attention projections. FINAL (gen 29, 12:00): acc_eval 0.137 (best 0.141), p_eval 0.119, 97 s/gen, 7.5 GB = chance, confirmed.
- 10-26 SFT warm start (`sft_warm.py`, LoRA r=16, CE over the 9 letter logits at the last token, 2 epochs, lr 2e-4, batch 16, gpu 0, 150 s, 12.2 GB, `data/sft-20261026-lora/`, log `data/sft-20261026.log`): acc_eval 0.125 -> 0.652 (ep 1) -> 0.664 (ep 2), p_eval 0.130 -> 0.535, train loss 1.694 -> 1.008 (still falling: try 3+ epochs / `--targets` with linear-attention projections).
- 10-26 EGGROLL from SFT (`--init-lora data/sft-20261026-lora --fitness logprob --lr 0.0005`, 30 gens, `data/es-20261026-eggroll-sft.jsonl`, lora `data/es-20261026-eggroll-sft-lora/`): acc 0.664 -> 0.664, best 0.668 at gen 11, p_eval 0.535 -> 0.547, 42 s/gen, 7.5 GB. With the default `--lr 0.01` ES wiped the SFT start (0.66 -> 0.28 by gen 6, 0.11 by gen 12; kept as `data/es-20261026-eggroll-sft-lr0.01-diverged.*`).
- VERDICT 10-26: ES adds nothing on top of SFT (flat within the 256-row noise, +-0.012); SFT alone does the job in 2.5 min. Drop ES for this task; next = SFT 3+ epochs, then GRPO (sec 3) only if SFT plateaus. NOTE: 0.664 is acc on the 256 held-out trace rows (9-letter choice), NOT the eval_all.py Hit@1 bar (.58/.646/.793); eval_all.py does not load LoRAs yet, so the sec 7.2 comparison needs a `--lora` path in eval_all.py first. Refactor: loader -> `es_data.py`; es_train.py flags `--init-lora`, `--fitness prob|logprob` (default logprob), `--tag`.
- 10-26 12:30 RERANK-TIER SCORE (`policy_eval.py` 59 lines via `eval_all.py --lora data/sft-20261026-lora`, `data/policy-20261026.json`, time split test n 1237 = 400 labelled + 837 none, candidates = knn top-8 shuffled + I) none, 18 s, 2.3 GB on gpu 1): sft_policy_rerank hit1 .585 / hit3 .685 vs base knn .500 / .642, recall8 ceiling .780 (zero-shot candidates: hit1 .595, recall8 .75); unseen-only hit1 .547 (150 of the test queries are in the SFT train set). Hit1 clears gate_soft .58, misses BM25+prior .646. ABSTAIN BROKEN: abstain_acc .242, false_route .920, abstain_on_labelled .018 (always-abstain on this test = .677; the .793 bar is the prompt split, not reproducible here). Cause: SFT train has 81 none rows of 794 (10%) while the time test is 68% none. NEXT: (1) rebalance SFT none rows (oversample nomatch + the 837-style none queries from pairs.jsonl, or weight class I) and/or a tuned threshold on P(I) in policy_eval.py; (2) rerun; target abstain_acc > base knn .699 while keeping hit1 >= .585; (3) only then SFT 3+ epochs.
- 10-26 13:00 REBALANCED SFT (`sft_warm.py --exclude-eval --none-frac 0.5 --exclude-time-test --exclude-val`: 1116 rows, 558 none, 155 leaked rows dropped, 477 none rows from the time-split train part with knn top-8 candidates; loss 1.423 -> 0.833, acc_eval .594, 631 s on gpu 1, 13.7 GB peak; `data/sft-20261026-bal-lora/`, `data/sft-20261026-bal.log`) + `policy_eval.py --none-thr auto` (thr tuned on train positions 0.49-0.70, chosen .0774; `data/policy-20261026-sft-20261026-bal-lora.json`, `data/eval-20261026b.json`): hit1 .547 / hit3 .665 (unseen = all 400 labelled now, 0 overlap) vs base knn .500 / .642; abstain_acc argmax .677 (false_route .16), thr .690 (false_route .012, abstain_on_labelled .92) vs base knn .699 and always-abstain .677. BAR FAILED (.699 abstain, .585 hit1). The earlier .585 was leakage (150 test queries in SFT train); honest hit1 = .547.
- VERDICT 10-26 (policy track): a 0.8B SFT policy reranking knn top-8 beats knn hit1 by +.047 but cannot abstain better than always-abstain and stays far under BM25+prior .646 on 1.1k train rows. PARK the LLM-policy rerank (SFT/ES/GRPO) until the trace set is 3-5x bigger (sec 6 nightly route-trace routine). Next lever = sec 4 AVB decision-model reranker (order-invariant head over candidate embeddings, trains on the same rows in minutes, no 0.8B forward pass at route time) and more none-aware training data.

- 10-27 ref: Julia Turc ug9el1OGGEw (video-notes tree/348): LoRA r=1 matched r=16 on a 27B query-difficulty router; adapter stable rank median 3/16; label ORDER in the prompt swung zero-shot 26->48.5%. If 3b revives: r=1-4, measure stable rank, test candidate order in judge/policy prompts.

## 4. Distil to the tiny model (LATER)
- Target: a model small enough to stay resident beside other small LLMs: <= 300M params
  (EmbeddingGemma size) or the predictor MLP itself (today ~MB, numpy) with the GRPO policy as
  teacher. Methods, cheapest first (video-notes tree/81): sequence-level KD (teacher writes the
  skill name, student trains next-token) -> on-policy distillation (student samples, teacher
  scores every token, reverse KL; ~150 steps gave the 60 -> 70 % jump in the Thinking Machines
  run) -> LoRA merge. Needs shared vocabulary for token-level OPD; the MLP-student path needs only
  the teacher's chosen label + confidence.
- Pass/fail: student within 2 points Hit@1 of the teacher on the time split; < 1 GB VRAM resident;
  route in < 50 ms on CPU or the shared GPU.

- 10-26 reranker architecture candidate (from Windows support 3, video-notes tree/340, AVB "decision model", repo github.com/avbiswas/bev-train): Qwen-0.6B layers 1-20 + LoRA on the last 12 + 2-layer self-attn head + bilinear = one prob per choice, order-invariant (per-choice position reset + attention mask); 65.8% on his bench at ~450M params, ~$20 training. Fits us as the re-ranker over the BM25 top-N with route-log labels (sec 1) and Sage-A1.5 (localread --model sage, 8093, 9 GB) as the cheap judge/label source. Try after the SFT+ES loop settles; same eval_all.py bar.
- 10-26 16:30 AVB-STYLE HEAD BUILT (`avb_rerank.py` 87 lines, torch CPU, `--no-attn` ablation; head over the existing EmbeddingGemma embeddings: choice token = [query ctx-prevskill emb, cand emb, q*c, cos/knn energy/rank/prev-skill feats] -> d256 -> 2-layer self-attn over 8 cands + learned none, no positional enc -> logit per choice; CE; early stop on val CE (picked epoch 1-4 = overfits fast); ~1.2-1.7M params, 70 s for 3 seeds; `data/avb-20261026.json`, `data/avb-20261026.pt`, `data/eval-20261026c.json` via `eval_all.py --avb`). Head train 2024 rows (positions <0.49 of each chat, leave-one-out knn), val 876 (0.49-0.70), test 1237 (400 labelled / 837 none). Time-split test, mean+-std over 3 seeds: avb_head hit1 .557+-.008 / hit3 .680 (knn .500/.642, SFT policy .547/.665, recall8 .780); abstain_acc with val-tuned none threshold .740+-.012 (false_route .052, abstain_on_labelled .571) vs knn .699, SFT .690, always-abstain .677; argmax-none always abstains (useless). No-attn ablation: hit1 .531, abstain_thr .740 -> cross-choice attention = +.026 hit1, nothing on abstain. VERDICT: AVB head beats knn and the parked 0.8B policy on both axes at 1/400 the size and no LLM pass at route time, but still under gate_soft .578 and BM25+prior .646. Cheap next levers: refit on the full train part after thr tuning, label out-of-top-8 rows as none instead of dropping them, top-16 candidates (recall8 .78 caps hit1), add BM25 score + prior as feats, more rows (nightly trace).
- 10-26 17:30 AVB HEAD V2 (`avb_rerank.py` 86 lines rewritten, `--v1` keeps v1; `avb_feats.py` 99 lines: candidates = knn/BM25 interleaved union, BM25 > 0 only, K=16; feats + BM25 raw/rank, prev_prior (lifted from ops/treeroute_eval.py, reads >= 60 s before the prompt, non-zero on 236/1237 test rows since usage.jsonl starts 10-17), in-BM25-top1/3; out-of-K rows labelled none, not dropped; 3 seeds, thr on val, then REFIT on the full train part 2948 rows; `data/avb-20261026-v2.{json,pt}`, `data/eval-20261026d.json`; 1.68M params, 152 s CPU main run, ~8 min with ablations). recallK test .877 (K8 .780), out-of-K test 49. Time-split test: avb2_head_thr hit1 .566+-.005 / hit3 .726 / abstain .751+-.005 (false_route .073); avb2_refit_thr hit1 .579+-.015 / hit3 .751 / abstain .736+-.014; ablations: K8 .540/.718, no-BM25-feats .557/.742. VERDICT: v2 passes v1 (.557/.740) by ~1 std on both axes; refit matches gate_soft hit1 .578 but loses abstain; BM25+prior .646 still far. Gains: K16 +.026 hit1, BM25/prior feats +.01 each axis. Caveats: BM25 over the 77 skills.json skills, not the production set; minute-truncated usage times can leak the prior in production (`ut < t`). Remaining levers are data-bound (2.1k train rows, head overfits by epoch 4): nightly route-trace rows, Sage-labelled none/relabel pass, then retrain.

## 5. Mixture-of-models runtime (LATER)
- One loader (`Documents/Tools/llm/mix/mixd.py`, to write) that keeps several small models on one
  GPU: 8084 embedder (CPU today), the tiny router, and 1-3 other small specialists (e.g. a judge,
  a summariser); each answers the same prompt, a cheap arbiter (the router's confidence + agreement
  vote) picks or merges. Diversity = different base families, not seeds.
- Fits the rules: everything per job, stop when idle, pick the freest GPU, never gpu 0 while
  gaming; opsd owns start/stop (`/api/ask` already queues to Strata).
- Pass/fail: the mixture beats the best single member on the route label set and on a 50-prompt
  judged QA set, at <= 2x the single-member latency.

## 6. Environment prep checklist
- [x] Training venv: `C:\llm\qlora\venv` has torch 2.14.1+cu130, trl 1.15.0, peft 0.21.2,
      transformers 5.17.0, unsloth 2026.10.6 (checked 10-25). py 3.13 has no torch; keep it so.
- [x] `skilljepa/trace_pairs.py` (10-26, Strata draft fixed by Sonnet: 62 lines; first run rows 1635, brief_only 632, full_read 530, miss 462, none 5, none+read 6, rewarded 1612 -> the 300-row pass is met, but rows are replayed route.log, see sec 8 leakage note). [x] nightly routine (10-26 20:30): `trace_merge.py` 72 lines, opsd routine "trace-merge" at 07:05 after route-trace (ops/routines.json). Eval exclusion done inside trace_merge instead of a trace_pairs flag: drops empty sid (1398 of 1688 replayed rows), junk sid, any sid present in data/pairs.jsonl (70), unknown skill (4) -> 213 rows kept, embedded via 8084 into data/emb_trace.npz, appended to data/pairs_trace.jsonl (key sid|t|q, idempotent, 0 on rerun), state data/trace_state.json, log data/trace_merge.log. At >= 300 new rows (`--retrain-at`, `--force-retrain`) it runs qlora-venv `avb_rerank.py --extra data/pairs_trace.jsonl --tag trace-<date>` hidden: extra rows join the head train/refit pools only, val/test unchanged = comparable with v2. Forced first retrain (222 s, 213 extra): avb2_head_thr hit1 .599+-.010 / hit3 .766 / abstain .751+-.010 (v2 .566/.751); refit .583/.754 (v2 .579/.736); `data/avb-trace-20261026.json`. ~3 std hit1 gain from 213 rows = the head is data-bound as predicted. Caveat: brief_only rows carry the router's own top-1 as label (self-labelling, ~1/3 of trace rows), so part of the gain may be the head learning the production router; a Sage/none relabel pass on those rows is the next data lever.
- [x] No-match prompt set (experiment 3): `skilljepa/data/nomatch.jsonl`, 97 prompts (23 pt-BR), written by Strata on the 4070 10-26 (`notes/local-jobs-1026/nomatch_gen.md`); label `none`, use as the abstain half of the eval.
- [x] GPU picker shared by all starters (10-26: `Documents/Tools/llm/gpu.py`, 109 lines, free/pick/start/stop CLI + game check; start/stop exercised 10-26 (picked gpu 1, health ok, stop clean); game check only matches paths with steamapps/games (gpu.py:30); lifted from localread.py into
      `llm/gpu.py` so trainers and mixd use the same game check).
- [x] Policy model (10-26): huihui-ai/Huihui-Qwen3.5-0.8B-abliterated, 1.77 GB safetensors at `C:\llm\models\qwen35-0.8b-abl` (SOURCE.txt). [ ] 300M-class student still to pick.
- [x] Eval harness (10-26, Sonnet from the Strata draft): `skilljepa/eval_all.py` (105 lines): parses `ops/treeroute_eval.py` rows + reads/`--rerun`s `skilljepa.py eval` splits, writes `data/eval-<date>.json`, `--compare PATH` diffs. First run 10-26: tree:base 0.581/0.817, tree:base+prev 0.586/0.828, prompt-prevskill-all gate_soft 0.647/0.809/abstain 0.820. ES/GRPO policy + student rows still to add once they exist. [x] `--lora DIR` in eval_all.py (10-26, policy_eval.py): SFT adapter hit1 .585 on the time split, abstain broken (sec 3b). [x] none-rebalanced SFT + P(I) threshold (10-26 13:00): bar failed, policy track parked (sec 3b verdict).
- [x] SFT warm start + ES-from-SFT (10-26, `sft_warm.py` 62 lines, `es_data.py` 38, `es_train.py --init-lora/--fitness/--tag`): SFT 0.125 -> 0.664, ES flat on top; see sec 3b verdict.

## 7. From the Windows-support chats (Strata digs, 10-26)
Dug by Strata on the 4070 (`notes/local-jobs-1026/dig_ws1_*.md`, `dig_ws2_*.md`, `drill_es.md`, `drill_tfidf.md`). Line numbers = the chats' transcript extracts; ws1 7501-10000 and ws2 2501-5000 failed twice (timeouts), earlier partials held nothing new.

### 7.1 ES / EGGROLL instead of GRPO (ws1 L31647-L31764)
- User decision 10-15: "lets experiment es and eggroll". Plan then: decider-2b fine-tuned by Evolution Strategies (OpenES) or EGGROLL with low-rank, LoRA-shaped noise; scalar reward = rule label right/wrong; no backprop, no reference model, no KL.
- Why it fits here: fits a 12 GB card (inference-only passes), works for non-differentiable fitness (route log reward, later game sim / TRM enemy brain). Papers: arXiv 1703.03864 (OpenES), 2509.24372, 2511.16652 (EGGROLL).
- => sec 3 gets an ES/EGGROLL branch: try it first when only the 3060 is free; GRPO only if ES stalls below the BM25 baseline.

### 7.2 Production rerank tier already in opsd (ws2 L1317-L1666)
- Idea #10 fuzzy router (`ops/idea10_fuzzy_router.py`): truth pairs = route.log query -> usage.jsonl skill within 600 s; leave-one-out TF-IDF word + char-trigram centroids per skill; rerank only inside the text top-5; score floor 0.15 -> 0.25. Approved, measured gain 0.082; target `ops/opsd.py` (backups .bak-20261017l/m).
- Rejected: idea #9 BM25 body layer (gain -0.056); idea #8 SmolLM fallback archived (model not on disk).
- => any learned router must beat BM25 + this rerank tier, not plain BM25 (plan sec 0 numbers are plain BM25).

### 7.3 Rules re-found
- 8084 CPU embedder stays resident; swap to a lower-VRAM model at equal score; never persist route scores into index.json; Pareto = success + tokens.
- Nothing in ws1-3 about a mixture runtime or a from-scratch tiny model: sec 5 stays Claude's own design. skill-jepa history there (1c8bab4..aa8ff56) matches sec 0.

## 8. Review by the local model (Strata on the 4070, 10-26; raw: `notes/local-jobs-1026/plan_review.md`)
Accepted fixes, in order of weight:
- Leakage: before any training (sec 3), drop every sid that is in the `--sid-only` eval rows and every row after the time-split boundary from `trace_pairs.jsonl` (flag `--exclude-eval` in trace_pairs.py; eval sids read from `data/eval-session-*.json` / `eval-time-*.json`). The replayed route.log rows are overcounted; count a sid once.
- Pass criteria get a generalization gap: student and policy must be within 2 points of the teacher on the UNSEEN split (0.50 today), not only on the mixed Hit@1.
- Judge check: label 50 ambiguous trace rows by hand before the judge pass is trusted; judge precision/recall < 0.90 means train on unambiguous rows only.
- Distillation path: teacher and student tokenizers differ (Qwen/Gemma policy vs EmbeddingGemma+MLP student), so token-level on-policy distillation is out; distil the label distribution over the 78 skills (soft targets + abstain logit), i.e. sequence/label KD only.
- GPU rule stays "gpu 0 off while a game exe runs" (the review's "any GPU" reading is wrong); `llm/gpu.py` enforces it for trainers and mixd alike.
Local-model job log (what the 4070 did, serial, Strata 8090): `notes/local-jobs-1026/log.txt`.
