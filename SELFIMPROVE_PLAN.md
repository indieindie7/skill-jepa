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
- [x] `skilljepa/trace_pairs.py` (10-26, Strata draft fixed by Sonnet: 62 lines; first run rows 1635, brief_only 632, full_read 530, miss 462, none 5, none+read 6, rewarded 1612 -> the 300-row pass is met, but rows are replayed route.log, see sec 8 leakage note). [ ] nightly routine in opsd (after "route-trace"); [ ] `--exclude-eval` flag.
- [x] No-match prompt set (experiment 3): `skilljepa/data/nomatch.jsonl`, 97 prompts (23 pt-BR), written by Strata on the 4070 10-26 (`notes/local-jobs-1026/nomatch_gen.md`); label `none`, use as the abstain half of the eval.
- [x] GPU picker shared by all starters (10-26: `Documents/Tools/llm/gpu.py`, 109 lines, free/pick/start/stop CLI + game check; start/stop exercised 10-26 (picked gpu 1, health ok, stop clean); game check only matches paths with steamapps/games (gpu.py:30); lifted from localread.py into
      `llm/gpu.py` so trainers and mixd use the same game check).
- [ ] Models to download (uncensored first): a 0.5-1.5B instruct policy; a 300M-class student.
- [x] Eval harness (10-26, Sonnet from the Strata draft): `skilljepa/eval_all.py` (105 lines): parses `ops/treeroute_eval.py` rows + reads/`--rerun`s `skilljepa.py eval` splits, writes `data/eval-<date>.json`, `--compare PATH` diffs. First run 10-26: tree:base 0.581/0.817, tree:base+prev 0.586/0.828, prompt-prevskill-all gate_soft 0.647/0.809/abstain 0.820. ES/GRPO policy + student rows still to add once they exist.

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
