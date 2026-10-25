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
- [ ] `skilljepa/trace_pairs.py` + nightly routine in opsd (after "route-trace").
- [ ] No-match prompt set (experiment 3): `skilljepa/data/nomatch.jsonl`, 100 prompts.
- [ ] GPU picker shared by all starters (localread.py already has `pick_gpu`; lift it into
      `llm/gpu.py` so trainers and mixd use the same game check).
- [ ] Models to download (uncensored first): a 0.5-1.5B instruct policy; a 300M-class student.
- [ ] Eval harness: one script that prints Hit@1/Hit@3/abstain for BM25, BM25+prior, gate_soft,
      GRPO policy, student on the same rows (`ops/treeroute_eval.py` + `skilljepa.py eval`).

## 7. From the Windows-support chats (localread dig, pending)
(filled from wsN.answer.md when the Coder finishes; only cited lines that were checked)
