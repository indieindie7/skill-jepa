# skill-jepa

A small prototype: route a coding agent's prompt to one of its **skills** (folders of instructions, like Claude Code `~/.claude/skills/<name>/SKILL.md`) with a **JEPA-style latent predictor** instead of an LLM router.

- **Context encoder:** frozen [EmbeddingGemma-300M](https://huggingface.co/google/embeddinggemma-300m) embeds the prompt as `x`.
- **Target encoder:** the same frozen model embeds each skill's `name + description` as `y_s`.
- **Predictor:** a residual MLP `g` (768 → 512 → 768, initialised to the identity, so it starts as the zero-shot cosine router). It is trained so that `g(x)` lands on the `y_s` of the skill the agent actually used.
- **Energy:** `E(x, s) = 1 − cos(g(x), y_s)`. The route is `argmin E`. The router **abstains** when the best energy is above a threshold τ, which is tuned on the training data.

Loss = latent regression (`1 − cos`) + α · InfoNCE over the skill targets. The targets are frozen, so the negatives play the anti-collapse role that VICReg / EMA play in image JEPAs.

Nothing is generated and the encoders never train. Training is plain NumPy on a CPU and takes about a minute; inference is one embedding call plus one matrix multiply.

## Data

`build_data.py` mines `(prompt → skills used)` pairs from local Claude Code transcripts (`~/.claude/projects/*/*.jsonl`). A user prompt is labelled with the skills the assistant touched before the next user prompt:
- `Skill` tool calls;
- reads under `skills/<name>/`;
- skill-brief API calls.

Prompts that touched no skill get the label `none`, and the router should abstain on those.

Your transcripts are private, so `data/` is git-ignored and nothing from it is in this repo.

## Results (one user's history: 4169 prompts, 862 labelled, 77 skills, 46 chats, 5-fold CV)

| split | router | Hit@1 | Hit@3 |
|---|---|---|---|
| unseen chats (grouped by session) | zero-shot cosine | 0.225 | 0.389 |
| | kNN over past prompts | 0.232 | 0.344 |
| | **skill-jepa** | 0.234 | 0.388 |
| seen chats (random prompts) | zero-shot cosine | 0.233 | 0.422 |
| | kNN over past prompts | 0.511 | 0.725 |
| | **skill-jepa** | **0.568** | **0.726** |

**Seen chats.** For projects the router has history for (the normal case for long-running chats), the predictor more than doubles zero-shot Hit@1 and beats nearest-neighbour voting.

**Unseen chats.** Nothing beats zero-shot. The skills used are too chat-specific, and many prompts are bare follow-ups ("continue", "ok") whose skill lives in the chat context rather than the prompt.

**Abstention is not solved yet.** About 79% of prompts use no skill, and no router's accuracy-with-abstain beats "always abstain" (0.793). Labels are noisy: a skill read later in a turn is not always the one the prompt needed.

## Usage

```bash
# 1. an OpenAI-style embeddings endpoint serving EmbeddingGemma, e.g. llama.cpp:
llama-server -m embeddinggemma-300M-Q8_0.gguf --embedding --port 8084 -ngl 0
# 2. data, embeddings, evaluation, model
python build_data.py
python skilljepa.py embed
python skilljepa.py eval --split session   # or --split prompt
python skilljepa.py train
python skilljepa.py route "summarize this youtube video for me"
python skilljepa.py eval --split session --ctx prevskill   # previous-turn context
python skilljepa.py eval --split time --ctx prevskill --labels clean   # labels without skill-file edits
```

- **Endpoint:** set `SKILLJEPA_EMB_URL` to use a different embeddings endpoint.
- **Requirements:** NumPy and Python 3.10+.

## Previous-turn context (`--ctx`)

Many prompts are follow-ups ("ok, do it") that only make sense given the turn before. `--ctx` adds that turn:
- `prev`: x + 0.5 · embedding of the previous prompt;
- `prevskill`: also + 0.5 · target embedding of the skill the previous turn used (known at routing time);
- `cat`: one embedding of the text `previous: … | now: …` (`python skilljepa.py embedctx` first).

A new baseline, `repeat_prev_skill`, just routes to whatever the previous turn used.

Hit@1 (Hit@3), same data and folds as above:

| ctx | router | unseen chats | seen chats |
|---|---|---|---|
| none | zero-shot | 0.225 (0.389) | 0.233 (0.422) |
| none | skill-jepa | 0.234 (0.388) | 0.568 (0.726) |
| — | repeat_prev_skill | 0.419 (0.431) | 0.415 (0.430) |
| prev | skill-jepa | 0.244 (0.374) | 0.638 (0.821) |
| cat | zero-shot | 0.293 (0.443) | 0.317 (0.474) |
| prevskill | **zero-shot** | **0.500 (0.628)** | 0.484 (0.643) |
| prevskill | kNN | 0.299 (0.390) | 0.616 (0.817) |
| prevskill | **skill-jepa** | 0.304 (0.539) | **0.647 (0.804)** |

- **The previous turn is the strongest signal.** Just repeating last turn's skill beats every prompt-only router on unseen chats.
- **Unseen chats:** zero-shot cosine with `prevskill` context wins (0.50). The trained predictor falls to 0.30 there, because it learns each chat's frequent skills and pulls bare follow-ups toward them. For example, "ok do it" after a video-notes turn routes to video-notes zero-shot, but the predictor sends it to the user's most-used skill.
- **Seen chats:** skill-jepa with context is best (0.647 Hit@1, up from 0.568).
- **Abstention:** routing accuracy with abstain reaches 0.803–0.806 with `prevskill`, the first setting to beat "always abstain" (0.793), but only just.

## Gate router (`gate_hard`, `gate_soft`)

Zero-shot + context generalises to new chats, and the trained predictor wins once a chat has history, so the gate uses both. `hist` is the number of labelled prompts the chat already had in training, with `k = 5`:
- `gate_hard` uses the predictor once `hist ≥ k`, and zero-shot before that.
- `gate_soft` blends the two energies with weight `hist / (hist + k)`.

A third split, `--split time`, tests on the last 30% of every chat, so each chat arrives with its own past. That is the realistic case. All rows below use `--ctx prevskill`.

| router | unseen chats | seen chats | time split |
|---|---|---|---|
| zero-shot | **0.500** (0.628) | 0.484 (0.643) | 0.551 (0.681) |
| kNN | 0.299 (0.390) | 0.616 (0.817) | 0.499 (0.640) |
| skill-jepa | 0.304 (0.539) | **0.647** (0.804) | 0.570 (0.706) |
| repeat_prev_skill | 0.419 (0.431) | 0.415 (0.430) | 0.519 (0.533) |
| gate_hard | **0.500** (0.628) | 0.645 (0.801) | 0.573 (0.704) |
| **gate_soft** | **0.500** (0.628) | 0.645 (0.804) | **0.580 (0.731)** |

- `gate_soft` is best or within 0.002 of the best on every split, so it is the one to use.
- **Abstain threshold, fixed:** τ is now tuned per branch (zero-shot and predictor), and the gate blends the two τ with the same weight it gives the energies.
  - New chats: accuracy-with-abstain goes from 0.793 to 0.801, the same as zero-shot.
  - Seen chats: 0.815 to 0.818.
  - Time split: 0.705 to 0.699.
  - Example: "ok do it" after a video-notes turn in a new chat now routes to video-notes; before, it abstained.
- **Previous skill as a separate input (`--ctx sep`), negative result.** The predictor gets the previous skill's target as extra input columns beside the `prevskill` mix. It did not stop the pull: with long history, "ok do it" still goes to the chat's most-used skill. It also costs Hit@1:

  | split | `gate_soft` Hit@1, `prevskill` | `gate_soft` Hit@1, `sep` |
  |---|---|---|
  | seen | 0.645 | 0.625 |
  | time | 0.580 | 0.553 |

  The pull comes from the predictor memorising each chat's skill prior, not from missing input. `prevskill` stays the default.

Route with the gate: `python skilljepa.py route "ok do it" --prev "watch this video and write notes" --prev-skill video-notes --hist 0`

## Follow-up detector (`gate_soft_followup`), negative result

**Rule.** A prompt is a follow-up when it is short (≤ w words) or barely prefers any skill on its own (zero-shot margin ≤ m). A follow-up repeats the previous turn's decision: that turn's skill, or abstain if it used none. The thresholds w and m are tuned on an inner 20% hold-out of each training fold.

**Tuning picked "never" on every split.** A sweep on the time split's test set shows why. Baseline is `gate_soft` (`prevskill`): Hit@1 0.580, accuracy-with-abstain 0.699.

| words ≤ | margin ≤ | prompts flagged | Hit@1 | accuracy-with-abstain |
|---|---|---|---|---|
| 0 | 0.005 | 18% | 0.578 | 0.706 |
| 3 | off | 20% | 0.521 | 0.718 |
| 0 | 0.04 | 74% | 0.541 | 0.731 |
| 12 | 0.04 | 88% | 0.511 | 0.727 |

- **Hit@1 drops at every setting.** Short prompts do not reliably reuse the last skill, and the `prevskill` context already gives the router the previous skill as a hint without forcing it.
- **The accuracy-with-abstain gain is misleading.** It comes from copying the previous turn's "no skill", which is what `repeat_prev_skill` does (0.734).
- **What would fix the pull:** better labels (which skill each turn actually needed) or an abstain head, not a heuristic override.

## Clean labels and an abstain head

**Clean labels (`--labels clean`).** `build_data.py` now also writes `skills_clean`, which drops skill files the turn only *edited* (Edit/Write, or a shell command writing a SKILL file). Those are mostly end-of-turn skill upkeep, not the skill the prompt needed. 636 of 4185 prompts keep a clean label (862 with all labels), and the first label changes on 264 prompts.

**Abstain head (`gate_soft_abstain_head`).** Routing comes from `gate_soft`; a separate logistic regression decides route vs abstain. Its inputs are the prompt embedding, the zero-shot best energy and margin, word count, whether the previous turn used a skill or none, and the chat's history. Its threshold is tuned on the train fold.

Hit@1 / accuracy-with-abstain, `--ctx prevskill`. Always abstain scores 0.794 with all labels and 0.848 with clean labels on the first two splits:

| labels | router | unseen chats | seen chats | time split |
|---|---|---|---|---|
| all | zero-shot | 0.499 / 0.802 | 0.484 / 0.801 | 0.550 / 0.677 |
| all | repeat_prev_skill | 0.419 / 0.801 | 0.415 / 0.801 | 0.520 / **0.738** |
| all | gate_soft | 0.499 / **0.802** | 0.647 / 0.820 | 0.578 / 0.707 |
| all | **gate_soft_abstain_head** | 0.499 / 0.794 | 0.647 / **0.826** | 0.578 / 0.735 |
| clean | zero-shot | 0.427 / 0.848 | 0.409 / 0.848 | 0.471 / 0.751 |
| clean | skill-jepa | 0.371 / 0.817 | 0.578 / 0.834 | 0.532 / 0.736 |
| clean | gate_soft | 0.427 / 0.848 | 0.585 / **0.860** | 0.545 / 0.748 |
| clean | gate_soft_abstain_head | 0.427 / 0.848 | 0.585 / 0.842 | 0.545 / **0.755** |

- **Abstain head: small, mixed gain.** It helps where the chat has history (seen chats +0.006; time split +0.028 with all labels, +0.007 with clean labels), but still does not beat `repeat_prev_skill` on the time split (0.735 vs 0.738). On new chats it falls back to always abstaining. Whether a turn needs a skill is mostly not visible in the prompt.
- **Clean labels: the trained predictor holds its lead.** Hit@1 drops for every router, because the easy "upkeep" labels are gone. The predictor's gain over zero-shot on seen chats stays about the same (+0.17 clean, +0.16 all). On new chats the gap between zero-shot and the predictor narrows (0.427 vs 0.371, was 0.499 vs 0.302), so much of the predictor's chat-prior pull came from upkeep labels.
- **Clean labels make abstention the main job:** 85% of prompts need no skill, and only `gate_soft` on seen chats (0.860) beats always abstaining.
- `gate_soft` with all labels stays the default. Use `--labels clean` to measure "the skill the prompt needed".

## Note: the production trigger/BM25 router, measured the same way (2026-10-23)

The router that actually runs on this machine is not skill-jepa but a hand-built one: weighted trigger words from each SKILL.md + BM25 over the skill bodies + a skill-name hit, with a tf-idf centroid re-ranker learned from past routes. We measured it with the same kind of label (skill read within 10 min of the prompt) over its own log: 1120 logged prompts, 684 with a label, no cross-validation (the centroid re-ranker is trained on the same log, so Hit@1 is optimistic).

| setting | Hit@1 | Hit@3 | abstain rate |
|---|---|---|---|
| plain | 0.621 | 0.835 | 0 |
| + hand-written aliases on 4 skills | 0.613 | 0.835 | <0.013 |
| + no-match floor 0.35-0.7 on the top score | 0.613 | 0.835 | <0.013 |
| + floor 0.9 | 0.598 | 0.814 | 0.02-0.04 |
| + floor 1.1 | 0.579 | 0.763 | 0.10-0.12 |

Takeaways, matching the skill-jepa results above:
- Hand-written aliases (ideas from the "Beyond Linear Context" critique) did not help: the trigger weights already cover the names, and an alias like "vram" pulls a skill onto prompts that needed another one.
- A score floor is not an abstention signal. Scores below the floor are rare, and raising it loses correct routes faster than it gains correct abstentions. Prompt-only signals do not say whether a skill is needed; see "Abstention is not solved yet".
- The hand router's Hit@1 (0.62) is above skill-jepa's seen-chats number (0.57) but the labels and splits differ (in-sample vs 5-fold CV), so this is not a fair comparison; it is the baseline the re-ranker idea below would sit behind.

## Next ideas

- Stop the prior pull: train with the prompt dropped (prompt dropout) so the predictor learns when to defer to the previous skill.
- Use the router as a re-ranker behind a trigger/BM25 router rather than as the primary router.
- Abstention needs a signal from outside the prompt, such as the agent's own first tool call.

MIT licence.
