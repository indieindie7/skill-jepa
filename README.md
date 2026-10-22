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
```

- **Endpoint:** set `SKILLJEPA_EMB_URL` to use a different embeddings endpoint.
- **Requirements:** NumPy and Python 3.10+.

## Next ideas

- Add the previous turn to the context (fixes follow-up prompts).
- Use cleaner labels (first skill read only, or the skill a hook logged as useful).
- Train a separate abstain head.
- Use the router as a re-ranker behind a trigger/BM25 router rather than as the primary router.

MIT licence.
