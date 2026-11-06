<!-- branch of skill astate-lite (root: ~/.claude/skills/astate-lite/SKILL.md); paths are relative to the skill folder -->
## 7. Validity of the ideas (as of 5 November 2026)

Evidence is small: one user's 4,169 prompts (public skill-jepa repo) plus single-corpus compression runs. Nothing has a 1k-task corpus. Every verdict is provisional until the weekly metric says otherwise.

| Idea | Verdict | Evidence so far | Confidence |
|---|---|---|---|
| Deterministic gate instead of learned abstention | Adopt | Public repo: ~79% of prompts use no skill; always-abstain (0.793) unbeaten by any tested router. No daily-use test yet | Medium |
| JEPA router | Park | Hit@1 0.568 vs 0.233 zero-shot on seen chats, 0.234 vs 0.225 on unseen; one user's data; maintainer inactive | Medium (data) / Low ("it is a state problem" is inference) |
| Skill contracts (requires, reads, produces, effects, conflicts) | Adopt | Analogy to ECS (Bevy, Flecs) and engine init-order problems; no direct test | Low to Medium |
| Append-only LOG + derived STATE | Adopt (shrunk) | Event-sourcing literature, including its complexity warning; no test | Medium for the shrink, Low for any specific format |
| ASTATE claim, evidence, conflict skeleton | Adopt skeleton only | Passed clean-room mechanical tests; no shown economic gain vs RAG + compression | Low to Medium |
| Custom codecs (ASTATE+R, AR4/5, ASTR-S/SR) | Kill, keep receipts | Measured: AR4+ASTATE 55,871 B vs zlib 46,615 B; AR5 59,883 B vs 49,599 B; Brotli 11 best size, Zstd 3–10 best speed | High (measured, one corpus each) |
| Bounded explore budget (act, explore, abstain) | Adopt as a rule | Reasoning only | Low to Medium |
| Repetition DAG, variable-depth labyrinth | Park | No test; daily volume too small to need it | Low |
| Small local LLM as target | Keep as target | Reasoning only | Low |
| 1k-bundle benchmark | Replace by weekly metric | Not affordable now; the dogfood metric is the test | n/a |
