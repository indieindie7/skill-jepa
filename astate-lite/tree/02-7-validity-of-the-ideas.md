## 7. Validity of the ideas (as of 6 November 2026, v2.2)
Evidence is small: one user's 4,169 prompts (public skill-jepa repo), single-corpus compression runs, and from now on your own weekly line. Nothing has a 1k-task corpus; the weekly line is not independent validation. Every verdict is provisional.

| Idea | Verdict | Evidence so far | Confidence |
|---|---|---|---|
| Previous skill as minimal state (`active skill`) | Keep | Repo, read 2026-11-06: previous-skill zero-shot Hit@1 0.500 unseen / 0.484 seen; repeat-previous-skill 0.419 / 0.415; previous+current prompt text 0.293 / 0.317. Cheap and easy to disprove. One user; skill labels may run in streaks, which would flatter it | Medium |
| ASTATE as routing context | The one experiment (section 5) | No result yet; neither README read (11-05, 11-06) has an ASTATE-labelled run. Pass bar fixed before running | — |
| Gate: user asked, or a confirmed claim requires it | Keep, simplified | Repo: ~79% of prompts use no skill, so always-abstain scores 0.793 — class imbalance, not proof that abstaining helps. v2.1's "every `requires` item confirmed in STATE" would have blocked nearly every load, since most preconditions are never recorded. No daily-use test | Low–Medium |
| Append-as-you-go LOG + checkpointed STATE, bare | Keep, bare | Event-sourcing pattern supports the shape, not this format. `supersedes` / `rejects` / archive rotation dropped: no logged failure needs them | Medium for the shape, Low for the format |
| Claim · status · pointer | Keep, cut | Clean-room mechanical tests only; no shown gain over notes + search. IDs, [SRC]/[SYN]/[ø], verified, recheck, since cut: they duplicated status and pointer, and the upkeep would end daily use before any benefit showed | Low–Medium |
| Contracts: `requires` + `effects` | Keep, cut | Analogy to ECS / engine init order; no test. `reads`, `produces`, `conflicts` cut: they fix no logged failure and cannot be tested here | Low |
| Stop rule (name what the next step decides) | Keep | Reasoning only. The fixed "5 steps" was arbitrary and is dropped | Low |
| Custom codecs (ASTATE+R, AR4/5, ASTR-S/SR) | Killed | Measured: AR4+ASTATE 55,871 B vs zlib 46,615 B; AR5 59,883 B vs 49,599 B; Brotli 11 best size, Zstd 3–10 best speed — on those corpora only. The only conclusion here that was actually measured | High (one corpus each) |
| JEPA router | Killed (was Park) | Repo, read 2026-11-06: with previous-skill context Hit@1 0.647 seen / 0.304 unseen, against 0.500 unseen for the zero-shot previous-skill router; prompt-only JEPA 0.568 seen / 0.234 unseen. That gap is not noise and a small corpus cannot rescue it; seen-chat wins read as memorisation. Maintainer reportedly dropped the project (user, 11-05); MIT, so a branch could continue | Medium (enough to kill, not to revive) |
| "It's a state problem" | Reworded | Measured: the previous skill predicts the next one. Not measured: that a structured state file helps. Only the experiment above can show that | — |
| SHA-256 receipts / zstd archives | Dropped | Benefit untestable at this scale; a link and a read date cover recovery in practice | — |
| Repetition DAG, small-local-LLM target, 1k-bundle benchmark | Killed | Nothing to measure and no plan to measure it | — |
| Six weekly metrics | Cut to 2 + cold test | One person logging weekly cannot separate six signals; task success, maintenance minutes and tokens-at-open dropped | — |
