[part 1, lines 1-90]
1. **Data Leakage / Overcounting**: L38 admits replaying old `route.log` is "overcounted" and uses a loose time-window join, risking train/eval contamination if historical traces overlap with the eval set defined in L45/L54. Fix: Strictly separate trace generation (L29) from evaluation splits; exclude any sid present in the 48-row `--sid-only` eval set (L13) or time-split boundaries before training.

2. **Unrealistic Pass Criterion**: L65 requires student Hit@1 within 2 points of teacher, but L17 notes current best unseen performance is only 0.50 vs seen 0.647. Distilling a tiny model (L58) to match a GRPO policy that itself struggles with generalization (L54) is optimistic without explicit regularization for unseen data. Fix: Add a "generalization gap" metric to the pass criteria, requiring student performance on *unseen* splits to be within 2 points of teacher's *unseen* performance, not just overall Hit@1.

3. **Hardware/VRAM Conflict**: L51 states GRPO fits in 12GB (3060) but L74 mandates "never gpu 0 while gaming" and L84 requires a shared GPU picker. If the game runs on the 3060, training halts; if it runs on the 4070 Ti, the router must fit there too. Fix: Implement dynamic VRAM reservation in `llm/gpu.py` (L84) that aborts or pauses GRPO/mixd loading if `nvidia-smi` detects a game process on *any* target GPU, not just gpu 0.

4. **Missing Step for Judge Reliability**: L35 mentions using a judge pass only when `usage.jsonl` is ambiguous, but no criteria define "ambiguous" or validate the judge's accuracy against human labels. This risks propagating noise into rewards (L28). Fix: Before step 1 completion, manually label 50 ambiguous rows to measure judge precision/recall; if <90%, restrict training to unambiguous traces only until volume suffices.

5. **Vocabulary Mismatch in Distillation**: L63 notes token-level OPD needs "shared vocabulary," but teacher (Qwen/Gemma) and student (EmbeddingGemma/MLP) likely have different tokenizers.

-- localread: SELFIMPROVE_PLAN.md, lines 1-90 of 90, 1 part(s), 152s, Strata qwen3.8-flash-next-iq3_xxs on port 8090. Verify cited lines before relying on them.
