**Skill-JEPA Router Design, Data & Results**
*   **Repo/Paths:** Code at `C:\Users\john\Documents\Tools\skilljepa` (`build_data.py`, `skilljepa.py`). Repo: `github.com/indieindie7/skill-jepa`. Memory file: `~/.claude/projects/C--/memory/skill-jepa-prototype.md` [L5245-L5246, L5278].
*   **Commits:** `1c8bab4` (Prototype), `20ac085` (`--ctx`, `embedctx`), `d6b2d3b` (gate router, time split), `bcdd0c3` (per-branch tau), `cd4e450` (follow-up detector, NEGATIVE), `99c01ca` (clean labels + abstain head), `aa8ff56` (README note on aliases) [L5260-L5264, L5379, L6223].
*   **Data:** Mined from `~/.claude/projects/*/*.jsonl` to `data/pairs.jsonl`. Initial: 4169 prompts/862 labelled. Cleaned labels reduced to 636/4185 [L5061-L5079, L5387].
*   **Architecture:** Frozen EmbeddingGemma via `http://127.0.0.1:8084/v1/embeddings`. Predictor is residual MLP (768→512→768). Loss: `(1−cos) + α·InfoNCE`. Routing energy `1−cos` with abstain threshold `tau` [L5061-L5079].
*   **Results:** Hit@1 for seen chats 0.57, new chats 0.23 (early). Later: zero-shot 0.50 vs JEPA 0.304 on unseen; gate_soft 0.50. Always-abstain score 0.793 [L5024-L5038, L5269-L5273]. Abstain head: Seen 0.826, Time split 0.735, New 0.794 [L5379-L5385]. Production trigger/BM25 router Hit@1 dropped from 62.1 (plain) to 61.3 with aliases; gain -0.8 rejected [L6209-L6217, L6343].
*   **Decisions:** "No skill" detection unsolved via prompt alone.

-- localread: ws2.sel.txt, lines 5001-7560 of 7560, 18 part(s), 867s, Strata qwen3.8-flash-next-iq3_xxs on port 8090. Verify cited lines before relying on them.
