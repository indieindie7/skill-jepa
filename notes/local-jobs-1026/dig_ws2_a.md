**Skill-Jepa Router (Design, Data, Results, Decisions)**
*   **Design:** The router (`opsd.py`) combines trigger matching with a "fuzzy first-tier" decision model (Idea #10). It uses log-learned tf-idf centroids (words + char trigrams) from logs to rerank the top-5 results. A BM25 body-aware layer (Idea #9) was tested but rejected due to negative gain (-0.056) [L931, L1470]. Idea #8 (SmolLM fallback) was archived because the model wasn't on disk [L1207, L1318].
*   **Data:** Training/evaluation data comes from `ops/route.log` and `~/.claude/skills/usage.jsonl`. Truth labels are derived from the skill used in the next turn of usage logs. Requires ~50 rows to start [L264, L295].
*   **Results:** Idea #10 was approved with a gain of 0.082 (Hit@1 improved from 24/49 to 28/49 in dry runs) [L1317, L1406]. Live rerank tests were written (`idea12_rerank_live.py`) [L1483-L1484].
*   **Decisions/Rules:** Fuzzy tier is restricted to **reranking top-5** only (no free override) because `route.log` lacks session IDs. Floor score raised to 0.25 [L1181, L1468]. Never persist route scores into `index.json`. Pareto optimization required (success + tokens) [L264-L265].
*   **Paths/Commands:** File: `C:\Users\john\Documents\Tools\ops\opsd.py` [L31, L128]. Test script: `ops\idea10_fuzzy_router.py` [L1167]. Probe command: `curl.exe "http://127.0.0.1:8765/api/route?q=..."` [L1459-L1460].

-- localread: ws2.sel.txt, lines 1-2500 of 7560, 21 part(s), 388s, Strata qwen3.8-flash-next-iq3_xxs on port 8090 (left running). Verify cited lines before relying on them.
