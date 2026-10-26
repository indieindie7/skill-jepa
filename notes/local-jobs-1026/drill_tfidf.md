**Router Rerank Design Report**

**Design & Methodology**
*   **Method:** TF-IDF word + char-trigram centroids per skill (L1167-L1169). Built from `ops/route.log` queries paired with `~/.claude/skills/usage.jsonl` rows within 600s (L1226-L1234).
*   **Logic:** Rerank-only tier for `opsd.route()` within text top-5. If a confident centroid match is in the top 5 but not position 0, it moves to position 0 with conf "high" (L1209, L1326-L1327). Otherwise uses trigger+BM25 (L1204).
*   **Floor Decision:** Initially tested at 0.15; raised to 0.25 after errors (L1209, L1317, L1344). Floors tested: 0.15/0.25/0.35/0.5 (L1167-L1169).

**File Paths & Commands**
*   **Test Script:** `C:\Users\john\Documents\Tools\ops\idea10_fuzzy_router.py` (L1167).
*   **Target File:** `ops/opsd.py` in `C:\Users\john\Documents\Tools\ops\opsd.py` (L1183, L1321).
*   **Backup:** `.bak-20261017l` (L1183) and `.bak-20261017m` (L1321).
*   **Implementation Details:** Reuse feats/vec/centroid code from test script; build index at startup, refresh on `usage.jsonl` mtime change or every 10 min (L1183, L1315).

-- localread: ws2.sel.txt, lines 1100-1350 of 7560, 3 part(s), 70s, Strata qwen3.8-flash-next-iq3_xxs on port 8090. Verify cited lines before relying on them.
