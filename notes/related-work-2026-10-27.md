# skill-jepa related work (web scan 2026-10-27)

Context: 77 skills, ~862 labelled prompts, ~79% "none". Best: BM25+prev-skill prior (hit@1 .646); AVB attention head over EmbeddingGemma (hit@1 .60, acc-with-abstain .75); ES + LoRA SFT of 0.8B policy parked.
Source: web search/fetch only. Only SkillRouter and Tool2Vec pages were fetched; the rest come from search snippets or secondary summaries (marked unverified where shaky). Check the paper before relying on a number.

## Index (ranked by usefulness to us)
1. SkillRouter (2603.22455): full-text retrieve+rerank for skills, listwise loss, false-negative filter. Code.
2. Tool2Vec / ToolRefiner / MLC (2409.02141): usage-driven tool embeddings, small reranker, multi-label head. Code.
3. Deep Gamblers + SelectiveNet: abstain as a learned (m+1)-th class / selection head.
4. Tool-DE / Tool-Embed / Tool-Rank (2510.22670): LLM-expanded tool docs (when-to-use, limits).
5. ToolRet + ToolRet-train (2503.01763): tool retrieval benchmark and 200k training tasks.
6. AgentAbstain + When2Tool (2607.10059, 2605.09252): paired should-act/should-abstain eval; linear probe for tool necessity.
7. Small reranker distillation: Ettin rerankers, Rank-DistiLLM, 2603.03010 (loss comparison).
8. Model2Vec: static-embedding distillation of a sentence encoder; CPU-tiny.
9. RouteLLM: threshold calibration to a target traffic share; MF router for low data.
10. Diverse Skill Routing, DPP (2609.05824): set selection for multi-skill prompts.
11. SkillRet (2605.05726): skill retrieval benchmark, 63k train queries, fine-tune gains.
12. SkillResolve-Bench (2606.10388): sibling-skill confusion eval.
13. LLM-JEPA (2509.14252): JEPA loss for text, view pairs. Code.
14. aurelio-labs semantic-router: per-route thresholds, None route, threshold optimiser. Code.
15. MCP-Zero (2506.01056): hierarchical server->tool routing, LLM-augmented descriptions, dataset.
16. Counterfactual / implicit-feedback learning-to-rank (Radlinski-Joachims): debiasing self-training from own logs.
17. LoRA Without Regret (Thinking Machines) + TRL guide: rank-1 LoRA for RL, caveats.
18. EGGROLL + "EGGROLL, Unrolled" (2511.16652, 2609.10980): low-rank ES, LOO-ROLL estimator.
19. Arch-Router 1.5B (2506.16655): generative route-description router, add routes without retraining.
20. LiveMCPBench + TheMCPCompany (2508.01780, 2510.19286): retrieval-error-dominated agent benchmarks.

Also (not ranked): Voyager skill library, ToolBench/ToolLLM BERT API retriever, Gorilla retriever-aware training (RAT), RAG-MCP.

---

## 1. SkillRouter
- Link: https://arxiv.org/abs/2603.22455 ; code https://github.com/zhengyanzhao1997/SkillRouter
- Does: two-stage routing over ~80K skills; Qwen3-Emb-0.6B bi-encoder (in-batch InfoNCE, tau .05), then Qwen3-Rank-0.6B cross-encoder over the top-20; 74.0% Hit@1, 5.8x faster than a 16B pipeline.
- Borrow:
  - Rerank loss: listwise CE beats pointwise BCE (74.0 vs 43.3 Hit@1). Switch the AVB head to listwise CE if it is not already.
  - False-negative filter on mined negatives: name match, trigram Jaccard > .6 on body, cosine > .92 (+4 pp Hit@1). Our 77 skills have many near-duplicates, so apply it to kNN negatives.
  - Negative mix per query: 4 semantic, 3 BM25, 2 same-category, 1 random.
  - Feature: embed the skill BODY, not only name+description. Removing the body costs 31-44 pp. Test EmbeddingGemma over chunked SKILL.md text, max-pooled.
  - Synthetic queries: an LLM writes queries from the skill body without naming the skill. Local-model job to grow our 862 labels.

## 2. Tool2Vec / ToolRefiner / MLC
- Link: https://arxiv.org/abs/2409.02141 ; code https://github.com/SqueezeAILab/Tool2Vec
- Does: tool embedding = mean of embeddings of example user queries that used the tool (not the description); ToolRefiner = DeBERTa-v3-xsmall cross-encoder on (query, candidate tool embeddings); MLC = multi-label BCE head over tools. Up to +27 Recall@K on ToolBench.
- Borrow:
  - Per-skill prototype = mean of past-prompt embeddings, mixed with the description embedding. Our kNN idea as a cheap centroid; add as a feature next to BM25.
  - An MLC head (sigmoid per skill, BCE): "none" = all zeros, and handles prompts that use two skills.
  - ToolRefiner is the template for a <100M reranker (xsmall encoder) distilled from the current stack.

## 3. Deep Gamblers + SelectiveNet
- Links: https://arxiv.org/abs/1907.00208 ; https://arxiv.org/abs/1901.09192
- Does: Deep Gamblers adds an (m+1)-th "abstain" output with a portfolio-theory loss; SelectiveNet trains a selection head jointly with the classifier under a target-coverage constraint.
- Borrow:
  - Gambler's loss with the abstain class = "none". With ~79% none, the threshold tau is fragile; learning it end to end may lift acc-with-abstain (.75 vs always-abstain .793).
  - Report risk-coverage curves, not one threshold: selective risk at fixed 30/50/70% coverage is a cleaner metric than accuracy-with-abstain.

## 4. Tool-DE / Tool-Embed / Tool-Rank
- Link: https://arxiv.org/abs/2510.22670 (ICLR 2026); repo probably EIT-NLP/Tool-REX (unverified, benchmark name differs between versions).
- Does: an LLM rewrites each tool doc into structured fields (function description, when to use, limitations); trains Tool-Embed (50k) and Tool-Rank (200k) on it; SOTA on ToolRet.
- Borrow: for each of the 77 skills, have the local model write "when to use" and "when NOT to use" fields; embed them as extra targets y_s in the JEPA energy and add the text to BM25. The "not use" field targets abstention. Cheap, offline, one-off.

## 5. ToolRet + ToolRet-train
- Link: https://arxiv.org/abs/2503.01763 (Findings of ACL 2025)
- Does: 7.6k retrieval tasks over 43k tools; strong IR models score low (NV-Embed-v1 ~34 nDCG@10, secondary summary); ToolRet-train (200k+ tasks) lifts retrievers and tool-use pass rate 10-20%.
- Borrow: report nDCG@10 and Recall@K next to hit@1; instruction-aware query prefix for EmbeddingGemma; try a slice of ToolRet-train as extra pre-training pairs for the head (domain mismatch risk: check on our CV folds).

## 6. AgentAbstain + When2Tool
- Links: https://arxiv.org/abs/2607.10059 (project site agentabstain.github.io per search) ; https://arxiv.org/abs/2605.09252
- Does: AgentAbstain = 263 paired tasks (should-act vs should-abstain variant) in 42 sandboxes; best of 17 models 59.5% paired accuracy. When2Tool: a linear probe on hidden states predicts tool necessity, AUROC .89-.96; Probe&Prefill cut tool calls 48% at 1.7% accuracy cost.
- Borrow:
  - Eval: build paired probes from our logs (a prompt that used a skill vs a minimal edit that should not); report paired accuracy.
  - Method: logistic probe on frozen EmbeddingGemma vectors (+ ctx features) for "any skill needed?" as a separate gate before the skill head. Compare against the energy threshold tau.

## 7. Small reranker distillation
- Links: Ettin rerankers (HF announcement; 17M-1B; pointwise MSE distillation from mxbai-rerank-large-v2; 17M beats 33M MiniLM-L12 by +.051 nDCG@10, unverified) ; https://arxiv.org/abs/2405.07920 (Rank-DistiLLM) ; https://arxiv.org/abs/2603.03010 (distillation study: MarginMSE and listwise InfoNCE beat pointwise across BERT/RoBERTa/ELECTRA/DeBERTa/ModernBERT).
- Borrow: when distilling the teacher (BM25+prior+AVB, or a local-LLM judge) into a <100M cross-encoder, use MarginMSE or listwise InfoNCE over the top-20, not pointwise. Backbones to try: Ettin-17M/32M, MiniLM-L6, DeBERTa-v3-xsmall. Teacher scores from logs plus local-LLM relabelling.

## 8. Model2Vec
- Link: https://github.com/MinishLab/model2vec
- Does: distils any sentence transformer into a static token-embedding model (PCA + Zipf weighting) in ~30 s on CPU; claims up to 50x smaller, 500x faster; ready models potion-base-8M/32M.
- Borrow: distil EmbeddingGemma into a static model so routing no longer needs the 8084 server (numpy only). Measure hit@1 loss versus EmbeddingGemma; also usable as a fast first-stage candidate generator.

## 9. RouteLLM
- Link: https://github.com/lm-sys/RouteLLM ; https://arxiv.org/abs/2406.18665
- Does: routers (matrix factorisation, BERT, causal-LLM, SW-ranking) trained on preference data; `routellm.calibrate_threshold` picks the threshold that sends a chosen % of traffic to the strong model. No abstention mechanism.
- Borrow: calibrate tau the same way: choose the threshold for a target fraction of routed prompts on held-out data instead of tuning on training data. The MF router (prompt embedding x skill embedding, learned bilinear score) is a low-data baseline for 862 labels.

## 10. Diverse Skill Routing (DPP)
- Link: https://arxiv.org/abs/2609.05824
- Does: reranks the shortlist as a set with a determinantal point process using a query-residual diversity kernel; gains on multi-skill queries.
- Borrow: only if we go multi-label (prompts that load 2 skills). The query-residual kernel (remove the shared query component before computing skill-skill similarity) is a small trick for our skill clusters.

## 11. SkillRet
- Link: https://arxiv.org/abs/2605.05726
- Does: ~16-18k public skills, ~4.4-5k eval queries (sources disagree), 63k training queries; best off-the-shelf 70.26 nDCG@10, fine-tuned 86.44. No code found.
- Borrow: its query-generation recipe and the +13 nDCG fine-tuning result; use as a second eval domain (public skills) to check we are not overfit to one user's 77 skills. Check whether the data is released.

## 12. SkillResolve-Bench
- Link: https://arxiv.org/abs/2606.10388
- Does: 661 helpful/risky sibling pairs, 7,982-skill pool; measures harmful-sibling exposure at top-K.
- Borrow: metric "wrong-sibling rate": among errors, how often the prediction is a near-duplicate of the right skill. Report per skill cluster; guides which skills to merge or whose descriptions to sharpen.

## 13. LLM-JEPA
- Link: https://arxiv.org/abs/2509.14252 ; code github.com/rbalestr-lab/llm-jepa (ICLR page lists galilai-group/llm-jepa; unverified which is current)
- Does: adds a JEPA term (predict the embedding of view B from view A, cosine-type distance) to the usual LM loss; views = NL description vs code.
- Borrow: our predictor already follows this. Borrow the multi-view data trick: views = (user prompt, LLM paraphrase, skill description, skill example) tied by the predictor loss; and their finding that the JEPA term resists overfitting, useful at 862 labels. The LoRA-SFT branch could take it as an auxiliary loss.

## 14. semantic-router (aurelio-labs)
- Link: https://docs.aurelio.ai/semantic-router/user-guide/features/threshold-optimization.md ; repo github.com/aurelio-labs/semantic-router
- Does: embedding routes with example utterances; per-route `score_threshold`; returns None when nothing passes; optimiser fits thresholds on labelled data.
- Borrow: per-skill tau (some skills are easy, some confusable) fit on dev folds with shrinkage toward the global tau. Behaviour changed in v0.1.3; read the source before copying.

## 15. MCP-Zero
- Link: https://arxiv.org/abs/2506.01056
- Does: agent emits a tool request; two-stage vector routing (server then tool); LLM-summarised server descriptions; released dataset of 308 servers / 2,797 tools.
- Borrow: our skills already form a tree (ops-manager tree/ branches). Route cluster first, skill second; compare to flat. Their dataset is an out-of-domain retrieval test.

## 16. Counterfactual LTR / learning from implicit feedback
- Links: https://www.cs.cornell.edu/~tj/implicit ; query chains (Radlinski and Joachims)
- Does: classic work turning click/reformulation logs into preference pairs and correcting exposure bias.
- Borrow: nightly self-training only sees skills the router (or user) surfaced. Log the router's top-k and scores, weight self-labels by inverse propensity, or treat "user re-prompted / loaded another skill" as a preference pair (chosen > shown). No agent-specific paper found; this is the closest principled fix for the feedback loop.

## 17. LoRA Without Regret
- Link: https://thinkingmachines.ai/blog/lora ; https://huggingface.co/docs/trl/main/en/lora_without_regret
- Does: claims rank-1 LoRA matches full fine-tuning for RL (policy gradients carry ~1 bit per episode); needs MLP layers adapted, LR about 10x full-FT. An HF replication on Qwen3-0.6B found full-FT pulls ahead (only partly holds); successful rank-1 reports were on math.
- Borrow: for the parked 0.8B policy use all linear layers (incl. MLP), LR ~10x, and run a rank sweep (1, 4, 16) before declaring ES/RL failed. The TRL guide has a working config.

## 18. EGGROLL + "EGGROLL, Unrolled"
- Links: https://arxiv.org/abs/2511.16652 ; https://arxiv.org/abs/2609.10980
- Does: ES with low-rank perturbations, ~91% of batch-inference throughput, competitive with GRPO on reasoning; the follow-up shows rank-1 perturbations can bias the update and proposes LOO-ROLL (leave-one-out, one eval per direction, halves estimator MSE at equal evals; preprint).
- Borrow: if ES comes back, replace antithetic pairs with the leave-one-out baseline and avoid rank-1 perturbations. Our routing reward is cheap and exact, so large populations are affordable.

## 19. Arch-Router
- Link: https://arxiv.org/abs/2506.16655 ; model https://huggingface.co/katanemo/Arch-Router-1.5B
- Does: 1.5B generative router; the prompt lists user-defined route descriptions (domain/action) and the model outputs a route; new routes need no retraining.
- Borrow: prompt template and output format for the 0.8B LoRA-SFT policy (route list in context, output a name or "none"). Test whether adding a skill needs no retraining; that would remove the nightly retrain for new skills.

## 20. LiveMCPBench + TheMCPCompany
- Links: https://arxiv.org/abs/2508.01780 ; https://arxiv.org/abs/2510.19286
- Does: agent benchmarks with 527 tools / 70 servers (retrieval errors ~half of failures) and 18k+ tools with gold tool annotations.
- Borrow: a cheap end-to-end check: replay a few agent tasks with router picks vs gold skills to turn hit@1 into task success. The "retrieval is the bottleneck" finding supports investing in the router.

## Not ranked
- Voyager: https://arxiv.org/abs/2305.16291 ; skills indexed by embedding of their description, retrieval query = task plan + environment feedback. Borrow: query = prompt + summary of the last assistant action (we have ctx via prev-skill).
- ToolBench/ToolLLM BERT API retriever: https://github.com/OpenBMB/ToolBench ; trained retriever weights on HF.
- Gorilla RAT: https://arxiv.org/abs/2305.15334 ; retriever-aware training puts retrieved docs in the training prompt. Borrow: train the LoRA policy with the router top-k in the prompt.
- RAG-MCP: https://arxiv.org/abs/2505.03275 ; retrieval before the LLM sees tools cuts prompt tokens ~73% (secondary). Baseline for the token-saving claim.

## Gaps
- No public paper found for self-training a skill/tool router from its own usage logs (item 16 is the closest); may be our novelty.
- No code found for SkillRet, When2Tool; AgentAbstain code only via its project site.
- No sub-100M reranker distilled from an LLM teacher found for tool/skill routing specifically.
