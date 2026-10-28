# Mindfield "Sparse-Savant" design doc (user photos 10-28; mindfield_sparse_s..._190917.pdf, Copilot-made, work laptop)
Index: 1 verdict+flip | 2 unresolved assumptions | 3 minimal experiment | 4 ablations | 5 governed mutation | 6 map to ledger | 7 batch 2: inference chain, authority table, arch A/B/C, H1-H6, roadmap, do-not-build | 8 our position (grep -n "^## " for lines)

## 1. Verdict + single flip assumption
- Order: do NOT test numogram topology first. (1) show frontier control works, (2) does JEPA add OOD information, (3) only then compile winning controller into ten-zone + conventional state-machine projections.
- Flip assumption (conf mu=0.80): Sparse-Savant frontier score incl. novelty reserve is predictive enough of downstream task value that aggressive pruning loses less than it saves. If false -> core shifts to broader/adaptive retrieval; JEPA + numogram secondary.
- Numogram falsifier: conventional state machine matches/beats it with fewer adapters/exceptions -> numogram = visualization/namespace only.

## 2. Unresolved assumptions (doc sec 7, ten)
1. Sparse context improves total utility, not only token count.
2. Weak signals pruned now have measurable later value.
3. Novelty/disagreement estimable without flooding the reserve with noise.
4. JEPA adds conditional information beyond symbolic retrieval + graph scoring.
5. Latent transition prediction generalizes across projects, sessions, time, unseen skill descriptions.
6. Routing evidence causally separable from execution/environment/outcome evidence.
7. Ten zones outperform an ordinary typed DAG/state machine.
8. Graph mutations compile into durable code/config without semantic drift/round-trip loss.
9. Audit criteria can stay external enough to resist recursive optimization.
10. Deterministic projection does not erase useful stochastic exploration before execution.

## 3. Minimal next experiment (doc sec 8): fixed budget, three arms, held out
- A0 symbolic Sparse-Savant, exploit-only frontier.
- A1 same controller + frontier size, small novelty reserve.
- B1 A1 + clipped advisory JEPA reranking.
- Identical retrieval results, total frontier K, injected tokens, executor, evidence gate. Task types: exact, paraphrase, confusable, no-match, composite/dependency, long-session, delayed-evidence.
- Record: route accuracy, false activation, abstention quality, premature-pruning failures, end-task success, evidence attribution, tokens, latency.

## 4. Ablation table (doc sec 5; freeze corpus/graph/executor/hardware/K/budgets; pre-register thresholds)
| ablation | compare | measure | reject if |
|---|---|---|---|
| sparse control | lexical top-k vs symbolic Sparse-Savant | task success, routing acc, tokens, latency | no task-value gain |
| weak reserve | r=0 vs r>0 same K | delayed-relevance recovery, premature-pruning failures | reserve adds cost/errors only |
| JEPA increment | symbolic vs symbolic+JEPA | top-1/Recall@k, false activation, abstention, task success | gains vanish OOD |
| state inputs | prompt vs prompt+state vs state+graph pos | transition acc/calibration | context model does not generalize |
| evidence quality | route label vs execution/tool/objective evidence | attribution rate, provenance coverage, verifier agreement | labels remain contaminated |
| numogram | ten-zone compiler vs typed DAG | invalid transitions, trace completeness, recovery defects, complexity | no measurable advantage |
| mutation | none vs governed candidate promotion | held-out correctness, regressions, rollback success | internal score rises while external holdout fails |
- Random splits = diagnostic only; deployment evidence must hold on session-, project-, temporal-, unseen-skill splits.

## 5. Governed mutation + recursive Goodharting (doc sec 6)
- Pipeline: propose -> sandbox -> unit/property/security tests -> differential eval vs frozen baseline -> held-out project/session/time eval -> human-reviewed promote|reject -> canary/shadow -> rollback on breach.
- Immutable/external: held-out sets + labels, promotion thresholds, evaluator impl + hashes, protected constraints/policy gates, evidence/audit log schema, rollback artifacts, authority over mutation/audit machinery.
- Failure def: promotion score improves while untouched external holdouts, adversarial tests, evidence completeness or human review degrade.
- Tripwires: patch edits evaluator/thresholds/logs/exclusions/provenance; abstention hidden by conditional accuracy; easy tasks routed into measured population; evidence deleted/reclassified after failures; mutations optimize acceptance proxies.
- Anything touching audit/eval = separate governed change. Automatic online weight/code updates = premature.

## 6. Map to our ledger (context-nudger-PLAN.md sec 2)
- doc 1 (sparse context = utility) ~ our A6 (entropy collapse readout; no utility measure yet).
- doc 2+3 (weak reserve, novelty w/o noise) = our A3 (explore random vs usual-suspects; split slot live 661d362; need 30+ rows).
- doc 4 (JEPA adds info) = JEPA chat's AVB-vs-BM25 question; AJAX file-split .448 says symbolic alone is weak OOD; AVB increment untested on file split.
- doc 6 (routing evidence separable) = our A4 (refute templates = f(conf,class): labels ARE contaminated by read behaviour) + A1 (judge vs router gap half context, half bias).
- doc 9 (external audit) = our closed-loop worry (A1); judge_fresh = the only external-ish evaluator; its hold-out sets are not frozen/hashed yet.
- doc 5, 7, 8, 10 = no data here; 7/8 explicitly deferred by the doc itself.
- NEW A8 = doc flip assumption (frontier score + reserve predicts task value, mu .80). Our retro proxy: suspect/explore hit rate vs full_read rate per conf; falsifier = reserve hits ~ random AND pruning (brief_only) rows show later full reads of unranked skills.
- 3-arm experiment maps to existing pieces: A0 = BM25 top-k (ajax_eval bm25), A1 = + explore slot (route.log col 6), B1 = + AVB rerank clipped. Held-out = file/session split (AJAX precedent). Owner: JEPA chat (eval code); this chat: pre-registered thresholds + readout.

## 7. Second photo batch 10-28 (doc title: "Mindfield -> Sparse-Savant -> Bounded JEPA -> Auditable Control Plane"; scope = architecture as provided, NOT verified; no code diff shows which recommendations are live, no benchmark yet)
Known: Mindfield/Labyrinth = possible typed relationships, deterministic projection = auditable order; JEPA already bounded correction, safeguards/abstention explicit; prior JEPA improved seen-session prompts, generalized WEAKLY to held-out sessions.
Missing: sparsity -> total task value (not token count); JEPA info beyond symbolic/graph; ten zones > typed state machine; causal trace route -> injected procedure -> execution -> outcome.
### Shortest defensible inference chain (sec 1)
1 linkage-first: working object = typed graph not ranked doc list. 2 competing paths not one narrative ("attractors" = repeated scoring makes paths dominant, operational not metaphysical). 3 auditability = projection not graph removal (freeze graph revision, stable eligibility + tie-breakers, emit topological trace). 4 finite context -> reasoning = allocation: which frontier nodes get context/retrieval/prediction/execution budget. 5 Sparse-Savant = controller (prune/expand by evidence, IG, uncertainty, cost, risk, provenance; bounded beam; scoring policy still a hypothesis). 6 JEPA local + advisory, only over eligible frontier, small residual subordinate to symbolic scores. 7 prediction cannot authorize execution (evidence+policy decide EXECUTE|EXPAND|SEARCH|ABSTAIN|REJECT; routing/execution/environment/outcome evidence channels distinct). 8 runtime = typed control plane; ten-zone shape optional.
=> defensible arch != "JEPA reasoning": symbolic eligibility+budget -> bounded prediction -> evidence-gated action -> verified state update.
### Sec 2 central tension: weak-signal preservation vs sparsity
Pure top-k = attractor problem (early high scores get more evidence, score higher; weak novel paths die before legible). Frontier K = F_exploit U F_reserve; reserve = low-conf, high novelty/disagreement/causal centrality/IG; nodes, retrieval calls, injected tokens fixed. Score s_i = R_i + lambda*IG_i + eta*Novelty_i + kappa*Disagreement_i - alpha*ContextCost_i - beta*Risk_i.
Falsifier (r>0): under equal K/tokens/retrieval, r>0 fails to improve delayed-relevance recovery or end-task success and raises false activation/latency/irrelevant context -> remove novelty preservation or enable only under detected distribution shift. Question = WHICH info is preserved after controlling quantity.
### Sec 3 authority decomposition (layer: authority / output / prohibited promotion)
Mindfield: typed relationships / candidate graph+conflicts / edge != truth. Sparse-Savant: eligibility, budget, retention, gating / bounded frontier / score != calibrated probability. JEPA: advisory local prediction / transition utility + uncertainty / prediction != permission or evidence. Evidence layer: validates claims + exec preconditions / accept|reject|unknown / usage != correctness. Audit/memory: governs durable state + learning admission / provenance-linked observations / outcome != router credit without attribution.
Coherent iff 4 boundaries: JEPA adds no out-of-policy candidates; JEPA weight clipped or separately reported; only externally verified observations become durable facts; audit/promotion machinery outside the mutation surface. (= controlled-conversion model: latent -> symbolic candidates w/ uncertainty; write-back needs observation+verification.)
Alternatives to benchmark: symbolic-only (interpretable, no learned transition); symbolic eligibility + learned rerank (recommended); JEPA proposes, symbolic re-derives eligibility (recall up, budget burnt pre-validation); separate predictor + verifier models (error independence, latency + correlated-data risk). RouteLLM shows cost/quality model routing, NOT skill/world-model/evidence routing; I-JEPA = vision, not language-agent transitions.
### Sec 4 candidate architectures
A symbolic Sparse-Savant baseline: typed graph, lexical/metadata/explicit-edge candidate gen, deterministic exploit + novelty reserve frontier, dependency expansion under fixed budget, sufficiency/evidence gate, typed execution, decomposed outcome trace; edge types REQUIRES/USES/PRODUCES/RELATED_TO/CONFLICTS_WITH/SUPERSEDES; outputs SUFFICIENT|EXPAND|SEARCH|ABSTAIN not one opaque confidence. = causal baseline; if it does not beat lexical top-k under matched budget, neither JEPA nor numogram is justified.
B A + bounded local JEPA: Score = Score_symbolic + delta*clip(JEPA), small delta, both logged, no JEPA authority over safeguards/identifiers/provenance/abstention. Failure hypotheses: OOD collapses on session/project/time splits; JEPA predicts usage fingerprints not transferable transition value; latency > saved retrieval; predictor confidence suppresses the reserve; symbolic features already saturate.
C numogram as typed projection: real runtime = conventional typed transition system (goal-ingest-parse-retrieve-predict-gate-execute-mutate-audit-memory); each transition: typed IO, preconditions + protected constraints, evidence refs, deterministic id, failure/recovery edges, immutable audit event.
### Printed pages (later doc section, paper copy): hypotheses H1-H6 + roadmap
H1 sparse freeform composition > fixed 3-stage pipeline at matched budget (falsifier: no gain or more failure amplification). H2 typed state improves long-session robustness + transition prediction (text-only matches while conversion errors rise). H3 distilled experts cut context/latency without lowering verified correctness (regression/fallback rate erases savings). H4 explicit fallback catches expert regressions (routing fails to detect degraded expert). H5 parameter-level composition beats small separate models under VRAM limit (switching/interference > weight reuse). H6 world model helps only transition-dependent tasks (universal activation better at matched compute).
Roadmap + stop gates: 1 instrument current router (route/outcome proofs, abstention labels, protected fields; stop: outcomes not classifiable). 2 externalize typed state (compaction/rehydration, provenance survival; stop: conversion loses constraints/rejected paths). 3 system-level specialists (verifier + small procedural modules; stop: orchestration overhead > gains). 4 adapters one at a time, no merging (stop: regression/switch cost > explicit skills). 5 world-transition learning narrow domain, shadow only (stop: poor calibration/systematic residuals). 6 reasoning specialization (search/debug/verify policies; stop: no gain over prompt/skill modules). 7 procedural distillation of verified high-volume skills only. 8 state-aware learned routing: rules stay eligibility + abstention boundary. 9 sparse cognitive composition only after simpler baselines win.
Do not build yet: automatic online weight updates; world-model predictions as facts; deleting SKILL.md after distillation; learned routing w/o no-match + fallback; unrestricted runtime adapter downloads; automatic adapter merging; raw reasoning histories as training truth; latent-only state; full cognitive MoE before system-level + LoRA baselines; mech-interp claims as proof of reversibility.
Conclusion: thesis narrower than "digital brain": explicit procedural knowledge bootstraps specialized neural behaviour, typed state + evidence = governance; reasoning modules decide how to investigate, world models predict transitions, procedural experts act; none universal, none authoritative. Strongest design = explicit authority + typed state + sparse capability routing + verified observation + reversible specialization. Confidence mu=0.84 for hybrid + staged roadmap; materially lower for latent world-model reliability + cognitive MoE. Flip: if disciplined explicit skills + typed-state compilation already saturate correctness/local efficiency, neural specialization = unnecessary complexity; if repeated verified workflows dominate cost and adapters hold under unseen-session eval, explicit-to-implicit migration = next layer.

## 8. Our position vs this doc (10-28)
- Roadmap step 1 = what route.log + usage.jsonl + judge_fresh already do, partly; missing: outcome proof beyond "read the skill", abstention label (router has none), protected fields.
- Doc's "prior JEPA improved seen-session, weak held-out" == AJAX .884 LOO vs .448 file split: same finding, independent source.
- A3 split slot (explore random / suspect) IS the r>0 reserve test; falsifier = the doc's: reserve hits ~ random + cost -> drop or shift-gated only. Pre-register: suspect_hit >= 2x random_hit at n>=30 each, else reject.
- A1 WEAKENED + A4 FALSIFIED = doc's "labels contaminated" (evidence quality row): route label != outcome. Router credit needs attribution (authority table last row).
- Nudger (A6) sits in the "reserve" + "abstention" slots; doc: prediction never authorizes -> nudger = advisory bits only, never a route override. Matches our <=3 lines design.
- B1 (clipped AVB) = JEPA chat's AVB reranker with small delta + logged separately; AVB currently NOT clipped/separately logged -> ask JEPA chat to log symbolic and AVB components side by side in route.log (branch, not opsd edit).
- Do-not-build list vs us: nightly AVB retrain from route_trace rewards = "automatic online weight updates" + "raw histories as training truth" -> flag to JEPA chat as don't-build #1 and #7. Held-out set must be frozen + hashed (judge_more.jsonl is not).
