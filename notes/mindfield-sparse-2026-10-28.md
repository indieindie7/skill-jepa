# Mindfield "Sparse-Savant" design doc (user photos 10-28; mindfield_sparse_s..._190917.pdf, Copilot-made, work laptop)
Index: 1 verdict+flip (l.4-8) | 2 unresolved assumptions (l.10-21) | 3 minimal experiment (l.23-29) | 4 ablations (l.31-40) | 5 governed mutation (l.42-52) | 6 map to our ledger (l.54-66)

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
