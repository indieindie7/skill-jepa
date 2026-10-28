# Intelligence ∩ autonomy research → skill-JEPA router (2026-10-26)
Full report: Documents/Tools/llm/research/runs/intelligence-autonomy/report.md (Sonnet, snippet-verified only).

## Findings
1. Shared core: model of self + world, maintained goal, error-correcting feedback, choice under bounded compute.
2. Autonomy = whose goals (Kant, Frankfurt, SDT, autopoiesis); intelligence = fit + efficient learning (Legg-Hutter, Chollet, Simon, Gigerenzer).
3. Capability ⟂ autonomy (Feng et al. 2025, arXiv 2506.12469): 5 levels by user role, operator → observer.
4. Intelligent decision: cheap enough, fits environment, corrects on feedback, serves endorsed goal.
5. Ashby ultrastability: 2nd loop rewrites own rules when essential variables leave bounds.

## Router tie-ins (proposals)
- Abstain head = act/ask/refuse routing (arXiv 2607.13594, reject-option + learning-to-defer). Abstain when entropy over skill candidates high or expected value of info high; matches README "abstain head" idea + floor 0.25.
- Satisficing: BM25 baseline first, JEPA re-rank only when top-k margin small (spend compute in proportion to uncertainty).
- 2nd loop = SELFIMPROVE_PLAN data loop: route_trace reward +1/0/-1 changes the router itself (ultrastability), not just the pick.
- Autonomy dial per skill: log decisions, widen auto-route only where reward history is good (Anthropic Feb 2026: experienced users auto-approve ~20%→40%+ but interrupt more).
- Eval idea: measure abstain calibration (ask when wrong) separately from Hit@1 — capability vs autonomy are separate axes.

## Breaking the alignment loop (user + Claude, 2026-10-27) -> ground basis for context + world-model falsifier
User: "we are both aligning each other, the question then becomes how to break the loop and think outside of it." Router is meant to grow into a context/world-model FALSIFIER, not just a picker.
- Problem: user and Claude co-train each other (notes, skills, memory = shared extended mind, Clark & Chalmers). Closed loop drifts into mutual confirmation; the router trained on our own route_trace rewards inherits the same bias.
- Ashby 2nd loop: when the 1st loop settles (rewards stable, Hit@1 flat), change the rules, not the weights. Trigger = low surprise for too long, not only high error.
- Outside input neither picked: random/serendipity slot in retrieval (user's unplanned videos are this); sample sources outside the top-k by design (exploration budget).
- Fresh-context falsifier: a reviewer/model that never saw our assumptions predicts the outcome; disagreement with our world model = signal to log, not noise.
- Falsification question per decision: "what would make this wrong?" -> store the predicted refutation with the route; later check it (Popper: a context model is only useful if it can be shown wrong).
- Eval idea: falsifier score = how often the model's prediction is refuted by outside evidence and the update sticks; track separately from Hit@1 and abstain calibration.
