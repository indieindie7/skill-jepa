# Gemma 8083 summaries of the scanned reports (2026-10-24)

## Pages 1-7
*   **Architecture Proposal:** A freeform modular cognitive architecture that treats Reasoning, World Models, and Expert Procedural capabilities as three sparsely activated families rather than a linear pipeline.
*   **Dynamic Controller:** A central controller evaluates goals, uncertainty, evidence requirements, and resource budgets to select both a **representation** (text, symbolic graph, or latent state) and a **model composition** (specialists, adapters, or sparse experts).
*   **Skill Lifecycle:** Explicit skills (`SKILL.md`) serve as the canonical authority and "regression oracle." The lifecycle moves from execution $\rightarrow$ verified traces $\rightarrow$ shadow experts $\rightarrow$ production specialization.
*   **Hybrid Near-Term Strategy:** Keep explicit skills and typed state; add a system-level world-model predictor for transition reasoning; use isolated reasoning and procedural LoRAs; retain explicit abstention and external verification.
*   **Core Thesis:** Explicit knowledge bootstraps specialization; typed state governs composition; neural experts optimize repeated behavior; and observations (not predictions) update factual state.
*   **Primary Failure Metric:** The architecture is considered unjustified if sparse modular composition does not outperform a carefully instrumented single-model baseline in quality, latency, VRAM, and auditability.
*   **Role Contracts:** Modules are classified by "role contracts" rather than identity. A single model (e.g., an LLM) can act as a reasoning model in one context and a procedural expert in another depending on the active adapter or task.
*   **Sparse Activation Examples:**
    *   *Exact workflow:* Procedural skill only.
    *   *Novel diagnosis:* Reasoning model + evidence tools.
    *   *Deployment change:* Procedural expert first; world model only if side effects are predicted.
    *   *Long-horizon intervention:* Reasoning + world simulation + procedural expert.
*   **State Schema:** A shared typed bus ($\chi$) coordinates projections into three views: machine state, verification certificate, and reader view.
*   **Stop Gates & Governance:**
    *   **Abstention Primitive:** Explicit mechanism for the system to refuse a task.
    *   **Route Proofs:** Certificates of compliance for the selected composition.
    *   **Verified Gates:** Observations must pass through a "verified gate" before updating the factual state to prevent latent predictions from becoming facts.
    *   **Usage/Outcome Records:** Instrumentation for training-data admission gates and auditability.

## Pages 8-14
*   **Architecture Overview**: Proposes a "Freeform Cognitive Architecture" where a State Bus coordinates sparse activations across heterogeneous world models, procedural capabilities, and observation filters to prevent latent predictions from "beaming" (hallucinating) facts.
*   **State Schema**: Defines a structured state including:
    *   **Symbolic State**: IDs, source links, authority, and uncertainty types.
    *   **Latent State**: Compact predictions, learned dynamics, and similarity scores.
    *   **Protected State**: Exact quotations, commands, safeguards, and errors.
*   **Namespace Isolation**: Strict separation between Facts, Observations, and Predictions; latent predictions generate "candidate hypotheses" that require verified observations to become "confirmed facts."
*   **Dynamic Composition Controller**: A decision engine that selects module sets ($M$), representations ($R$), and allocations ($B$) by maximizing goal progress while minimizing cost, risk, and unverifiability.
*   **Switching Rules**: Specific triggers for the controller, including:
    *   **Escalation**: Triggered by repeated failures, multiple plausible decompositions, or prediction disagreement.
    *   **Abstraction**: Switching to "reasoning search" or "world-model simulation" based on confidence.
    *   **Resource Management**: Compiling smaller states or unloading adapters under budget pressure.
*   **Routing Critique**: Notes that current "RouteLLM" approaches optimize cost/quality via preference labels but fail to route between distinct procedural, reasoning, and world capabilities.
*   **Representation Stack**: Proposes a 5-point conversion protocol:
    1. Text + True State (schema-constrained extraction).
    2. Verified State $\rightarrow$ Text (bounded compilation).
    3. Symbolic State + Latent (versioned projections).
    4. Latent $\rightarrow$ Symbolic Candidate (top-k hypotheses + uncertainty).
    5. Observation + Reconciliation (residual/error tracking).
*   **World-Model Variants**: Recommends three implementation paths:
    *   **Explicit Transition Rules**: For deterministic tools/permissions.
    *   **Domain Adapters**: Shared encoders with domain-specific LoRAs.
    *   **Separate Predictive Models**: For maximum isolation and calibration.
*   **Stop Gates & Constraints**: The controller must operate subject to "authority, protection, abstention, and resource constraints."
*   **Verification Mechanism**: Observations are the only mechanism to "update" the state; predictions are prohibited from direct factual write-back.
*   **Metrics/Goals**: Success is measured by "Goal Progress" vs. "Cost" and "Risk" trade-offs, with a focus on "Verifiability" as a primary constraint.

## Pages 15-22
**Concrete Proposals & Architectures**
*   **Hybrid Architecture:** Recommended approach combining shared base models with dynamic LoRA adapters and explicit skills/typed states.
*   **Modular Decomposition:** Separate services for high isolation, or "Cognitive Sparse MoE" for joint training with sparse parameter activation.
*   **Local Hardware Policy:** Prioritize "quality per retained token/millisecond" over model size; keep one quantized base resident, cache small adapters in CPU RAM, and load reasoning/procedural adapters on demand.
*   **State Management:** Compile "typed state" into bounded contexts; externalize state to support compaction/rehydration and provenance survival.
*   **World Model:** Run a small, separate world model only when "transition reasoning" is triggered.

**Mechanisms**
*   **Provenance Certificates:** Each expert must ship with a certificate including base model, adapter hash, skill revision, dataset digests, and known failures.
*   **Retrieval & Routing:** Use BM25/Lexical-plus-semantic design (Qdrant) with rank fusion; implement "retrieve before generating" and "authority fallback" mechanisms.
*   **Training Strategy:** Maintain separate corpora for procedural execution, reasoning trajectories, and negative traces; use session/project-grouped splits to prevent data leakage.
*   **Adapter Management:** Use PEFT for in-place LoRA swapping with compatibility constraints; avoid merging adapters to maintain auditability.

**Metrics**
*   **Core Performance:** Module-selection accuracy, unnecessary activation, transition calibration, and counterfactual ranking.
*   **System Overhead:** TTFI (Time to First Instruction) latency, peak VRAM, adapter-switch cost, and energy consumption.
*   **Reliability:** Conversion loss, provenance survival, fallback recovery interference, and unseen-session generalization.

**Stop Gates (Roadmap)**
*   **Router:** Stop if outcomes cannot be classified reliably.
*   **Typed State:** Stop if repeated conversion loses constraints or rejected paths.
*   **Specialists:** Stop if orchestration overhead exceeds gains.
*   **Adapters:** Stop if regression or switching cost exceeds explicit skill benefits.
*   **World Model:** Stop if poor calibration or systematic residuals occur.
*   **Reasoning:** Stop if no gain over prompt/skill modules under matched compute.
*   **Distillation:** Stop if regression rate or fallback frequency erases savings.

## Pages 23-29
**Concrete Proposals & Mechanisms**
*   **Architecture Thesis:** A "governance layer" model using explicit procedural knowledge to bootstrap neural behavior, combining explicit authority, typed state, sparse capability routing, and verified observations.
*   **Router Evolution:** A staged progression from a lexical control baseline $\rightarrow$ outcome instrumentation $\rightarrow$ fast-reading retrieval $\rightarrow$ typed context compilation $\rightarrow$ hybrid/graph retrieval $\rightarrow$ sufficiency/abstention.
*   **Fast-Reading Retrieval:** Shifts from "rank skill, inject body" to "rank headers $\rightarrow$ inspect topology $\rightarrow$ rank sections $\rightarrow$ expand only relevant sections" to manage token budgets.
*   **Typed Context Compiler:** Replaces flat concatenation with structured packets. Uses "Protection Classes" to define deterministic retention (e.g., *Preserve/Reference* vs. *Compressible* vs. *Disposable*).
*   **Hybrid Retrieval:** Independent generation of BM25 and dense lists, combined via reproducible fusion rules (e.g., Reciprocal Rank Fusion) to avoid learned-fusion confounders.
*   **Explicit Skill Graph:** Bounded retrieval using explicit edges (REQUIRES, USES, PRODUCES, CONFLICTS_WITH) to prevent recursive context loading.
*   **Sufficiency/Abstention:** Replaces single confidence scores with multi-state logic: SUFFICIENT, EXPAND, SEARCH, and ABSTAIN.

**Metrics & Data Management**
*   **Core Metrics:** Module-selection accuracy, unnecessary activation, transition calibration, counterfactual ranking, conversion loss, provenance survival, fallback recovery interference, unseen-session generalization, TTFI latency, tokens, peak VRAM, and adapter-switch cost.
*   **Data Corpora:** Separate datasets for procedural execution, reasoning trajectories, state transitions, negative traces, routing decisions, and final outcomes.
*   **Logging Requirements:** Every route must record query hash, router/catalog revisions, candidate scores, selected skills, abstentions, injected tokens, and multi-stage latencies.

**Stop Gates & Constraints**
*   **Verification Gates:** Only admit traces after verifier agreement and source-policy checks; raw reasoning histories are not canonical memory.
*   **Implementation "Do Not Build" List:** No automatic online weight updates, no world-model predictions as facts, no deletion of skills after distillation, no learned routing without no-match/fallback, and no unrestricted runtime adapter downloads.
*   **Validation Gates:** Context compilation must pass checks for instruction survival, conflict retention, provenance coverage, and recoverability.

## Pages 30-34
*   **Outcome-Based Learning (Phase 7):** Training examples must be weighted by validated route, execution, and task outcomes; transcript appearance alone is insufficient for supervision.
*   **JEPA Experimental Branch:** Predicts next-skill representations using a weighted formula: $S = \alpha \cdot \text{BM25} + \beta \cdot \text{dense} + \gamma \cdot \text{graph} + \delta \cdot \text{JEPA}$. JEPA is restricted to routing corrections and has no authority over safeguards or provenance.
*   **Benchmark Families:** Testing covers Exact Routing (names/aliases), Semantic Recall (paraphrases/vocabulary mismatch), No-match (abstention/false activation), Composite (bounded prerequisites), and Multilingual/Long-session scenarios.
*   **Long-Session & Survival Metrics:** Focus on "behavioral survival" (obedience, contradiction retention, evidence recovery) rather than simple summary similarity. Includes tracking "protected survival" (preserving accepted decisions/rejected paths).
*   **Resource Pareto Frontier:** Evaluation must balance Conceptual Quality against Resource Cost (VRAM, latency, input tokens, KV cache, and CPU RAM) to find the optimal performance-per-unit-cost envelope.
*   **Compaction Strategies (C0-C6):** Roadmap includes testing raw history, sliding windows, generic summaries, query-aware summaries, typed STATE, and typed STATE + hierarchical memory.
*   **Ablation Plan:** Systematic testing of input variables: (1) Prompt $\rightarrow$ Skill, (2) Prompt + Prev Skill $\rightarrow$ Skill, (3) Prompt + STATE $\rightarrow$ Skill, (4) Prompt + STATE + Graph $\rightarrow$ Skill, and (5) Outcome Weighting.
*   **Dataset Split Strategy:** Strict versioning to prevent "fingerprint learning." Validation/Test sets must use unseen sessions and projects; random splits are rejected if they fail to generalize to temporal/project-grouped data.
*   **Retrieval Components:** Evaluation of BM25 vs. Dense, and Fast-Reading variants (Full Body vs. Topology-based section retrieval) to determine if section-level retrieval reduces context cost without degrading success.
*   **Roadmap Stop Gates:** A critical gate exists before adding semantic, graph, or JEPA branches: the "Baseline vs. Fast Reading vs. Fast Reading + Typed Compaction" ablation must be completed across all benchmark families.
*   **Failure Conditions:** Deliberately unrelated prompts must return a "no-match" (failure to return a skill is a success); prompts requiring specific dependencies (A requires B) must trigger correct routing.
*   **Implementation Roadmap:** Prioritizes building the benchmark harness, outcome trace, fast-reading section retrieval, and typed STATE compiler as the first milestones.

## Cross-report takeaways
**Proposals Not Currently Implemented**

1. **Fast-Reading Retrieval:** Rank headers/topology first, then expand only relevant sections to manage token budgets (Metric: Context Cost vs. Success).
2. **Typed Context Compiler:** Replace flat concatenation with structured packets using "Protection Classes" (Preserve/Reference vs. Disposable) (Metric: Instruction Survival).
3. **Explicit Skill Graph:** Use bounded retrieval with edges (REQUIRES, USES, CONFLICTS_WITH) to prevent recursive loading (Metric: Unnecessary Activation).
4. **Abstention Primitive:** Explicit multi-state logic (SUFFICIENT, EXPAND, SEARCH, ABSTAIN) instead of just high/low confidence (Metric: No-match Accuracy).
5. **Provenance Certificates:** Ship skills with base model, adapter hash, revision, and known failure logs (Metric: Provenance Survival).
6. **Verified Gates:** Ensure only observations (not latent predictions) can update the factual state (Metric: Conversion Loss).
7. **Resource Pareto Frontier:** Explicitly optimize for "quality per retained token/millisecond" (Metric: TTFI Latency / VRAM / Energy).
8. **Outcome-Based Weighting:** Weight training examples by validated execution outcomes rather than transcript appearance (Metric: Goal Progress vs. Cost).

**Ranked Next Experiments (One-Person Local Setup)**

1. **Fast-Reading Retrieval:** Test "Rank Headers $\rightarrow$ Expand Section" vs. "Rank Skill $\rightarrow$ Inject Body" (Pass/Fail: Context Cost reduction with $\geq$ Hit@1).
2. **Abstention Primitive:** Implement a "No-match" trigger for deliberately unrelated prompts (Pass/Fail: No-match Accuracy).
3. **Typed Context Compiler:** Test "Structured Packet" vs. "Flat Concatenation" for long-horizon tasks (Pass/Fail: Instruction Survival).
4. **Outcome Weighting:** Weight router training data by successful task completion vs. mere prompt similarity (Pass/Fail: Goal Progress).
5. **Skill Graph:** Implement a simple "REQUIRES" edge between two skills to trigger sequential loading (Pass/Fail: Composite Routing Accuracy).
