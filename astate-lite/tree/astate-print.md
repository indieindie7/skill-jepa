# ASTATE print (the base spec), transcribed 2026-11-12 from the user's photos

Source photos: `~/.claude/uploads/<session>/cf4308fa-image.webp` (page 2), `f8379150-image.webp` + `6a44a376-image.webp` (page 1).
Transcribed by Claude from photos; wording checked against two shots of page 1, one shot of page 2.

## Index
1. Core definition (5 parts of the state)
2. State representation A_t = (M, B, R, I, P)
3. Learning and decimation operators
4. Optimization target (loss, keep / decimate rules)
5. Relearning memory (recovery record, 6 fields)
6. Safe decimation tests (5 probes)
7. Central hypothesis and falsifiers (6)

Title: **ASTATE. Learning through controlled forgetting.**
Banner: "A bounded state should preserve understanding, uncertainty, and the ability to relearn - not every historical detail."

## 1. Core definition
ASTATE is a bounded, evolving representation that improves its generative understanding of a domain by progressively replacing specific experiences with abstractions, exception boundaries, and procedures for reacquiring forgotten detail.
- Current understanding: models, concepts, rules, and causal structure.
- Boundaries: exceptions, uncertainty, scope, and known failure modes.
- Relearning memory: recognition cues and procedures for recovering specificity.
- Active cases: unresolved instances that cannot yet be safely abstracted.
- Provenance: pointers to external evidence for verification and recovery.

## 2. State representation
A_t = (M_t, B_t, R_t, I_t, P_t): M Models, B Boundaries, R Relearning, I Instances, P Provenance.
The raw evidence archive E_t sits outside ASTATE. ASTATE keeps compressed understanding plus routes back to evidence; it does not require the archive to remain in active context.

## 3. Learning and decimation
A_(t+1) = D_lambda( L(A_t, e_t) )
- L, the learning operator, integrates a new experience by revising models, boundaries, recovery procedures, active cases, and provenance.
- D_lambda, the decimation operator, forces the result under a state budget while minimizing future functional loss.
Key transformation: a detailed case is not merely deleted. Its useful residue is extracted first: model update, exception marker, recognition cue, recovery procedure, and evidence pointer.

## 4. Optimization target
Ordinary compression minimizes reconstruction error. ASTATE instead minimizes decision loss, relearning cost, verification loss, and active-state size.
Loss = alpha L_task + beta L_model + gamma L_boundary + delta C_recovery + eta L_provenance + kappa |A|

Keep a case when it:
- changes the current model
- defines an exception or boundary
- changes future decisions
- improves recovery or verification

Decimate a case when it:
- is explained by the current model
- adds no material boundary information
- can be cheaply reconstructed
- has a reliable provenance route

## 5. Relearning memory
ASTATE may forget an answer while retaining the mindset for reconstructing it. A recovery record should contain:
- Trigger: what situation signals that forgotten specificity is needed.
- Unknown: the exact fact, rule, or compatibility detail no longer active.
- Search space: likely evidence sources and variables to inspect.
- Procedure: an ordered sequence for reconstruction.
- Verification: tests that distinguish recovery from plausible guessing.
- Cost estimate: expected retrieval, compute, or reasoning effort.

## 6. Safe decimation tests
A deletion is acceptable only if the reduced state survives representative probes:
- Predictive: can it still anticipate representative outcomes?
- Counterexample: can it recognize where the general model may fail?
- Decision: does it choose approximately the same useful actions?
- Relearning: can it restore missing detail within an acceptable cost?
- Provenance: can it locate evidence needed to verify the reconstruction?

## 7. Central hypothesis and falsifiers
Hypothesis: A bounded system can improve its understanding while losing episodic detail, provided forgotten cases are converted into better models, boundary information, recognition cues, and validated procedures for reacquiring specificity.

The design is weakened if:
- episodic retrieval wins at comparable total cost;
- relearning fails without the original detailed case;
- decimation compounds incorrect abstractions;
- exception markers fail to prevent overgeneralization;
- safe-deletion evaluation costs as much as retention;
- recovery requires rereading the complete archive.

ASTATE = learning + epistemic-residue extraction + tested decimation + relearning memory
