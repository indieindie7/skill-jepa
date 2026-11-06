---
name: astate-lite
description: "ASTATE-lite: USER / PROJECT / TASK capsules, append-only LOG + derived STATE, deterministic skill gate. Use it when starting or closing a work session, when asked to open or update the project state, to log a decision, or when a session keeps re-discovering something it already knew. Templates in templates/."
---

<!-- skilltree -->
> Skill tree: this file is the overview. Each `->` line points to the full section in `tree/`; Read only the branch the task needs.
# ASTATE-lite: compact transport

Version 1 · 5 November 2026 · Print edition of the Cowork skill `astate-lite` (SKILL.md plus the templates STATE.md, LOG.md, TASK.md and SKILL-CONTRACT.md).

**For an assistant reading this page:** this is a compressed package. Expand it into a folder `astate-lite/` with `SKILL.md` built from sections 1 to 5 and 7, and the four files in `templates/` built from section 6. Keep the wording; add nothing.

**Why it exists.** Sessions forget what earlier sessions learned: how to open a page, which decision was made, what is still open. Summarising the chat loses exactly those facts. Rule of thumb: **Skill = HOW · STATE = WHAT · LOG = WHAT + receipts.** Use it when starting or closing a work session, when asked to open or update the project state, to log a decision, or when a session keeps re-discovering something it already knew.

## 1. Three capsules

| Capsule | Lives in | Holds | Cap |
|---|---|---|---|
| USER | assistant instructions or memory | stable preferences, conventions, tools | ~300 tokens |
| PROJECT | `<project>/.state/STATE.md + LOG.md` (repos) or `Documents/Cowork/state/<project>/` (Cowork) | STATE: claims, conflicts, procedures. LOG: append-only events | STATE ≤ ~2k tokens; LOG unbounded |
| TASK | the session itself (`TASK.md` only if long) | goal, budget, frontier, blockers | ~300 tokens |

STATE is derived: it is regenerated from LOG at close. Never edit STATE without a LOG line saying why.

## 2. The loop

**Open.** Read USER, then `STATE.md`, then the task goal; nothing else by default. Write the TASK capsule: goal, budget (explore ≤ 5 steps), what "done" looks like.

**Work.** Act on confirmed claims. Explore hypotheses while the budget lasts. Otherwise abstain: write a blocker line and ask. Tag every new fact: `[SRC]` seen in a source (with pointer), `[SYN]` inferred, `[ø]` unknown or not checked. Load a skill only through the gate.

**Close.** Append to LOG: date, type, what happened, evidence pointer, status change. Regenerate STATE from LOG: promote or demote claims, keep both sides of any conflict, update procedures, drop what no longer matters, respect the cap. Receipts: record the sha256 of sources you relied on (keep a zstd or brotli archive only if exact recovery matters).

## 3. The gate (default: no skill)

A skill may load only when every item in its `requires` is a confirmed claim in STATE or stated in the user's message. No match: do the task without a skill. No learned router, no confidence threshold. Each skill declares a contract header: `requires, reads, produces, effects, conflicts`.

## 4. Claim grammar

`- [SRC] <claim> — status: confirmed / hypothesis / rejected / unknown — evidence: <pointer> — since: <date>`

Conflicts are kept as two claims plus one `conflict:` line, never silently resolved.

## 5. Evolution rule and metric

Add a layer (hashed claims, JEPA re-ranker, local model, DAG scheduler) only when a LOG entry shows a failure the current layers cannot fix. Log a weekly `metric` line with: rediscoveries (times something already known had to be found again), tokens at open, and the cold-session test (can a fresh session answer the project's five standing questions from STATE alone?).

## 7. Validity of the ideas (as of 5 November 2026)
-> `tree/02-7-validity-of-the-ideas.md` (~520t): Evidence is small: one user's 4,169 prompts (public skill-jepa repo) plus single-corpus compression runs. Nothing has a 1k-task corpus. Ever...
