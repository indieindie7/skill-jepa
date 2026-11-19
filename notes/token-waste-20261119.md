# Token waste drill, week 2026-11-12 .. 11-19
One user's Claude Code week: 844.5M effective units (input + 0.1 cache read + 2 cache write + 5 output;
weights, not dollars). Numbers from session logs only (`usage_today.py`, `waste_drill.py`); no transcript text read.

## Where it went
- Subagents 54% (460M). One decomp chat ran 177 Sonnet subagents in the week, 209M.
- Main chats ran on the top model (~240M of 254M main units in the three biggest chats).

## The mistakes, biggest first
1. Fat context on every request. 86-96% of main cost was spent at >100k context; subagents averaged
   107-135k. Every step re-sends the whole pile. Compactions per chat in one week: 49 / 198 / 101.
2. Subagent startup tax. Every general-purpose subagent starts at ~48.4k tokens (rules, memory index,
   skill list, tool schemas) before doing anything; November median first request 71k. A 4-tool custom
   agent starts at 16-26k, a fetch-only agent at 3.8k: the tool list is most of the tax.
3. Cache writes (2x price) = 41% of subagent cost (53% for whole-file "deep read" agents): every file
   read is written to cache, then re-sent each step.
4. One command per step: 55-64 requests per notes agent, ~1750 Bash calls, no batching.
5. Long single agents polling (406 requests, 60 Monitor calls); re-reading the same file up to 58x.

## Decision
Subagents banned (user: "Thats 1/3 of the context almost ok we banning subagents"). Work is done in the
main loop with grep + line-range reads and batched calls; cheap offload goes to local models.
Not checked: whether the output quality needed the spend.
